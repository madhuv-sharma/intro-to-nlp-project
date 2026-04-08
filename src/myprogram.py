#!/usr/bin/env python
"""
Unified multilingual character-level Transformer LM
for the Interstellar Autocomplete Challenge.

Architecture
------------
- Single CharTransformerLM for ALL languages (language conditioning via embedding)
- Token embedding + learned positional embedding + language embedding → summed
- 4-layer causal (decoder-only) Transformer, pre-norm style (GPT-2 variant)
- Weight-tied input/output embeddings (saves ~3 M params, improves perplexity)
- Mixed-precision (fp16) training and inference
- CosineAnnealingLR

Why unified model vs. 10 separate GRUs
---------------------------------------
- Single model handles mixed-language completion naturally
- Shared character patterns (digits, punctuation, Latin base) transfer across langs
- One checkpoint; single GPU forward pass at inference (no bucketing overhead)
- Compatible with Qwen 3.5 0.8B translations, which are much cleaner than NLLB —
  fewer data-cleaning hacks needed

Usage
-----
  python myprogram.py train --work_dir ../work --train_dir ../data/train
  python myprogram.py test  --work_dir ../work --test_data ../kaggle-data/test.csv \\
                            --test_output ../submission.csv
"""

import argparse
import math
import os
import random
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

import pandas as pd
import torch
import torch.nn as nn

# ──────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────
SEQ_LEN = 128
D_MODEL = 512
N_HEADS = 8  # head_dim = 512 / 8 = 64
N_LAYERS = 4
DROPOUT = 0.15
EPOCHS = 20
BATCH_SIZE = 512
LR = 5e-4  # lower than GRU — transformers are sensitive to high LR
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
CHAR_DROPOUT = 0.03

CJK_LANGS = {"zh", "ja", "ko"}
LATIN_STRIDE_DIV = 2  # stride = seq_len // 2
CJK_STRIDE_DIV = 6  # stride = seq_len // 6 → ~3× more sequences for data-scarce CJK


# ──────────────────────────────────────────────
# Special token IDs
# ──────────────────────────────────────────────
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_IDS = {PAD_ID, UNK_ID, BOS_ID, EOS_ID}

# Inference batch size — larger than training since there are no gradients/optimizer state.
INF_BATCH = 2048


# ══════════════════════════════════════════════
# Vocabulary
# ══════════════════════════════════════════════
class Vocab:
    """Character-level vocabulary built from raw text."""

    _SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self):
        self._c2i = {t: i for i, t in enumerate(self._SPECIALS)}
        self._i2c = {i: t for i, t in enumerate(self._SPECIALS)}

    def build(self, lines, min_count: int = 1) -> "Vocab":
        counts = Counter(ch for line in lines for ch in line)
        for ch, cnt in sorted(counts.items()):  # sorted → deterministic vocab
            if cnt >= min_count and ch not in self._c2i:
                idx = len(self._c2i)
                self._c2i[ch] = idx
                self._i2c[idx] = ch
        return self

    def encode(self, ch: str) -> int:
        return self._c2i.get(ch, UNK_ID)

    def decode(self, idx: int) -> str:
        return self._i2c.get(idx, "?")

    def __len__(self) -> int:
        return len(self._c2i)


# ══════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════
class CharTransformerLM(nn.Module):
    """
    Unified multilingual character-level causal Transformer LM.

    A language embedding is added to every token position so one model
    handles all languages while keeping per-language specialization.

    Input : token ids (B, T), language ids (B,), optional pad mask (B, T)
    Output: per-position logits (B, T, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        n_langs: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
        max_seq: int = SEQ_LEN,
    ):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(max_seq + 2, d_model)
        self.lang_embed = nn.Embedding(n_langs, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm (more stable for deep transformers)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: share token embedding and output projection weights.
        self.fc.weight = self.tok_embed.weight
        self.d_model = d_model
        # Pre-build causal mask as a buffer so it lives on the model's device
        # and is never reallocated during training or inference.
        causal = torch.triu(
            torch.ones(max_seq + 2, max_seq + 2, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("_causal_buf", causal)

    def forward(
        self,
        x: torch.Tensor,  # (B, T)
        lang_ids: torch.Tensor,  # (B,)
        pad_mask: Optional[torch.Tensor] = None,  # (B, T) True = padding
    ) -> torch.Tensor:
        T = x.shape[1]
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        h = (
            self.tok_embed(x)  # (B, T, D)
            + self.pos_embed(pos)  # (1, T, D)
            + self.lang_embed(lang_ids).unsqueeze(1)  # (B, 1, D) broadcast over T
        )
        out = self.transformer(
            h,
            mask=self._causal_buf[:T, :T],  # slice cached mask — no allocation
            src_key_padding_mask=pad_mask,
        )
        return self.fc(out)  # (B, T, V)


# ══════════════════════════════════════════════
# Language identification helpers
# ══════════════════════════════════════════════
def detect_script(text: str):
    """
    Return a language code when text clearly uses a non-Latin script,
    otherwise return None (caller does bigram-profile LID for Latin langs).

    ja/zh disambiguation: Japanese uses hiragana/katakana exclusively, but also
    uses CJK ideographs (kanji) shared with Chinese. Previous code counted CJK
    as zh, so a Japanese sentence with only kanji (no kana) was misidentified
    as zh. Fix: any kana presence → ja immediately; CJK-only → zh.
    """
    counts = {"ru": 0, "hi": 0, "ar": 0, "ko": 0, "zh": 0}
    ja_kana = 0
    cjk = 0
    for ch in text:
        name = unicodedata.name(ch, "")
        if "CYRILLIC" in name:
            counts["ru"] += 1
        elif "DEVANAGARI" in name:
            counts["hi"] += 1
        elif "ARABIC" in name:
            counts["ar"] += 1
        elif "HANGUL" in name:
            counts["ko"] += 1
        elif "HIRAGANA" in name or "KATAKANA" in name:
            ja_kana += 1  # exclusively Japanese — any presence confirms ja
        elif "CJK UNIFIED" in name or "CJK COMPATIBILITY" in name:
            cjk += 1  # shared between zh and ja; resolved below

    # Kana is unambiguously Japanese
    if ja_kana > 0:
        return "ja"
    # CJK without kana is Chinese
    if cjk > 0:
        counts["zh"] = cjk

    if max(counts.values()) == 0:
        return None
    return max(counts, key=counts.get)


def build_bigram_profiles(lang_lines: dict) -> dict:
    """Build character bigram log-prob profiles for Latin-script LID fallback."""
    profiles = {}
    for lang, lines in lang_lines.items():
        counts = Counter()
        for line in lines:
            for a, b in zip(line, line[1:]):
                counts[(a, b)] += 1
        total = sum(counts.values()) + len(counts)
        profiles[lang] = {bg: math.log((cnt + 1) / total) for bg, cnt in counts.items()}
    return profiles


# Characters that are near-uniquely diagnostic for a specific Latin language.
_DIACRITIC_LANG: dict = {}
for _ch in "äöüÄÖÜß":
    _DIACRITIC_LANG[_ch] = "de"
for _ch in "éêëàâçîïôùûœæÉÊËÀÂÇÎÏÔÙÛŒÆ":
    _DIACRITIC_LANG[_ch] = "fr"
for _ch in "àèìíîòóùúÀÈÌÍÎÒÓÙÚ":
    _DIACRITIC_LANG[_ch] = "it"


def score_bigram(profile: dict, text: str) -> float:
    """Average bigram log-prob; unknown bigrams get a small penalty."""
    if len(text) < 2:
        return 0.0
    unk_lp = math.log(1e-6)
    return sum(profile.get((a, b), unk_lp) for a, b in zip(text, text[1:])) / (
        len(text) - 1
    )


def lid_latin(profiles: dict, latin_langs: list, text: str) -> str:
    """
    Identify which Latin language a context belongs to.

    1. Scan for language-diagnostic diacritic characters — instant & ~100% reliable.
    2. Fall back to bigram log-prob scoring for plain-ASCII contexts.
    """
    lang_votes: Counter = Counter()
    for ch in text:
        lang = _DIACRITIC_LANG.get(ch)
        if lang and lang in latin_langs:
            lang_votes[lang] += 1

    if lang_votes:
        return lang_votes.most_common(1)[0][0]

    if len(text) < 2:
        return "en" if "en" in latin_langs else latin_langs[0]

    return max(latin_langs, key=lambda l: score_bigram(profiles[l], text))


# ══════════════════════════════════════════════
# Encoding utilities
# ══════════════════════════════════════════════
def _encode_context(vocab: Vocab, text: str) -> list:
    """Prepend BOS, encode characters, clip to SEQ_LEN."""
    ids = [BOS_ID] + [vocab.encode(c) for c in text]
    ids = ids[-SEQ_LEN:]
    return ids if ids else [BOS_ID]


# ══════════════════════════════════════════════
# Training data cleaning
# ══════════════════════════════════════════════
def _clean_lines(lines: list, lang: str) -> list:
    """
    Remove low-quality lines from translated training data.

    Originally tuned for NLLB artifacts; thresholds relaxed for Qwen 3.5 0.8B
    which translates proper nouns and technical terms much more faithfully.
    Remaining checks are generic quality guards.
    """
    cleaned = []
    non_latin_langs = {"ar", "hi", "ko", "ja", "ru", "zh"}

    for line in lines:
        if not line.strip():
            continue

        # For non-Latin langs: drop lines that are almost entirely ASCII.
        # Threshold raised to 0.95 — Qwen rarely leaves text untranslated; keeping
        # mixed-script lines (e.g. zh/ja text with English product names) improves coverage.
        if lang in non_latin_langs:
            ascii_ratio = sum(c.isascii() for c in line) / max(len(line), 1)
            if ascii_ratio > 0.95 and len(line.strip()) > 8:
                continue

        # For zh: drop lines where Japanese kana makes up >30% of characters.
        # A small amount of kana in zh text is legitimate (loanwords, mixed docs);
        # only drop clearly mis-translated lines.
        if lang == "zh":
            ja_count = sum(
                1 for c in line
                if "HIRAGANA" in unicodedata.name(c, "")
                or "KATAKANA" in unicodedata.name(c, "")
            )
            if ja_count / max(len(line), 1) > 0.3:
                continue

        cleaned.append(line)

    return cleaned


def _make_sequences(token_ids: list, seq_len: int, stride_div: int):
    """
    Yield (input, target) integer-list pairs of length seq_len.
    stride = seq_len // stride_div; smaller stride → more sequences (used for CJK).
    """
    stride = max(seq_len // stride_div, 1)
    for i in range(0, len(token_ids) - seq_len, stride):
        chunk = token_ids[i : i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        yield chunk[:-1], chunk[1:]


# ══════════════════════════════════════════════
# Unified training
# ══════════════════════════════════════════════
def _train_unified(
    inputs_arr: "np.ndarray",  # (N, SEQ_LEN) int16
    targets_arr: "np.ndarray",  # (N, SEQ_LEN) int16
    lang_ids_arr: "np.ndarray",  # (N,)         int16
    vocab: Vocab,
    n_langs: int,
    device: torch.device,
) -> "CharTransformerLM":
    """Train a single CharTransformerLM on mixed multilingual sequences."""
    vocab_size = len(vocab)
    n_total = len(inputs_arr)
    model = CharTransformerLM(vocab_size, n_langs).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,}  vocab: {vocab_size}  langs: {n_langs}")

    # Pre-load all data to the target device once.
    # On GPU: eliminates per-batch CPU→GPU transfer and numpy fancy-indexing overhead.
    # Stored as int32 (half the footprint of int64); cast to long() per batch is cheap.
    # GPU VRAM cost: ~2 × N × SEQ_LEN × 4 B ≈ 670 MB for 655 K seqs — fits T4/A100.
    print("  Pre-loading data to device …")
    inputs_dev = torch.from_numpy(inputs_arr.astype(np.int32)).to(device)
    targets_dev = torch.from_numpy(targets_arr.astype(np.int32)).to(device)
    lang_dev = torch.from_numpy(lang_ids_arr.astype(np.int32)).to(device)

    # torch.compile fuses kernels and eliminates Python dispatch overhead.
    # Falls back silently if the environment doesn't support it.
    if device.type == "cuda":
        try:
            model = torch.compile(model)
            print("  torch.compile: enabled")
        except Exception:
            print("  torch.compile: unavailable, skipping")

    # fused=True runs AdamW as a single CUDA kernel (vs. one kernel per parameter).
    fused_ok = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=fused_ok
    )
    # 2-epoch linear warmup then cosine decay — transformers benefit from
    # a gradual ramp-up before the full learning rate is applied.
    _warmup_epochs = 2
    _warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=_warmup_epochs
    )
    _cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(EPOCHS - _warmup_epochs, 1), eta_min=LR * 0.05
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[_warmup, _cosine], milestones=[_warmup_epochs]
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # Accumulate loss as a GPU tensor — one CPU/GPU sync per epoch instead of per batch.
    running_loss = torch.zeros(1, device=device)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_total, device=device)
        running_loss.zero_()
        n_batches = 0

        for i in range(0, n_total, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            inputs = inputs_dev[idx].long()
            targets = targets_dev[idx].long()
            lang_ids = lang_dev[idx].long()

            # Character dropout: randomly mask 3% of input tokens to UNK
            if CHAR_DROPOUT > 0:
                mask = torch.rand_like(inputs, dtype=torch.float) < CHAR_DROPOUT
                mask &= (inputs != PAD_ID) & (inputs != BOS_ID)
                inputs = inputs.masked_fill(mask, UNK_ID)

            # Training sequences are always full-length (no padding from _make_sequences),
            # so pad_mask=None is correct and avoids an unnecessary boolean allocation.
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inputs, lang_ids)  # (B, T, V)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach()  # stays on GPU — no sync
            n_batches += 1

        avg = running_loss.item() / max(n_batches, 1)  # single sync per epoch
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch:2d}/{EPOCHS}  loss={avg:.4f}  lr={cur_lr:.2e}")

    return model


# ══════════════════════════════════════════════
# Train mode
# ══════════════════════════════════════════════
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = Path(args.train_dir)
    total_start = time.perf_counter()

    print(f"Device : {device}")
    print(
        f"Transformer  d_model={D_MODEL}  n_heads={N_HEADS}  n_layers={N_LAYERS}  "
        f"seq_len={SEQ_LEN}  epochs={EPOCHS}  batch={BATCH_SIZE}  fp16={device.type == 'cuda'}"
    )

    # ── Phase 1: Read & clean all language files ──────────────────────
    lang_lines: dict = {}
    latin_train_lines: dict = {}

    for file in sorted(train_dir.glob("*.txt")):
        lang = file.stem
        lines = []
        with file.open("r", encoding="utf-8") as f:
            for raw in f:
                line = unicodedata.normalize("NFC", raw.rstrip("\r\n").strip('"'))
                if line:
                    lines.append(line)
        lines = _clean_lines(lines, lang)
        lang_lines[lang] = lines
        if lang not in CJK_LANGS and lang not in {"ru", "hi", "ar"}:
            latin_train_lines[lang] = lines
        print(f"  [{lang}] {len(lines):,} lines after cleaning")

    # ── Phase 2: Build unified vocabulary ────────────────────────────
    print("\nBuilding unified vocabulary …")
    all_lines = [line for lines in lang_lines.values() for line in lines]
    # min_count=2: drop singleton characters (OCR noise, rare Unicode glyphs)
    vocab = Vocab().build(all_lines, min_count=2)
    print(f"  Unified vocab size: {len(vocab)}")

    # Language ID mapping (sorted for determinism)
    languages = sorted(lang_lines.keys())
    lang2id = {lang: i for i, lang in enumerate(languages)}
    n_langs = len(languages)
    print(f"  Languages ({n_langs}): {languages}")

    # ── Phase 3: Encode all data → numpy arrays (int16 saves ~10× memory vs Python lists)
    print("\nEncoding sequences …")
    inp_buf: list = []
    tgt_buf: list = []
    lid_buf: list = []

    for lang, lines in lang_lines.items():
        lang_id = lang2id[lang]
        stride_div = CJK_STRIDE_DIV if lang in CJK_LANGS else LATIN_STRIDE_DIV

        flat_ids: list = []
        for line in lines:
            flat_ids.append(BOS_ID)
            flat_ids.extend(vocab.encode(c) for c in line)
            flat_ids.append(EOS_ID)

        n_before = len(inp_buf)
        for inp, tgt in _make_sequences(flat_ids, SEQ_LEN, stride_div):
            inp_buf.append(inp)
            tgt_buf.append(tgt)
            lid_buf.append(lang_id)
        print(f"  [{lang}] {len(inp_buf) - n_before:,} sequences")

    # int16 is sufficient: max vocab id < 32 767 for all expected languages
    inputs_arr = np.array(inp_buf, dtype=np.int16)  # (N, SEQ_LEN)
    targets_arr = np.array(tgt_buf, dtype=np.int16)  # (N, SEQ_LEN)
    lang_ids_arr = np.array(lid_buf, dtype=np.int16)  # (N,)
    del inp_buf, tgt_buf, lid_buf  # free intermediate Python lists

    print(
        f"\nTotal sequences: {len(inputs_arr):,}  "
        f"(arrays: {inputs_arr.nbytes / 1024**2:.0f} MB)"
    )

    # ── Phase 4: Train unified model ─────────────────────────────────
    print("\n── Training unified CharTransformerLM ───────────────────────────")
    t0 = time.perf_counter()
    model = _train_unified(
        inputs_arr, targets_arr, lang_ids_arr, vocab, n_langs, device
    )
    model.eval()
    print(f"  Training time: {time.perf_counter() - t0:.1f}s")

    # ── Phase 5: Build Latin bigram LID profiles ──────────────────────
    print("\nBuilding Latin bigram LID profiles …")
    bigram_profiles = build_bigram_profiles(latin_train_lines)
    print(f"  Profiles built for: {sorted(bigram_profiles)}")

    # ── Save checkpoint ───────────────────────────────────────────────
    checkpoint = {
        "vocab": vocab,
        "vocab_size": len(vocab),
        "lang2id": lang2id,
        "n_langs": n_langs,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        # Unwrap torch.compile wrapper before saving so load_state_dict works cleanly.
        "state_dict": {
            k: v.cpu()
            for k, v in getattr(model, "_orig_mod", model).state_dict().items()
        },
        "__bigram_profiles__": bigram_profiles,
    }

    os.makedirs(args.work_dir, exist_ok=True)
    save_path = os.path.join(args.work_dir, "model.pt")
    torch.save(checkpoint, save_path)

    print(
        f"\n✓ Training complete — {time.perf_counter() - total_start:.1f}s  |  saved to {save_path}"
    )


# ══════════════════════════════════════════════
# Batched inference helpers
# ══════════════════════════════════════════════
def _batch_predict_top3(
    model: CharTransformerLM,
    vocab: Vocab,
    lang_ids_list: list,  # one lang_id per example (mixed-language batches supported)
    texts: list,
    device: torch.device,
    decode_arr: list,  # pre-built list: index → character string
) -> list:
    """
    Predict top-3 next characters for a list of texts in one padded forward pass.
    Accepts mixed languages in a single batch via per-example lang_ids.
    Returns a list of 3-char strings aligned with `texts`.
    """
    encoded = [_encode_context(vocab, t) for t in texts]
    lengths = [len(ids) for ids in encoded]
    max_len = max(lengths)
    B = len(texts)

    # Numpy pre-allocation is faster than Python list-of-lists for torch.tensor()
    x_np = np.zeros((B, max_len), dtype=np.int32)
    for i, ids in enumerate(encoded):
        x_np[i, : len(ids)] = ids
    x = torch.from_numpy(x_np).long().to(device)  # (B, T)
    lang_ids = torch.tensor(lang_ids_list, dtype=torch.long, device=device)
    pad_mask = x == PAD_ID  # (B, T)

    use_amp = device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=use_amp):
        logits = model(x, lang_ids, pad_mask)  # (B, T, V)

    idx_tensor = torch.tensor([l - 1 for l in lengths], dtype=torch.long, device=device)
    last_logits = logits[torch.arange(B, device=device), idx_tensor, :]  # (B, V)

    for sid in SPECIAL_IDS:
        if sid < last_logits.shape[1]:
            last_logits[:, sid] = float("-inf")

    # Temperature < 1.0 sharpens the distribution, concentrating probability mass
    # on the most likely next characters and improving top-3 hit rate.
    top3 = torch.topk(last_logits / 0.8, 3, dim=-1).indices.cpu().numpy()  # (B, 3)

    # decode_arr avoids a dict lookup per call — direct list indexing
    return ["".join(decode_arr[top3[i, j]] for j in range(3)) for i in range(B)]


# ══════════════════════════════════════════════
# Test mode
# ══════════════════════════════════════════════
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()

    print(f"Device: {device}")
    print("Loading checkpoint …")

    checkpoint = torch.load(
        os.path.join(args.work_dir, "model.pt"),
        map_location=device,
        weights_only=False,
    )

    vocab: Vocab = checkpoint["vocab"]
    lang2id: dict = checkpoint["lang2id"]
    n_langs: int = checkpoint["n_langs"]
    bigram_profiles: dict = checkpoint.get("__bigram_profiles__", {})
    model = CharTransformerLM(
        vocab_size=checkpoint["vocab_size"],
        n_langs=n_langs,
        d_model=checkpoint.get("d_model", D_MODEL),
        n_heads=checkpoint.get("n_heads", N_HEADS),
        n_layers=checkpoint.get("n_layers", N_LAYERS),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print(
        f"Loaded unified model  vocab={checkpoint['vocab_size']}  langs={sorted(lang2id)}"
    )
    print(f"Bigram LID profiles: {sorted(bigram_profiles)}")

    # Load test data
    if args.test_data.endswith(".csv"):
        df = pd.read_csv(args.test_data)
        contexts = df["context"].tolist()
        ids_col = df["id"].tolist()
        is_csv = True
    else:
        with open(args.test_data, "r", encoding="utf-8") as f:
            contexts = [line.rstrip("\r\n") for line in f if line.rstrip("\r\n")]
        ids_col = None
        is_csv = False

    latin_langs = [l for l in lang2id if l in bigram_profiles]
    fallback_lang = next(iter(lang2id))

    # Pre-built decode array: avoids a dict lookup per decoded token.
    decode_arr = [vocab.decode(i) for i in range(len(vocab))]

    # ── Normalise all contexts up front ──────────────────────────────
    clean = [unicodedata.normalize("NFC", str(c).strip('"')) for c in contexts]

    # ── Language detection — parallelised with threads ────────────────
    # detect_script calls unicodedata.name() (C extension, releases GIL) so
    # threads run truly in parallel; lid_latin bigram scoring is CPU-light.
    def _lid_one(ctx: str) -> int:
        detected = detect_script(ctx)
        if detected and detected in lang2id:
            return lang2id[detected]
        if latin_langs:
            return lang2id[lid_latin(bigram_profiles, latin_langs, ctx)]
        return lang2id[fallback_lang]

    n_workers = min(8, os.cpu_count() or 1)
    print(f"Running LID on {len(clean)} contexts ({n_workers} threads) …")
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        lang_id_per_ctx = list(ex.map(_lid_one, clean))

    # ── Sort by context length — reduces average padding per batch ────
    # Shorter contexts grouped together → tighter batches → less wasted compute.
    order = sorted(range(len(clean)), key=lambda i: len(clean[i]))

    # ── Single unified batched pass over ALL languages ────────────────
    # With a unified model, there is no reason to separate by language.
    # All examples are mixed into batches of INF_BATCH regardless of language.
    predictions = [""] * len(clean)
    inf_start = time.perf_counter()

    with torch.no_grad():
        for bi in range(0, len(order), INF_BATCH):
            batch_order = order[bi : bi + INF_BATCH]
            batch_texts = [clean[i] for i in batch_order]
            batch_lang_ids = [lang_id_per_ctx[i] for i in batch_order]
            preds = _batch_predict_top3(
                model, vocab, batch_lang_ids, batch_texts, device, decode_arr
            )
            for orig_i, pred_str in zip(batch_order, preds):
                predictions[orig_i] = pred_str

    inf_end = time.perf_counter()

    # ── Write output ──────────────────────────────────────────────────
    if is_csv:
        rows = [
            {"id": ids_col[i], "prediction": predictions[i]} for i in range(len(clean))
        ]
        pd.DataFrame(rows).to_csv(args.test_output, index=False)
    else:
        with open(args.test_output, "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(p + "\n")

    n = len(contexts)
    print(f"\n✓ Predictions written to {args.test_output}")
    print(
        f"  Inference : {inf_end - inf_start:.2f}s  "
        f"({(inf_end - inf_start) / n * 1000:.2f} ms/example)"
    )
    print(f"  Total     : {time.perf_counter() - t0:.2f}s")


# ══════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Unified multilingual char-level Transformer for next-char prediction."
    )
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--work_dir", default="../work")
    parser.add_argument("--train_dir", default="../data/train")
    parser.add_argument("--test_data", default="../kaggle-data/test.csv")
    parser.add_argument("--test_output", default="../submission.csv")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
