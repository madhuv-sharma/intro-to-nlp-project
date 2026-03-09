#!/usr/bin/env python
"""
Character-level GRU Language Model for the Interstellar Autocomplete Challenge.

Architecture
------------
- One CharGRULM per language
- Embedding → 2-layer GRU → Linear → softmax
- Language detection: Unicode-script heuristic for non-Latin scripts;
  bigram-profile LID for Latin-script languages (en/fr/de/it) — fast & accurate
- Mixed-precision (fp16) training and inference
- LR warmup (1 epoch linear) → CosineAnnealingLR
- Character dropout for robustness
- Per-language epoch + capacity overrides (CJK needs more)
- Batched inference (512 examples per GPU call)

Usage
-----
  python myprogram.py train --work_dir ../work --train_dir ../data/train
  python myprogram.py test  --work_dir ../work --test_data ../kaggle-data/test.csv \
                            --test_output ../submission.csv
"""

import argparse
import math
import os
import random
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

# ──────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────
SEQ_LEN      = 128    # context window
EMBED_DIM    = 128
HIDDEN_DIM   = 512
NUM_LAYERS   = 2
DROPOUT      = 0.3
EPOCHS       = 20     # default; CJK langs get more via LANG_OVERRIDES
BATCH_SIZE   = 512    # larger = better GPU utilisation with fp16
LR           = 1e-3
WEIGHT_DECAY = 1e-4   # AdamW regularisation; helps prevent overfitting
GRAD_CLIP    = 1.0
CHAR_DROPOUT = 0.03   # probability of replacing an input char with UNK during train

# Per-language overrides: capacity + extra epochs for underfitting CJK langs
LANG_OVERRIDES = {
    "zh": {"embed_dim": 256, "hidden_dim": 768, "epochs": 30},
    "ja": {"embed_dim": 256, "hidden_dim": 768, "epochs": 30},
    "ko": {"embed_dim": 192, "hidden_dim": 640, "epochs": 25},
}

# Stride for sliding-window sequence extraction.
# CJK gets a smaller stride (more sequences from same data) since they have
# fewer unique lines and their vocabs are huge — they need more exposure.
LATIN_STRIDE_DIV = 2   # stride = seq_len // 2
CJK_STRIDE_DIV   = 4   # stride = seq_len // 4  → ~2× more sequences
CJK_LANGS        = {"zh", "ja", "ko"}

# ──────────────────────────────────────────────
# Special token IDs
# ──────────────────────────────────────────────
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_IDS = {PAD_ID, UNK_ID, BOS_ID, EOS_ID}

# Inference batch size (examples per GPU forward pass)
INF_BATCH = 512


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
        for ch, cnt in sorted(counts.items()):   # sorted → deterministic vocab
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
class CharGRULM(nn.Module):
    """
    Character-level autoregressive GRU language model.

    Input  : integer token sequence  (batch, seq_len)
    Output : per-position logits      (batch, seq_len, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int   = EMBED_DIM,
        hidden_dim: int   = HIDDEN_DIM,
        num_layers: int   = NUM_LAYERS,
        dropout:    float = DROPOUT,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.gru   = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, h=None):
        emb    = self.drop(self.embed(x))   # (B, T, E)
        out, h = self.gru(emb, h)           # (B, T, H)
        logits = self.fc(self.drop(out))    # (B, T, V)
        return logits, h


# ══════════════════════════════════════════════
# Language identification helpers
# ══════════════════════════════════════════════
def detect_script(text: str):
    """
    Return a language code when text clearly uses a non-Latin script,
    otherwise return None (caller does bigram-profile LID for Latin langs).
    """
    counts = {"ru": 0, "hi": 0, "ar": 0, "ko": 0, "ja": 0, "zh": 0}
    for ch in text:
        name = unicodedata.name(ch, "")
        if   "CYRILLIC"    in name:                        counts["ru"] += 1
        elif "DEVANAGARI"  in name:                        counts["hi"] += 1
        elif "ARABIC"      in name:                        counts["ar"] += 1
        elif "HANGUL"      in name:                        counts["ko"] += 1
        elif "HIRAGANA" in name or "KATAKANA" in name:    counts["ja"] += 1
        elif "CJK UNIFIED" in name:                        counts["zh"] += 1
    if max(counts.values()) == 0:
        return None
    return max(counts, key=counts.get)


def build_bigram_profiles(lang_lines: dict) -> dict:
    """
    For each Latin-script language, compute a normalised log-probability
    distribution over character bigrams from the training lines.
    Returns: {lang: {bigram: log_prob}}
    Used at test time for fast, accurate Latin LID without a GPU forward pass.
    """
    profiles = {}
    for lang, lines in lang_lines.items():
        counts = Counter()
        for line in lines:
            for a, b in zip(line, line[1:]):
                counts[(a, b)] += 1
        total = sum(counts.values()) + len(counts)   # +1 smoothing
        profiles[lang] = {
            bg: math.log((cnt + 1) / total)
            for bg, cnt in counts.items()
        }
    return profiles


def score_bigram(profile: dict, text: str) -> float:
    """Average bigram log-prob of text under a language profile."""
    if len(text) < 2:
        return 0.0
    unk_lp = math.log(1e-6)
    scores = [profile.get((a, b), unk_lp) for a, b in zip(text, text[1:])]
    return sum(scores) / len(scores)


# ══════════════════════════════════════════════
# Encoding utilities
# ══════════════════════════════════════════════
def _encode_context(vocab: Vocab, text: str) -> list:
    """Prepend BOS, encode characters, clip to SEQ_LEN.
    Always returns at least [BOS_ID] so RNN seq_len >= 1.
    """
    ids = [BOS_ID] + [vocab.encode(c) for c in text]
    ids = ids[-SEQ_LEN:]
    return ids if ids else [BOS_ID]


# ══════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════
def _make_sequences(token_ids: list, seq_len: int, stride_div: int):
    """
    Yield (input, target) integer-list pairs of length seq_len.
    stride = seq_len // stride_div; smaller stride → more sequences.
    """
    stride = max(seq_len // stride_div, 1)
    for i in range(0, len(token_ids) - seq_len, stride):
        chunk = token_ids[i : i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        yield chunk[:-1], chunk[1:]


def _train_one_lang(lines: list, device: torch.device, lang: str = "") -> tuple:
    """Build vocab, encode corpus, train a CharGRULM, return (model, vocab)."""
    vocab = Vocab().build(lines, min_count=1)

    # Per-language capacity + epoch overrides
    overrides  = LANG_OVERRIDES.get(lang, {})
    embed_dim  = overrides.get("embed_dim",  EMBED_DIM)
    hidden_dim = overrides.get("hidden_dim", HIDDEN_DIM)
    epochs     = overrides.get("epochs",     EPOCHS)
    stride_div = CJK_STRIDE_DIV if lang in CJK_LANGS else LATIN_STRIDE_DIV

    # Flatten corpus into one long token stream
    all_ids: list = []
    for line in lines:
        all_ids.append(BOS_ID)
        all_ids.extend(vocab.encode(c) for c in line)
        all_ids.append(EOS_ID)

    seqs = list(_make_sequences(all_ids, SEQ_LEN, stride_div))
    print(f"    vocab={len(vocab)}  sequences={len(seqs)}  "
          f"embed={embed_dim}  hidden={hidden_dim}  epochs={epochs}")

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    model     = CharGRULM(len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    # AdamW: Adam + decoupled weight decay → better generalisation than plain Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # 1-epoch linear warmup, then cosine decay to 5% of LR
    warmup    = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=1
    )
    cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - 1, 1), eta_min=LR * 0.05
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[1]
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    vocab_size = len(vocab)

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(seqs)
        total_loss, n_batches = 0.0, 0

        for i in range(0, len(seqs), BATCH_SIZE):
            batch   = seqs[i : i + BATCH_SIZE]
            inputs  = torch.tensor([s[0] for s in batch], dtype=torch.long, device=device)
            targets = torch.tensor([s[1] for s in batch], dtype=torch.long, device=device)

            # Character dropout: randomly mask 3% of input tokens → UNK
            if CHAR_DROPOUT > 0:
                mask   = torch.rand_like(inputs, dtype=torch.float) < CHAR_DROPOUT
                # Don't drop PAD or BOS
                mask  &= (inputs != PAD_ID) & (inputs != BOS_ID)
                inputs = inputs.masked_fill(mask, UNK_ID)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _ = model(inputs)                       # (B, T, V)
                loss = criterion(
                    logits.view(-1, vocab_size),                # (B*T, V)
                    targets.view(-1),                           # (B*T,)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches  += 1

        avg = total_loss / max(n_batches, 1)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"    epoch {epoch:2d}/{epochs}  loss={avg:.4f}  lr={cur_lr:.2e}")

    return model, vocab


# ══════════════════════════════════════════════
# Train mode
# ══════════════════════════════════════════════
def train(args):
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir   = Path(args.train_dir)
    total_start = time.perf_counter()

    print(f"Device : {device}")
    print(f"Default epochs={EPOCHS}  seq_len={SEQ_LEN}  hidden={HIDDEN_DIM}  "
          f"batch={BATCH_SIZE}  fp16={device.type == 'cuda'}")

    checkpoint: dict  = {}
    latin_train_lines: dict = {}   # saved for bigram profile building

    for file in sorted(train_dir.glob("*.txt")):
        lang = file.stem
        t0   = time.perf_counter()
        print(f"\n── Training [{lang}] ─────────────────────────────────────────")

        lines = []
        with file.open("r", encoding="utf-8") as f:
            for raw in f:
                line = unicodedata.normalize("NFC", raw.rstrip("\r\n").strip('"'))
                if line:
                    lines.append(line)

        print(f"  Lines: {len(lines):,}  |  device: {device}")

        if lang not in CJK_LANGS and lang not in {"ru", "hi", "ar"}:
            latin_train_lines[lang] = lines

        model, vocab = _train_one_lang(lines, device, lang=lang)
        model.eval()

        overrides = LANG_OVERRIDES.get(lang, {})
        checkpoint[lang] = {
            "vocab":      vocab,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "vocab_size": len(vocab),
            "embed_dim":  overrides.get("embed_dim",  EMBED_DIM),
            "hidden_dim": overrides.get("hidden_dim", HIDDEN_DIM),
        }

        print(f"    {time.perf_counter() - t0:.1f}s")

    # Build and save bigram profiles for Latin LID
    print("\nBuilding Latin bigram LID profiles …")
    bigram_profiles = build_bigram_profiles(latin_train_lines)
    checkpoint["__bigram_profiles__"] = bigram_profiles
    print(f"  Profiles built for: {sorted(bigram_profiles)}")

    os.makedirs(args.work_dir, exist_ok=True)
    save_path = os.path.join(args.work_dir, "model.pt")
    torch.save(checkpoint, save_path)

    print(f"\n✓ Training complete — {time.perf_counter() - total_start:.1f}s  |  saved to {save_path}")


# ══════════════════════════════════════════════
# Batched inference helpers
# ══════════════════════════════════════════════
def _batch_predict_top3(
    model:   CharGRULM,
    vocab:   Vocab,
    texts:   list,
    device:  torch.device,
) -> list:
    """
    Predict top-3 next characters for a list of texts in one padded forward pass.
    Returns a list of 3-char strings aligned with `texts`.
    """
    encoded  = [_encode_context(vocab, t) for t in texts]
    lengths  = [len(ids) for ids in encoded]
    max_len  = max(lengths)

    padded = [ids + [PAD_ID] * (max_len - len(ids)) for ids in encoded]

    use_amp = device.type == "cuda"
    x       = torch.tensor(padded, dtype=torch.long, device=device)   # (B, T)

    with torch.cuda.amp.autocast(enabled=use_amp):
        logits, _ = model(x)                                           # (B, T, V)

    idx_tensor  = torch.tensor([l - 1 for l in lengths], dtype=torch.long, device=device)
    last_logits = logits[torch.arange(len(texts), device=device), idx_tensor, :]  # (B, V)

    for sid in SPECIAL_IDS:
        if sid < last_logits.shape[1]:
            last_logits[:, sid] = float("-inf")

    top3_ids = torch.topk(last_logits, 3, dim=-1).indices   # (B, 3)

    return [
        "".join(vocab.decode(top3_ids[i, j].item()) for j in range(3))
        for i in range(len(texts))
    ]


# ══════════════════════════════════════════════
# Test mode
# ══════════════════════════════════════════════
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0     = time.perf_counter()

    print(f"Device: {device}")
    print("Loading checkpoint …")

    checkpoint = torch.load(
        os.path.join(args.work_dir, "model.pt"),
        map_location=device,
        weights_only=False,
    )

    # Extract bigram profiles (saved alongside model weights)
    bigram_profiles: dict = checkpoint.pop("__bigram_profiles__", {})

    # Reconstruct per-language GRU models
    models: dict = {}
    vocabs: dict = {}

    for lang, state in checkpoint.items():
        vocab = state["vocab"]
        vocabs[lang] = vocab
        m = CharGRULM(
            state["vocab_size"],
            embed_dim=state.get("embed_dim",  EMBED_DIM),
            hidden_dim=state.get("hidden_dim", HIDDEN_DIM),
        ).to(device)
        m.load_state_dict(state["state_dict"])
        m.eval()
        models[lang] = m

    print(f"Loaded {len(models)} language models: {sorted(models)}")
    print(f"Bigram LID profiles: {sorted(bigram_profiles)}")

    # Load test data
    if args.test_data.endswith(".csv"):
        df       = pd.read_csv(args.test_data)
        contexts = df["context"].tolist()
        ids_col  = df["id"].tolist()
        is_csv   = True
    else:
        with open(args.test_data, "r", encoding="utf-8") as f:
            contexts = [line.rstrip("\r\n") for line in f if line.rstrip("\r\n")]
        ids_col = None
        is_csv  = False

    latin_langs = [l for l in models if l in bigram_profiles]

    # ── Normalise all contexts up front ──────────────────────────────
    clean = [unicodedata.normalize("NFC", str(c).strip('"')) for c in contexts]

    # ── Step 1: Language detection ────────────────────────────────────
    # Unicode script → non-Latin langs (instant, no GPU)
    # Bigram profile scoring → Latin langs (CPU, ~100× faster than GRU scoring)
    lang_buckets: dict = {l: [] for l in models}

    for i, ctx in enumerate(clean):
        detected = detect_script(ctx)
        if detected and detected in models:
            lang_buckets[detected].append((i, ctx))
        else:
            # Bigram LID for Latin langs
            if latin_langs:
                best_lang = max(latin_langs, key=lambda l: score_bigram(bigram_profiles[l], ctx))
            else:
                best_lang = next(iter(models))
            lang_buckets[best_lang].append((i, ctx))

    # ── Step 2: Batched GRU prediction per language ───────────────────
    predictions = [""] * len(clean)
    inf_start   = time.perf_counter()

    for lang, bucket in lang_buckets.items():
        if not bucket:
            continue
        model = models[lang]
        vocab = vocabs[lang]
        idxs  = [i for i, _ in bucket]
        texts = [t for _, t in bucket]

        with torch.no_grad():
            for bi in range(0, len(texts), INF_BATCH):
                batch_idxs  = idxs[bi : bi + INF_BATCH]
                batch_texts = texts[bi : bi + INF_BATCH]
                preds = _batch_predict_top3(model, vocab, batch_texts, device)
                for orig_i, pred_str in zip(batch_idxs, preds):
                    predictions[orig_i] = pred_str

        print(f"  [{lang}] {len(bucket)} examples done")

    inf_end = time.perf_counter()

    # ── Write output ─────────────────────────────────────────────────
    if is_csv:
        rows = [{"id": ids_col[i], "prediction": predictions[i]} for i in range(len(clean))]
        pd.DataFrame(rows).to_csv(args.test_output, index=False)
    else:
        with open(args.test_output, "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(p + "\n")

    n = len(contexts)
    print(f"\n✓ Predictions written to {args.test_output}")
    print(f"  Inference : {inf_end - inf_start:.2f}s  "
          f"({(inf_end - inf_start) / n * 1000:.2f} ms/example)")
    print(f"  Total     : {time.perf_counter() - t0:.2f}s")


# ══════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Character-level GRU LM for next-character prediction."
    )
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--work_dir",    default="../work")
    parser.add_argument("--train_dir",   default="../data/train")
    parser.add_argument("--test_data",   default="../kaggle-data/test.csv")
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
