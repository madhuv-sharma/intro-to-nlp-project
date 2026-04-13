#!/usr/bin/env python
"""
Character-level GRU Language Model for the Interstellar Autocomplete Challenge.

Replaces the N-gram approach with a neural GRU model per language, as outlined
in the project proposal (Jurafsky & Martin §9; Vaswani et al. 2017; Kim et al. 2016).

Architecture
------------
- One CharGRULM per language (same per-language structure as the N-gram baseline)
- Embedding → 2-layer GRU → Linear → softmax
- Language detection: Unicode-script heuristic for non-Latin scripts;
  GRU log-likelihood scoring for Latin-script languages (en/fr/de/it)
- Inference: forward pass on the last SEQ_LEN characters, argmax top-3

Usage
-----
  python myprogram.py train --work_dir ../work --train_dir ../data/train
  python myprogram.py test  --work_dir ../work --test_data ../kaggle-data/test.csv \
                            --test_output ../submission.csv
"""

import argparse
import os
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

# ──────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────
SEQ_LEN = 128  # longer context helps; open-dev inputs are 100s of chars
EMBED_DIM = 128  # 64 was too small, especially for large-vocab CJK languages
HIDDEN_DIM = 512  # 256 left zh/ja/ko with loss ~3-3.6 after 8 epochs
NUM_LAYERS = 2
DROPOUT = 0.3  # slightly more regularisation at higher capacity
EPOCHS = 20  # all 10 langs still declining at ep8 in logs; train to convergence
BATCH_SIZE = 256
LR = 1e-3  # initial LR; CosineAnnealingLR decays this to 5% over EPOCHS
GRAD_CLIP = 1.0

# CJK scripts have 1000s of unique chars vs ~100 for Latin — give them more capacity
LANG_OVERRIDES = {
    "zh": {"embed_dim": 256, "hidden_dim": 768},
    "ja": {"embed_dim": 256, "hidden_dim": 768},
    "ko": {"embed_dim": 192, "hidden_dim": 640},
}

# ──────────────────────────────────────────────
# Special token IDs
# ──────────────────────────────────────────────
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_IDS = {PAD_ID, UNK_ID, BOS_ID, EOS_ID}


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
        for ch, cnt in counts.items():
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
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, h=None):
        emb = self.drop(self.embed(x))  # (B, T, E)
        out, h = self.gru(emb, h)  # (B, T, H)
        logits = self.fc(self.drop(out))  # (B, T, V)
        return logits, h


# ══════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════
def detect_script(text: str):
    """
    Return a language code when the text clearly uses a non-Latin script,
    otherwise return None (caller falls back to GRU scoring).
    """
    counts = {"ru": 0, "hi": 0, "ar": 0, "ko": 0, "ja": 0, "zh": 0}
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
            counts["ja"] += 1
        elif "CJK UNIFIED" in name:
            counts["zh"] += 1
    if max(counts.values()) == 0:
        return None
    return max(counts, key=counts.get)


def _encode_context(vocab: Vocab, text: str) -> list:
    """Prepend BOS, encode characters, and clip to SEQ_LEN.
    Always returns at least [BOS_ID] so RNN seq_len >= 1.
    """
    ids = [BOS_ID] + [vocab.encode(c) for c in text]
    ids = ids[-SEQ_LEN:]
    return ids if ids else [BOS_ID]  # guard: never empty


def score_context(
    model: CharGRULM,
    vocab: Vocab,
    text: str,
    device: torch.device,
) -> float:
    """
    Average per-character log-likelihood of the context under the model.
    Higher = more likely under this language model → used for LID.
    """
    ids = _encode_context(vocab, text)
    if len(ids) < 2:
        return 0.0
    model.eval()
    with torch.no_grad():
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        logits, _ = model(x)  # (1, T-1, V)
        log_p = torch.log_softmax(logits[0], dim=-1)  # (T-1, V)
        targets = torch.tensor(ids[1:], dtype=torch.long, device=device)
        score = log_p[range(len(targets)), targets].mean().item()
    return score


def predict_top3(
    model: CharGRULM,
    vocab: Vocab,
    text: str,
    device: torch.device,
) -> list:
    """Return the three most probable next characters as strings."""
    ids = _encode_context(vocab, text)
    model.eval()
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x)  # (1, T, V)
        last = logits[0, -1, :].clone()  # (V,)
        # Suppress special tokens so they are never predicted
        for sid in SPECIAL_IDS:
            if sid < last.shape[0]:
                last[sid] = float("-inf")
        top3_ids = torch.topk(last, 3).indices.tolist()
    return [vocab.decode(i) for i in top3_ids]


# ══════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════
def _make_sequences(token_ids: list, seq_len: int):
    """
    Yield (input, target) integer-list pairs of length seq_len using a
    stride of seq_len // 2 to increase effective training data volume.
    """
    stride = max(seq_len // 2, 1)
    for i in range(0, len(token_ids) - seq_len, stride):
        chunk = token_ids[i : i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        yield chunk[:-1], chunk[1:]


def _train_one_lang(lines: list, device: torch.device, lang: str = "") -> tuple:
    """Build vocab, encode corpus, train a CharGRULM, and return (model, vocab)."""
    vocab = Vocab().build(lines, min_count=1)

    # Flatten corpus into one long token stream
    all_ids = []
    for line in lines:
        all_ids.append(BOS_ID)
        all_ids.extend(vocab.encode(c) for c in line)
        all_ids.append(EOS_ID)

    seqs = list(_make_sequences(all_ids, SEQ_LEN))

    # Per-language capacity overrides for large-vocab CJK scripts
    overrides = LANG_OVERRIDES.get(lang, {})
    embed_dim = overrides.get("embed_dim", EMBED_DIM)
    hidden_dim = overrides.get("hidden_dim", HIDDEN_DIM)
    print(
        f"    vocab={len(vocab)}  sequences={len(seqs)}  "
        f"embed={embed_dim}  hidden={hidden_dim}"
    )

    model = CharGRULM(len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.05  # decays to 5% of initial LR
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(seqs)
        total_loss, n_batches = 0.0, 0

        for i in range(0, len(seqs), BATCH_SIZE):
            batch = seqs[i : i + BATCH_SIZE]
            inputs = torch.tensor(
                [s[0] for s in batch], dtype=torch.long, device=device
            )
            targets = torch.tensor(
                [s[1] for s in batch], dtype=torch.long, device=device
            )

            optimizer.zero_grad()
            logits, _ = model(inputs)  # (B, T, V)
            loss = criterion(
                logits.view(-1, len(vocab)),  # (B*T, V)
                targets.view(-1),  # (B*T,)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        scheduler.step()  # cosine step: called once per epoch, not per loss value
        print(
            f"    epoch {epoch}/{EPOCHS}  loss={avg:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    return model, vocab


# ══════════════════════════════════════════════
# Train mode
# ══════════════════════════════════════════════
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = Path(args.train_dir)
    total_start = time.perf_counter()

    print(f"Device : {device}")
    print(f"Epochs : {EPOCHS}  |  seq_len={SEQ_LEN}  |  hidden={HIDDEN_DIM}")

    checkpoint: dict = {}

    for file in sorted(train_dir.glob("*.txt")):
        lang = file.stem
        t0 = time.perf_counter()
        print(f"\n── {lang} ──────────────────────────────")

        lines = []
        with file.open("r", encoding="utf-8") as f:
            for raw in f:
                line = unicodedata.normalize("NFC", raw.rstrip("\r\n").strip('"'))
                if line:
                    lines.append(line)

        model, vocab = _train_one_lang(lines, device, lang=lang)
        model.eval()

        overrides = LANG_OVERRIDES.get(lang, {})
        checkpoint[lang] = {
            "vocab": vocab,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "vocab_size": len(vocab),
            "embed_dim": overrides.get("embed_dim", EMBED_DIM),
            "hidden_dim": overrides.get("hidden_dim", HIDDEN_DIM),
        }

        print(f"    {time.perf_counter() - t0:.1f}s")

    os.makedirs(args.work_dir, exist_ok=True)
    save_path = os.path.join(args.work_dir, "model.pt")
    torch.save(checkpoint, save_path)

    print(f"\nSaved checkpoint → {save_path}")
    print(f"Total training time: {time.perf_counter() - total_start:.1f}s")


# ══════════════════════════════════════════════
# Batched inference helpers
# ══════════════════════════════════════════════
INF_BATCH = 512  # examples per GPU batch during inference


def _batch_predict_top3(
    model: CharGRULM,
    vocab: Vocab,
    texts: list,
    device: torch.device,
) -> list:
    """
    Predict top-3 next characters for a list of texts in one padded forward
    pass, which is dramatically faster than looping one-by-one on GPU.

    Returns a list of 3-char strings aligned with `texts`.
    """
    # Encode all contexts; track their true lengths for last-position indexing
    encoded = [_encode_context(vocab, t) for t in texts]
    lengths = [len(ids) for ids in encoded]
    max_len = max(lengths)

    # Pad to max_len with PAD_ID
    padded = [ids + [PAD_ID] * (max_len - len(ids)) for ids in encoded]

    x = torch.tensor(padded, dtype=torch.long, device=device)  # (B, T)
    logits, _ = model(x)  # (B, T, V)

    # Gather the logit at the last real token for each example
    idx_tensor = torch.tensor(
        [l - 1 for l in lengths], dtype=torch.long, device=device
    )  # (B,)
    # last_logits[i] = logits[i, lengths[i]-1, :]
    last_logits = logits[
        torch.arange(len(texts), device=device), idx_tensor, :
    ]  # (B, V)

    # Suppress special tokens
    for sid in SPECIAL_IDS:
        if sid < last_logits.shape[1]:
            last_logits[:, sid] = float("-inf")

    top3_ids = torch.topk(last_logits, 3, dim=-1).indices  # (B, 3)

    return [
        "".join(vocab.decode(top3_ids[i, j].item()) for j in range(3))
        for i in range(len(texts))
    ]


def _score_contexts_batch(
    model: CharGRULM,
    vocab: Vocab,
    texts: list,
    device: torch.device,
) -> list:
    """
    Compute avg per-character log-likelihood for a batch of texts at once.
    Returns a list of float scores aligned with `texts`.
    """
    encoded = [_encode_context(vocab, t) for t in texts]
    # Need at least 2 tokens (BOS + one char) to score; fall back to 0.
    scores = []
    valid_idx, valid_enc = [], []
    for i, ids in enumerate(encoded):
        if len(ids) >= 2:
            valid_idx.append(i)
            valid_enc.append(ids)
        else:
            scores.append((i, 0.0))

    if valid_enc:
        lengths = [len(ids) for ids in valid_enc]
        max_len = max(lengths)
        padded = [ids + [PAD_ID] * (max_len - len(ids)) for ids in valid_enc]

        x = torch.tensor([ids[:-1] for ids in padded], dtype=torch.long, device=device)
        logits, _ = model(x)  # (B, T-1, V)
        log_p = torch.log_softmax(logits, dim=-1)  # (B, T-1, V)

        for k, (i, ids) in enumerate(zip(valid_idx, valid_enc)):
            tgt = torch.tensor(ids[1:], dtype=torch.long, device=device)  # (T-1,)
            t = len(tgt)
            sc = log_p[k, :t, :][torch.arange(t), tgt].mean().item()
            scores.append((i, sc))

    scores.sort()
    return [s for _, s in scores]


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

    # Reconstruct models
    models: dict = {}
    vocabs: dict = {}

    for lang, state in checkpoint.items():
        vocab = state["vocab"]
        vocabs[lang] = vocab
        m = CharGRULM(
            state["vocab_size"],
            embed_dim=state.get("embed_dim", EMBED_DIM),
            hidden_dim=state.get("hidden_dim", HIDDEN_DIM),
        ).to(device)
        m.load_state_dict(state["state_dict"])
        m.eval()
        models[lang] = m

    print(f"Loaded {len(models)} language models: {sorted(models)}")

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

    # Latin-script languages need GRU scoring to distinguish
    latin_langs = [l for l in models if l in {"en", "fr", "de", "it"}]

    # ── Step 1: language detection (CPU-side, fast) ──────────────────
    # Normalise all contexts up front
    clean = [unicodedata.normalize("NFC", str(c).strip('"')) for c in contexts]

    # Separate examples by detected language; latin-script ones go into a
    # holding list for batch LID scoring.
    lang_buckets: dict = {l: [] for l in models}  # lang → list of (orig_idx, ctx)
    latin_queue: list = []  # (orig_idx, ctx) needing LID

    for i, ctx in enumerate(clean):
        detected = detect_script(ctx)
        if detected and detected in models:
            lang_buckets[detected].append((i, ctx))
        else:
            latin_queue.append((i, ctx))

    # ── Step 2: batch-score latin examples across all latin LMs ──────
    if latin_queue and latin_langs:
        latin_texts = [ctx for _, ctx in latin_queue]
        # Score all latin langs at once; shape: (n_langs, n_examples)
        all_scores = {}
        for lang in latin_langs:
            with torch.no_grad():
                # Process in sub-batches to avoid OOM on very large sets
                s_list = []
                for bi in range(0, len(latin_texts), INF_BATCH):
                    s_list.extend(
                        _score_contexts_batch(
                            models[lang],
                            vocabs[lang],
                            latin_texts[bi : bi + INF_BATCH],
                            device,
                        )
                    )
            all_scores[lang] = s_list

        for k, (orig_i, ctx) in enumerate(latin_queue):
            best_lang = max(latin_langs, key=lambda l: all_scores[l][k])
            lang_buckets[best_lang].append((orig_i, ctx))

    # ── Step 3: batched GRU prediction per language ──────────────────
    predictions = [""] * len(clean)  # pre-allocate result array

    inf_start = time.perf_counter()

    for lang, bucket in lang_buckets.items():
        if not bucket:
            continue
        model = models[lang]
        vocab = vocabs[lang]
        idxs = [i for i, _ in bucket]
        texts = [t for _, t in bucket]

        with torch.no_grad():
            for bi in range(0, len(texts), INF_BATCH):
                batch_idxs = idxs[bi : bi + INF_BATCH]
                batch_texts = texts[bi : bi + INF_BATCH]
                preds = _batch_predict_top3(model, vocab, batch_texts, device)
                for orig_i, pred_str in zip(batch_idxs, preds):
                    predictions[orig_i] = pred_str

        print(f"  [{lang}] {len(bucket)} examples done")

    inf_end = time.perf_counter()

    # ── Write output ─────────────────────────────────────────────────
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
    print(f"\nPredictions written → {args.test_output}")
    print(f"Inference time : {inf_end - inf_start:.2f}s")
    print(f"Avg per example: {(inf_end - inf_start) / n:.5f}s")
    print(f"Total test time: {time.perf_counter() - t0:.2f}s")


# ══════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Character-level GRU LM for next-character prediction."
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
