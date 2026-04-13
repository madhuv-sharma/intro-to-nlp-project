#!/usr/bin/env python
"""
Unified multilingual character-level Transformer LM
for the Interstellar Autocomplete Challenge.

Architecture
------------
- Single CharTransformerLM for ALL languages (language conditioning via embedding)
- Token embedding + learned positional embedding + language embedding → summed
- Deeper causal (decoder-only) Transformer, pre-norm style (GPT-2 variant)
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
  python myprogram.py train --work_dir ../work --train_dir "../data/train v2 (with test)"
  python myprogram.py train --work_dir ../work --resume_model ../work/model.pt --extra_epochs 8
  python myprogram.py test  --work_dir ../work --test_data ../kaggle-data/test.csv \\
                            --test_output ../submission.csv
"""

import argparse
import os
import random
import time
import unicodedata
import warnings
from collections import Counter
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
D_MODEL = 576
N_HEADS = 9  # head_dim = 576 / 9 = 64
N_LAYERS = 5
DROPOUT = 0.10
EPOCHS = 12
BATCH_SIZE = 256
LR = 3e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
CHAR_DROPOUT = 0.02

CJK_LANGS = {"zh", "ja", "ko"}
LATIN_STRIDE_DIV = 2  # stride = seq_len // 2
CJK_STRIDE_DIV = 8  # stride = seq_len // 8 → moderate extra CJK sampling


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

    Input : token ids (B, T), optional pad mask (B, T)
    Output: per-position logits (B, T, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
        max_seq: int = SEQ_LEN,
    ):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(max_seq + 2, d_model)
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
        pad_mask: Optional[torch.Tensor] = None,  # (B, T) True = padding
    ) -> torch.Tensor:
        T = x.shape[1]
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        h = (
            self.tok_embed(x)  # (B, T, D)
            + self.pos_embed(pos)  # (1, T, D)
        )
        out = self.transformer(
            h,
            mask=self._causal_buf[:T, :T],  # slice cached mask — no allocation
            src_key_padding_mask=pad_mask,
        )
        return self.fc(out)  # (B, T, V)



# ══════════════════════════════════════════════
# Encoding utilities
# ══════════════════════════════════════════════
def _encode_context(vocab: Vocab, text: str, seq_len: int = SEQ_LEN) -> list:
    """Prepend BOS, encode characters, clip to seq_len."""
    ids = [BOS_ID] + [vocab.encode(c) for c in text]
    ids = ids[-seq_len:]
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
    Yield fixed-length (input, target) pairs from one line only.

    This function never crosses sample boundaries. For short lines, it emits
    one padded sequence so each dialogue line still contributes supervision.
    """
    n = len(token_ids)
    if n < 2:
        return

    # Short line: one padded sequence (targets on PAD positions are ignored).
    if n <= seq_len:
        inp = token_ids[:-1]
        tgt = token_ids[1:]
        pad = seq_len - len(inp)
        if pad > 0:
            inp = inp + [PAD_ID] * pad
            tgt = tgt + [PAD_ID] * pad
        yield inp, tgt
        return

    stride = max(seq_len // stride_div, 1)
    last_start = n - (seq_len + 1)
    i = 0
    while i <= last_start:
        chunk = token_ids[i : i + seq_len + 1]
        yield chunk[:-1], chunk[1:]
        i += stride

    # Ensure we cover the line tail as well.
    if (i - stride) != last_start:
        chunk = token_ids[last_start : last_start + seq_len + 1]
        yield chunk[:-1], chunk[1:]


# ══════════════════════════════════════════════
# Unified training
# ══════════════════════════════════════════════
def _train_unified(
    inputs_arr: "np.ndarray",  # (N, SEQ_LEN) int16
    targets_arr: "np.ndarray",  # (N, SEQ_LEN) int16
    vocab: Vocab,
    device: torch.device,
    max_seq: int = SEQ_LEN,
    d_model: int = D_MODEL,
    n_heads: int = N_HEADS,
    n_layers: int = N_LAYERS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    start_epoch: int = 1,
    target_epochs: int = EPOCHS,
    resume_model_state: Optional[dict] = None,
    resume_optimizer_state: Optional[dict] = None,
    resume_scheduler_state: Optional[dict] = None,
    resume_scaler_state: Optional[dict] = None,
) -> tuple:
    """Train a single CharTransformerLM on mixed multilingual sequences."""
    vocab_size = len(vocab)
    n_total = len(inputs_arr)
    model = CharTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq=max_seq,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,}  vocab: {vocab_size}")

    # Pre-load all data to the target device once.
    # On GPU: eliminates per-batch CPU→GPU transfer and numpy fancy-indexing overhead.
    # Stored as int32 (half the footprint of int64); cast to long() per batch is cheap.
    # GPU VRAM cost: ~2 × N × SEQ_LEN × 4 B ≈ 670 MB for 655 K seqs — fits T4/A100.
    print("  Pre-loading data to device …")
    inputs_dev = torch.from_numpy(inputs_arr.astype(np.int32)).to(device)
    targets_dev = torch.from_numpy(targets_arr.astype(np.int32)).to(device)

    # torch.compile fuses kernels and eliminates Python dispatch overhead.
    # Falls back silently if the environment doesn't support it.
    if device.type == "cuda":
        try:
            model = torch.compile(model)
            print("  torch.compile: enabled")
        except Exception:
            print("  torch.compile: unavailable, skipping")

    if resume_model_state is not None:
        getattr(model, "_orig_mod", model).load_state_dict(resume_model_state)
        print("  Loaded model weights for resume")

    # fused=True runs AdamW as a single CUDA kernel (vs. one kernel per parameter).
    fused_ok = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, fused=fused_ok
    )
    # 2-epoch linear warmup then cosine decay — transformers benefit from
    # a gradual ramp-up before the full learning rate is applied.
    _warmup_epochs = 2
    _warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=_warmup_epochs
    )
    _cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(target_epochs - _warmup_epochs, 1), eta_min=lr * 0.05
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[_warmup, _cosine], milestones=[_warmup_epochs]
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.05)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        print("  Loaded optimizer state for resume")
    if resume_scheduler_state is not None:
        scheduler.load_state_dict(resume_scheduler_state)
        print("  Loaded scheduler state for resume")
    if resume_scaler_state is not None and use_amp:
        scaler.load_state_dict(resume_scaler_state)
        print("  Loaded AMP scaler state for resume")

    # If no scheduler state exists (or horizon changed), approximate by stepping.
    if resume_scheduler_state is None and start_epoch > 1:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, module=r"torch\.optim\.lr_scheduler"
            )
            for _ in range(1, start_epoch):
                scheduler.step()
        print(
            f"  Scheduler advanced to epoch {start_epoch - 1} "
            f"(target horizon: {target_epochs})"
        )

    # Accumulate loss as a GPU tensor — one CPU/GPU sync per epoch instead of per batch.
    running_loss = torch.zeros(1, device=device)

    if start_epoch > target_epochs:
        print(
            f"  start_epoch={start_epoch} is greater than target_epochs={target_epochs}; "
            "skipping training."
        )
        return model, optimizer, scheduler, scaler, start_epoch - 1

    for epoch in range(start_epoch, target_epochs + 1):
        model.train()
        perm = torch.randperm(n_total, device=device)
        running_loss.zero_()
        n_batches = 0

        for i in range(0, n_total, batch_size):
            idx = perm[i : i + batch_size]
            inputs = inputs_dev[idx].long()
            targets = targets_dev[idx].long()
            pad_mask = inputs == PAD_ID

            # Character dropout: randomly mask 3% of input tokens to UNK
            if CHAR_DROPOUT > 0:
                mask = torch.rand_like(inputs, dtype=torch.float) < CHAR_DROPOUT
                mask &= (inputs != PAD_ID) & (inputs != BOS_ID)
                inputs = inputs.masked_fill(mask, UNK_ID)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inputs, pad_mask)  # (B, T, V)
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
        print(f"  epoch {epoch:2d}/{target_epochs}  loss={avg:.4f}  lr={cur_lr:.2e}")

    return model, optimizer, scheduler, scaler, target_epochs


# ══════════════════════════════════════════════
# Train mode
# ══════════════════════════════════════════════
def _default_train_state_path(model_checkpoint_path: Path) -> Path:
    return model_checkpoint_path.with_name(f"{model_checkpoint_path.stem}.train_state.pt")


def _checkpoint_seq_len(checkpoint: dict, fallback: int = SEQ_LEN) -> int:
    if "seq_len" in checkpoint:
        return int(checkpoint["seq_len"])
    state_dict = checkpoint.get("state_dict", {})
    pos_w = state_dict.get("pos_embed.weight")
    if torch.is_tensor(pos_w) and pos_w.ndim == 2 and pos_w.shape[0] >= 3:
        return int(pos_w.shape[0] - 2)
    return fallback


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = Path(args.train_dir)
    total_start = time.perf_counter()

    resume_model_ckpt = None
    resume_train_state = None
    resume_optimizer_state = None
    resume_scheduler_state = None
    resume_scheduler_state_raw = None
    resume_scheduler_target_epochs = None
    resume_scaler_state = None
    resume_epoch = 0

    if args.resume_model:
        resume_model_path = Path(args.resume_model)
        if not resume_model_path.exists():
            raise FileNotFoundError(f"resume model not found: {resume_model_path}")
        print(f"Resume model checkpoint: {resume_model_path}")
        resume_model_ckpt = torch.load(
            str(resume_model_path), map_location="cpu", weights_only=False
        )
        for k in ("vocab", "vocab_size", "state_dict"):
            if k not in resume_model_ckpt:
                raise ValueError(f"resume model checkpoint missing key: {k}")
        resume_epoch = int(resume_model_ckpt.get("epoch", 0))

        if args.resume_state:
            resume_state_path = Path(args.resume_state)
        else:
            resume_state_path = _default_train_state_path(resume_model_path)
        if resume_state_path.exists():
            print(f"Resume train-state checkpoint: {resume_state_path}")
            resume_train_state = torch.load(
                str(resume_state_path), map_location="cpu", weights_only=False
            )
            state_epoch = int(resume_train_state.get("epoch", -1))
            if state_epoch == resume_epoch:
                resume_optimizer_state = resume_train_state.get("optimizer_state_dict")
                resume_scheduler_state_raw = resume_train_state.get("scheduler_state_dict")
                _sched_target = resume_train_state.get("target_epochs")
                if _sched_target is not None:
                    resume_scheduler_target_epochs = int(_sched_target)
                resume_scaler_state = resume_train_state.get("scaler_state_dict")
            else:
                print(
                    f"  Warning: model epoch ({resume_epoch}) and train-state epoch "
                    f"({state_epoch}) differ; optimizer/scheduler/scaler state will be ignored."
                )
        else:
            print(
                "  No train-state checkpoint found; resuming from model weights only "
                "(optimizer/scheduler will restart)."
            )

    configured_epochs = int(args.epochs)
    if configured_epochs < 1:
        raise ValueError("--epochs must be >= 1")

    batch_size = int(args.batch_size) if args.batch_size is not None else BATCH_SIZE
    if batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    lr = float(args.lr) if args.lr is not None else LR
    if lr <= 0:
        raise ValueError("--lr must be > 0")

    if resume_model_ckpt:
        ckpt_d = int(resume_model_ckpt.get("d_model", D_MODEL))
        ckpt_h = int(resume_model_ckpt.get("n_heads", N_HEADS))
        ckpt_l = int(resume_model_ckpt.get("n_layers", N_LAYERS))
        ckpt_seq_len = _checkpoint_seq_len(resume_model_ckpt, SEQ_LEN)

        # Guardrails: resumed checkpoints must keep architecture identical.
        if args.d_model is not None and int(args.d_model) != ckpt_d:
            raise ValueError(
                f"--d_model={args.d_model} does not match checkpoint d_model={ckpt_d}. "
                "Start a fresh run (no --resume_model) to change model size."
            )
        if args.n_heads is not None and int(args.n_heads) != ckpt_h:
            raise ValueError(
                f"--n_heads={args.n_heads} does not match checkpoint n_heads={ckpt_h}. "
                "Start a fresh run (no --resume_model) to change model size."
            )
        if args.n_layers is not None and int(args.n_layers) != ckpt_l:
            raise ValueError(
                f"--n_layers={args.n_layers} does not match checkpoint n_layers={ckpt_l}. "
                "Start a fresh run (no --resume_model) to change model size."
            )
        if args.seq_len is not None and int(args.seq_len) != ckpt_seq_len:
            raise ValueError(
                f"--seq_len={args.seq_len} does not match checkpoint seq_len={ckpt_seq_len}. "
                "Start a fresh run (no --resume_model) to change context length."
            )

        model_d = ckpt_d
        model_h = ckpt_h
        model_l = ckpt_l
        model_seq_len = ckpt_seq_len
    else:
        model_d = int(args.d_model) if args.d_model is not None else D_MODEL
        model_h = int(args.n_heads) if args.n_heads is not None else N_HEADS
        model_l = int(args.n_layers) if args.n_layers is not None else N_LAYERS
        model_seq_len = int(args.seq_len) if args.seq_len is not None else SEQ_LEN

    if model_d % model_h != 0:
        raise ValueError(
            f"d_model ({model_d}) must be divisible by n_heads ({model_h})."
        )
    if model_l < 1:
        raise ValueError("--n_layers must be >= 1")
    if model_seq_len < 8:
        raise ValueError("--seq_len should be >= 8")

    start_epoch = resume_epoch + 1
    if resume_model_ckpt:
        extra_epochs = args.extra_epochs if args.extra_epochs > 0 else configured_epochs
        target_epochs = resume_epoch + extra_epochs
    else:
        target_epochs = configured_epochs

    if resume_scheduler_state_raw is not None:
        if resume_scheduler_target_epochs == target_epochs:
            resume_scheduler_state = resume_scheduler_state_raw
        else:
            print(
                "  Scheduler horizon changed; rebuilding scheduler for the new target and "
                "advancing from epoch instead of loading old scheduler state."
            )

    print(f"Device : {device}")
    print(
        f"Transformer  d_model={model_d}  n_heads={model_h}  n_layers={model_l}  "
        f"seq_len={model_seq_len}  target_epochs={target_epochs}  "
        f"batch={batch_size}  lr={lr:.2e}  fp16={device.type == 'cuda'}"
    )
    if resume_model_ckpt:
        print(
            f"  Resume window: start_epoch={start_epoch}  (+{target_epochs - resume_epoch} epochs)"
        )
    elif args.extra_epochs > 0:
        print("  Note: --extra_epochs has no effect without --resume_model")

    # Phase 1: Read & clean all language files
    lang_lines: dict = {}

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
        print(f"  [{lang}] {len(lines):,} lines after cleaning")

    # Phase 2: Build/load vocabulary
    print("\nBuilding unified vocabulary ...")
    if resume_model_ckpt is not None:
        vocab: Vocab = resume_model_ckpt["vocab"]
        print(f"  Reusing checkpoint vocabulary  size={len(vocab)}")
    else:
        all_lines = [line for lines in lang_lines.values() for line in lines]
        # min_count=1: keep rare characters to reduce UNK on CJK tails.
        vocab = Vocab().build(all_lines, min_count=1)
        print(f"  Unified vocab size: {len(vocab)}")

    print(f"  Languages: {sorted(lang_lines.keys())}")

    # Phase 3: Encode all data -> numpy arrays
    print("\nEncoding sequences ...")
    inp_buf: list = []
    tgt_buf: list = []

    for lang, lines in lang_lines.items():
        stride_div = CJK_STRIDE_DIV if lang in CJK_LANGS else LATIN_STRIDE_DIV

        n_before = len(inp_buf)
        for line in lines:
            line_ids = [BOS_ID]
            line_ids.extend(vocab.encode(c) for c in line)
            line_ids.append(EOS_ID)
            for inp, tgt in _make_sequences(line_ids, model_seq_len, stride_div):
                inp_buf.append(inp)
                tgt_buf.append(tgt)
        print(f"  [{lang}] {len(inp_buf) - n_before:,} sequences")

    # int16 is sufficient: max vocab id < 32 767 for all expected languages
    inputs_arr = np.array(inp_buf, dtype=np.int16)
    targets_arr = np.array(tgt_buf, dtype=np.int16)
    del inp_buf, tgt_buf

    print(
        f"\nTotal sequences: {len(inputs_arr):,}  "
        f"(arrays: {inputs_arr.nbytes / 1024**2:.0f} MB)"
    )

    # Phase 4: Train unified model
    print("\n-- Training unified CharTransformerLM ---------------------------")
    t0 = time.perf_counter()
    model, optimizer, scheduler, scaler, final_epoch = _train_unified(
        inputs_arr=inputs_arr,
        targets_arr=targets_arr,
        vocab=vocab,
        device=device,
        max_seq=model_seq_len,
        d_model=model_d,
        n_heads=model_h,
        n_layers=model_l,
        batch_size=batch_size,
        lr=lr,
        start_epoch=start_epoch,
        target_epochs=target_epochs,
        resume_model_state=(
            resume_model_ckpt["state_dict"] if resume_model_ckpt is not None else None
        ),
        resume_optimizer_state=resume_optimizer_state,
        resume_scheduler_state=resume_scheduler_state,
        resume_scaler_state=resume_scaler_state,
    )
    model.eval()
    print(f"  Training time: {time.perf_counter() - t0:.1f}s")

    # Save checkpoints
    os.makedirs(args.work_dir, exist_ok=True)
    save_path = os.path.join(args.work_dir, "model.pt")
    checkpoint = {
        "vocab": vocab,
        "vocab_size": len(vocab),
        "d_model": model_d,
        "n_heads": model_h,
        "n_layers": model_l,
        "seq_len": model_seq_len,
        "epoch": final_epoch,
        "target_epochs": target_epochs,
        "batch_size": batch_size,
        "lr": lr,
        # Unwrap torch.compile wrapper before saving so load_state_dict works cleanly.
        "state_dict": {
            k: v.cpu()
            for k, v in getattr(model, "_orig_mod", model).state_dict().items()
        },
    }
    torch.save(checkpoint, save_path)

    train_state_path = _default_train_state_path(Path(save_path))
    train_state = {
        "epoch": final_epoch,
        "target_epochs": target_epochs,
        "seq_len": model_seq_len,
        "batch_size": batch_size,
        "lr": lr,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if device.type == "cuda" else None,
        "model_checkpoint_path": save_path,
    }
    torch.save(train_state, str(train_state_path))

    print(
        f"\nTraining complete - {time.perf_counter() - total_start:.1f}s  "
        f"| model: {save_path} | resume state: {train_state_path}"
    )

def _batch_predict_top3(
    model: CharTransformerLM,
    vocab: Vocab,
    texts: list,
    device: torch.device,
    decode_arr: list,  # pre-built list: index → character string
    seq_len: int = SEQ_LEN,
) -> list:
    """
    Predict top-3 next characters for a list of texts in one padded forward pass.
    Returns a list of 3-char strings aligned with `texts`.
    """
    encoded = [_encode_context(vocab, t, seq_len=seq_len) for t in texts]
    lengths = [len(ids) for ids in encoded]
    max_len = max(lengths)
    B = len(texts)

    # Numpy pre-allocation is faster than Python list-of-lists for torch.tensor()
    x_np = np.zeros((B, max_len), dtype=np.int32)
    for i, ids in enumerate(encoded):
        x_np[i, : len(ids)] = ids
    x = torch.from_numpy(x_np).long().to(device)  # (B, T)
    pad_mask = x == PAD_ID  # (B, T)

    use_amp = device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=use_amp):
        logits = model(x, pad_mask)  # (B, T, V)

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

    ckpt_seq_len = _checkpoint_seq_len(checkpoint, SEQ_LEN)
    vocab: Vocab = checkpoint["vocab"]
    model = CharTransformerLM(
        vocab_size=checkpoint["vocab_size"],
        d_model=checkpoint.get("d_model", D_MODEL),
        n_heads=checkpoint.get("n_heads", N_HEADS),
        n_layers=checkpoint.get("n_layers", N_LAYERS),
        max_seq=ckpt_seq_len,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print(
        f"Loaded unified model  vocab={checkpoint['vocab_size']}  seq_len={ckpt_seq_len}"
    )

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

    # Pre-built decode array: avoids a dict lookup per decoded token.
    decode_arr = [vocab.decode(i) for i in range(len(vocab))]

    # ── Normalise all contexts up front ──────────────────────────────
    clean = [unicodedata.normalize("NFC", str(c).strip('"')) for c in contexts]

    # ── Sort by context length — reduces average padding per batch ────
    order = sorted(range(len(clean)), key=lambda i: len(clean[i]))

    # ── Single unified batched pass ───────────────────────────────────
    predictions = [""] * len(clean)
    inf_start = time.perf_counter()

    with torch.no_grad():
        for bi in range(0, len(order), INF_BATCH):
            batch_order = order[bi : bi + INF_BATCH]
            batch_texts = [clean[i] for i in batch_order]
            preds = _batch_predict_top3(
                model, vocab, batch_texts, device, decode_arr, seq_len=ckpt_seq_len
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
    parser.add_argument("--train_dir", default="../data/train v2 (with test)")
    parser.add_argument(
        "--resume_model",
        default=None,
        help="Path to a previous model checkpoint (model.pt) for continued training.",
    )
    parser.add_argument(
        "--resume_state",
        default=None,
        help="Optional optimizer/scaler state checkpoint; defaults to model.train_state.pt next to --resume_model.",
    )
    parser.add_argument(
        "--extra_epochs",
        type=int,
        default=0,
        help="Additional epochs to train when resuming. If omitted, trains for another EPOCHS.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Total epochs for a fresh run; or default extra epochs when resuming.",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=None,
        help="Transformer embedding width for a fresh run.",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=None,
        help="Number of attention heads for a fresh run.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=None,
        help="Number of Transformer layers for a fresh run.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Sequence length for a fresh run.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size override.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning-rate override.",
    )
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

