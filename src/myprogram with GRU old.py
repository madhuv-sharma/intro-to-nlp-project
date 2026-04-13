import argparse
import math
import os
import pickle
import re
import time
import unicodedata

# import urllib.request
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

# import fasttext
import pandas as pd

# ===============================
# Character Ngram Language Model
# ===============================


class CharNgramLM:
    def __init__(self, n_min, n_max, alpha):
        self.n_min = n_min
        self.n_max = n_max
        self.alpha = alpha
        self.n_orders = list(range(self.n_min, self.n_max + 1))
        self.n_orders_rev = list(reversed(self.n_orders))
        self.counts = {n: defaultdict(Counter) for n in self.n_orders}
        self.vocab = set()
        self.vocab_size = 0
        self.log_vocab_size = 0.0
        self.cache = {}

    def train_text(self, text):
        text_len = len(text)
        for i in range(text_len):
            for n in self.n_orders:
                if i - n < 0:
                    break
                context = text[i - n : i]
                char = text[i]
                self.counts[n][context][char] += 1
                self.counts[n][context]["__total__"] += 1
                self.vocab.add(char)

    def score_context(self, context):
        score = 0.0

        for i in range(1, len(context)):
            char = context[i]
            max_n = min(self.n_max, i)
            found = False
            for n in range(max_n, self.n_min - 1, -1):
                sub_context = context[i - n : i]
                counter = self.counts[n].get(sub_context)
                if not counter:
                    continue

                found = True
                context_count = counter["__total__"]
                char_count = counter.get(char, 0)

                score += math.log(char_count + self.alpha) - math.log(
                    context_count + (self.alpha * self.vocab_size)
                )
                break
            if not found:
                score -= self.log_vocab_size

        # score = score / max(len(context), 1) if context else 0.0
        return score

    def prob(self, context, char):
        for n in self.n_orders_rev:
            if len(context) < n:
                continue

            sub_context = context[-n:]
            counter = self.counts[n].get(sub_context)
            if not counter:
                continue
            context_count = counter["__total__"]
            char_count = counter.get(char, 0)
            return (char_count + self.alpha) / (
                context_count + (self.alpha * self.vocab_size)
            )
        return 1 / self.vocab_size if self.vocab_size else 0.0

    @lru_cache(maxsize=50000)
    def _predict_top3_cached(self, context):
        candidates = set()

        # collect possible next chars from all n-gram orders
        for n in self.n_orders:
            if len(context) >= n:
                sub_context = context[-n:]
                if sub_context in self.counts[n]:
                    candidates.update(
                        c
                        for c in self.counts[n][sub_context].keys()
                        if c != "__total__"
                    )

        # fallback if unseen
        if not candidates:
            candidates = self.vocab

        scores = []
        for char in candidates:
            scores.append((self.prob(context, char), char))

        scores.sort(reverse=True)
        result = tuple(c for _, c in scores[:3])

        if len(result) < 3:
            for char in self.vocab:
                if char not in result:
                    result = result + (char,)
                if len(result) == 3:
                    break

        return result

    def predict_top3(self, context):
        return list(self._predict_top3_cached(context))


# ===============================
# Character GRU
# ===============================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


class CharGRULM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 96, hidden_dim: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_ids: torch.Tensor):
        # x_ids: [B, T]
        h = self.emb(x_ids)  # [B, T, E]
        out, _ = self.gru(h)  # [B, T, H]
        logits = self.proj(out)  # [B, T, V]
        return logits


class CharSeqDataset(Dataset):
    def __init__(self, text: str, char2id: dict[str, int], seq_len: int = 256):
        self.seq_len = seq_len
        self.char2id = char2id
        self.unk = char2id["<UNK>"]
        ids = [char2id.get(ch, self.unk) for ch in text]
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.ids) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.ids[start : start + self.seq_len]
        y = self.ids[start + 1 : start + self.seq_len + 1]
        return x, y


def train_gru_lm(
    text: str,
    char2id: dict[str, int],
    device: str,
    epochs: int = 2,
    seq_len: int = 256,
    batch_size: int = 64,
    lr: float = 2e-3,
):

    ds = CharSeqDataset(text, char2id, seq_len=seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = CharGRULM(vocab_size=len(char2id)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        steps = 0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # [B, T, V]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item())
            steps += 1

        print(f"epoch {ep+1} loss {(total_loss/max(steps,1)):.4f}")

    return model


@torch.no_grad()
def gru_next_char_probs(
    model, context: str, char2id: dict[str, int], device: str, window: int = 256
):
    model.eval()
    unk = char2id["<UNK>"]
    ids = [char2id.get(ch, unk) for ch in context[-window:]]
    if not ids:
        ids = [char2id.get("^", unk)]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1,T]
    logits = model(x)[:, -1, :]  # [1,V]
    probs = torch.softmax(logits, dim=-1).squeeze(0)  # [V]
    return probs


def hybrid_predict_top3(
    ngram_lm,
    gru_model,
    char2id,
    id2char,
    context: str,
    device: str,
    mix_lambda: float = 0.7,
    gru_topk: int = 200,
):

    gru_probs = gru_next_char_probs(gru_model, context, char2id, device=device)

    # candidates from ngram contexts
    candidates = set()
    for n in ngram_lm.n_orders:
        if len(context) >= n:
            sub = context[-n:]
            if sub in ngram_lm.counts[n]:
                candidates.update(
                    c for c in ngram_lm.counts[n][sub] if c != "__total__"
                )

    # add topK from GRU
    topk_ids = torch.topk(
        gru_probs, k=min(gru_topk, gru_probs.numel())
    ).indices.tolist()
    for cid in topk_ids:
        candidates.add(id2char[cid])

    if not candidates:
        candidates = ngram_lm.vocab

    scored = []
    for ch in candidates:
        p_ng = ngram_lm.prob(context, ch)
        cid = char2id.get(ch, char2id["<UNK>"])
        p_gru = float(gru_probs[cid].item())
        p = mix_lambda * p_ng + (1.0 - mix_lambda) * p_gru
        scored.append((p, ch))

    scored.sort(reverse=True)
    return [ch for _, ch in scored[:3]]


# ===============================
# Utilities
# ===============================


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ===============================
# Train Mode
# ===============================


def train(args):
    start_time = time.perf_counter()

    os.makedirs(args.work_dir, exist_ok=True)
    train_dir = Path(args.train_dir)

    lms = {}
    for file in train_dir.glob("*.txt"):
        lang = file.stem
        print(f"Training {lang}")
        n_min = args.n_min
        n_max = args.n_max
        alpha = args.alpha
        if lang in ["zh", "ja"]:
            n_min = 1
            n_max = 3
            alpha = 1.5
        elif lang == "ko":
            n_min = 1
        lms[lang] = CharNgramLM(n_min=n_min, n_max=n_max, alpha=alpha)

        with file.open("r", encoding="utf-8") as f:
            for line in f:
                # for i, line in enumerate(f):
                # if (i + 1) % 10000 == 0:
                #     print(f"Processing line {i + 1}")
                line = line.strip()
                if not line:
                    continue
                line = unicodedata.normalize("NFC", line)
                line = ("^" * lms[lang].n_max) + line + "$"
                lms[lang].train_text(line)

        # text = file.read_text(encoding="utf-8")
        # lms[lang].train_text(text)

        print(f"Finished training N-gram for {lang}")

        text = ""
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = unicodedata.normalize("NFC", line)
                line = ("^" * n_max) + line + "$"
                text += line

        # build vocab
        chars = sorted(set(text))
        chars = ["<UNK>"] + chars
        char2id = {c: i for i, c in enumerate(chars)}
        id2char = {i: c for c, i in char2id.items()}

        gru_model = train_gru_lm(text, char2id, device, epochs=3)

        torch.save(
            gru_model.state_dict(), os.path.join(args.work_dir, f"gru_{lang}.pt")
        )
        pickle.dump(
            char2id, open(os.path.join(args.work_dir, f"vocab_{lang}.pkl"), "wb")
        )
        print(f"Finished training GRU for {lang}")

    for lm in lms.values():
        lm.vocab.discard("^")
        lm.vocab.discard("$")
        lm.vocab_size = len(lm.vocab)
        lm.log_vocab_size = math.log(lm.vocab_size)

    save_model(lms, os.path.join(args.work_dir, "model.pkl"))

    end_time = time.perf_counter()

    print("Training complete")
    print(f"Training time: {end_time - start_time:.2f} seconds")


# ===============================
# Test Mode
# ===============================


def normalize_caps(text):
    def fix_word(match):
        word = match.group(0)

        if len(word) <= 3:
            return word

        return word.lower()

    return re.sub(r"\b[A-Z]{2,}\b", fix_word, text)


def detect_script(text):
    counts = {
        "ru": 0,
        "hi": 0,
        "ar": 0,
        "ko": 0,
        "ja": 0,
        "zh": 0,
    }

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


def test(args):
    total_start = time.perf_counter()

    gru_models = {}
    char2id_map = {}
    id2char_map = {}
    lms = load_model(os.path.join(args.work_dir, "model.pkl"))

    for lang in lms:
        path = os.path.join(args.work_dir, f"gru_{lang}.pt")
        vocab_path = os.path.join(args.work_dir, f"vocab_{lang}.pkl")
        if os.path.exists(path):
            char2id = pickle.load(open(vocab_path, "rb"))
            id2char = {i: c for c, i in char2id.items()}

            model = CharGRULM(len(char2id))
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()

            gru_models[lang] = model
            char2id_map[lang] = char2id
            id2char_map[lang] = id2char

    if args.test_data.endswith(".csv"):
        test_df = pd.read_csv(args.test_data)
        contexts = test_df["context"].tolist()
        ids = test_df["id"].tolist()
        is_csv = True
    else:
        with open(args.test_data, "r", encoding="utf-8") as f:
            contexts = [line.strip() for line in f if line.strip()]
        ids = None
        is_csv = False

    # # Optional test-time adaptation
    # if not args.no_adapt:
    #     adapt_start = time.perf_counter()

    #     adapt_lm = CharNgramLM(n_max=lm.n_max)
    #     for context in test_df["context"]:
    #         adapt_lm.train_text(context)

    #     original_prob = lm.prob

    #     def blended_prob(context, char):
    #         return 0.7 * original_prob(context, char) + 0.3 * adapt_lm.prob(
    #             context, char
    #         )

    #     lm.prob = blended_prob

    #     adapt_end = time.perf_counter()
    #     print(f"Adaptation time: {adapt_end - adapt_start:.2f} seconds")

    rows = []

    # if not os.path.exists("lid.176.bin"):
    #     print("Downloading fastText language ID model...")
    #     urllib.request.urlretrieve(
    #         "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    #         "lid.176.bin",
    #     )
    # ft_model = fasttext.load_model("lid.176.bin")

    inference_start = time.perf_counter()

    for idx, context in enumerate(contexts):
        if (idx + 1) % 1000 == 0:
            print(f"Predicting {idx + 1}th entry")

        best_lang = None
        best_score = float("-inf")

        context = unicodedata.normalize("NFC", context)
        # TODO: uncomment with normalized caps data
        # context = normalize_caps(context)
        detected = detect_script(context)

        if detected and detected in lms:
            candidate_langs = [detected]
        else:
            # labels, probs = ft_model.predict(context, k=1)
            # label = labels[0]
            # conf = float(probs[0])
            # predicted_lang = label.replace("__label__", "")

            # latin_langs = {"en", "fr", "de", "it"}

            # if predicted_lang in latin_langs and conf > 0.8:
            #     candidate_langs = [predicted_lang]
            # else:
            #     candidate_langs = latin_langs
            candidate_langs = lms.keys()

        for lang in candidate_langs:
            lm = lms[lang]
            s = lm.score_context(context)
            if s > best_score:
                best_score = s
                best_lang = lang

        # preds = lms[best_lang].predict_top3(context)

        if best_lang in ["zh", "ja", "ko"]:
            mix_lambda = 0.55
        else:
            mix_lambda = 0.7
        preds = hybrid_predict_top3(
            lms[best_lang],
            gru_models[best_lang],
            char2id_map[best_lang],
            id2char_map[best_lang],
            context,
            device,
            mix_lambda=mix_lambda,
        )
        pred_str = "".join(preds)
        if is_csv:
            rows.append({"id": ids[idx], "prediction": pred_str})
        else:
            rows.append(pred_str)

    inference_end = time.perf_counter()

    if is_csv:
        pd.DataFrame(rows).to_csv(args.test_output, index=False)
    else:
        with open(args.test_output, "w", encoding="utf-8") as f:
            for pred in rows:
                f.write(pred + "\n")

    print("Submission file created")

    total_end = time.perf_counter()

    print(f"Inference time (total): {inference_end - inference_start:.2f} seconds")
    print(
        f"Average time per example: {(inference_end - inference_start)/len(contexts):.6f} seconds"
    )
    print(f"Total test time: {total_end - total_start:.2f} seconds")


# ===============================
# Main
# ===============================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--work_dir", default="../work")
    parser.add_argument("--train_dir", default="../data/train")
    # parser.add_argument("--test_csv", default="../kaggle-data/test.csv")
    # parser.add_argument("--output_csv", default="../submission.csv")
    parser.add_argument("--test_data", default="../kaggle-data/test.csv")
    parser.add_argument("--test_output", default="../submission.csv")
    parser.add_argument("--n_min", type=int, default=2)
    parser.add_argument("--n_max", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--no_adapt", action="store_true")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
