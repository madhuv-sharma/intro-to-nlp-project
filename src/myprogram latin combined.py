import argparse
import math
import os
import pickle
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

SPECIAL_CHARS = {"^", "$"}


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
        return score

    def prob(self, context, char):
        if char in SPECIAL_CHARS:
            return 0.0
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

    def predict_top3(self, context):
        if context in self.cache:
            return self.cache[context]

        candidates = set()
        for n in self.n_orders:
            if len(context) >= n:
                sub_context = context[-n:]
                if sub_context in self.counts[n]:
                    candidates.update(
                        c
                        for c in self.counts[n][sub_context]
                        if c != "__total__" and c not in SPECIAL_CHARS
                    )

        if not candidates:
            candidates = {c for c in self.vocab if c not in SPECIAL_CHARS}

        scores = sorted(
            ((self.prob(context, char), char) for char in candidates), reverse=True
        )
        result = [c for _, c in scores[:3]]

        if len(result) < 3:
            for char in self.vocab:
                if char not in result and char not in SPECIAL_CHARS:
                    result.append(char)
                if len(result) == 3:
                    break

        if len(self.cache) > 50000:
            self.cache.clear()

        self.cache[context] = result
        return result


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
# Script Detection
# ===============================


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


# ===============================
# Train Mode
# ===============================


def train(args):
    start_time = time.perf_counter()
    train_dir = Path(args.train_dir)

    lms = {}

    script_langs = {"ru", "hi", "ar", "ko", "ja", "zh"}

    # Train script-unique LMs
    for lang in script_langs:
        file = train_dir / f"{lang}.txt"
        if not file.exists():
            continue

        print(f"Training {lang}")

        n_min = args.n_min
        n_max = args.n_max
        alpha = args.alpha

        if lang in {"zh", "ja"}:
            n_min = 1
            n_max = 3
            alpha = 1.5
        elif lang == "ko":
            n_min = 1

        lms[lang] = CharNgramLM(n_min, n_max, alpha)

        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = unicodedata.normalize("NFC", line)
                line = ("^" * n_max) + line + "$"
                lms[lang].train_text(line)

        print(f"Finished {lang}")

    latin_file = train_dir / "latin.txt"
    if latin_file.exists():
        print("Training latin")
        lms["latin"] = CharNgramLM(args.n_min, args.n_max, args.alpha)
        with latin_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = unicodedata.normalize("NFC", line)
                line = ("^" * args.n_max) + line + "$"
                lms["latin"].train_text(line)
        print("Finished latin")

    for lm in lms.values():
        for ch in SPECIAL_CHARS:
            lm.vocab.discard(ch)
        lm.vocab_size = len(lm.vocab)
        lm.log_vocab_size = math.log(lm.vocab_size)

    os.makedirs(args.work_dir, exist_ok=True)
    save_model(lms, os.path.join(args.work_dir, "model.pkl"))

    print(f"Training complete in {time.perf_counter() - start_time:.2f} seconds")


# ===============================
# Test Mode
# ===============================


def test(args):
    total_start = time.perf_counter()
    lms = load_model(os.path.join(args.work_dir, "model.pkl"))

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

    rows = []
    inference_start = time.perf_counter()

    for idx, context in enumerate(contexts):
        if (idx + 1) % 5000 == 0:
            print(f"Predicting {idx + 1}th entry")

        context = unicodedata.normalize("NFC", context)
        detected = detect_script(context)

        if detected and detected in lms:
            best_lang = detected
        else:
            best_lang = "latin"

        lm = lms[best_lang]

        context = ("^" * lm.n_max) + context
        preds = lm.predict_top3(context)
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
    print(f"Inference time: {inference_end - inference_start:.2f} seconds")
    print(f"Total test time: {time.perf_counter() - total_start:.2f} seconds")


# ===============================
# Main
# ===============================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--work_dir", default="../work")
    parser.add_argument("--train_dir", default="../data/train")
    parser.add_argument("--test_data", default="../kaggle-data/test.csv")
    parser.add_argument("--test_output", default="../submission.csv")
    parser.add_argument("--n_min", type=int, default=2)
    parser.add_argument("--n_max", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.3)

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
