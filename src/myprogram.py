import argparse
import math
import os
import pickle
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===============================
# Character Ngram Language Model
# ===============================

SPECIAL_CHARS = {"^", "$"}
CJK_LANGS = ("zh", "ja", "ko")


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
                        for c in self.counts[n][sub_context].keys()
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
                if char in SPECIAL_CHARS:
                    continue
                if char not in result:
                    result.append(char)
                if len(result) == 3:
                    break

        if len(self.cache) > 50000:
            print("Cache size exceeded, clearing cache")
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


def parse_int_list(value):
    values = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def parse_float_list(value):
    values = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def read_lines_keep_empty(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f]


def maybe_clean_text(text, use_data_cleaning):
    if not use_data_cleaning:
        return text
    text = text.strip('"')
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u0000-\u001F\u007F]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def resolve_lang_hyperparams(
    lang,
    n_min,
    n_max,
    alpha,
    enable_cjk_tuning,
    cjk_n_min,
    cjk_n_max,
    cjk_alpha,
    ko_alpha,
):
    if not enable_cjk_tuning:
        return n_min, n_max, alpha

    if lang in ("zh", "ja"):
        return cjk_n_min, cjk_n_max, cjk_alpha
    if lang == "ko":
        ko_alpha_value = ko_alpha if ko_alpha is not None else alpha
        return cjk_n_min, cjk_n_max, ko_alpha_value
    return n_min, n_max, alpha

def train_language_models(
    train_dir,
    n_min,
    n_max,
    alpha,
    use_data_cleaning,
    enable_cjk_tuning,
    cjk_n_min,
    cjk_n_max,
    cjk_alpha,
    ko_alpha,
):
    train_dir = Path(train_dir)
    files = sorted(train_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt training files found in {train_dir}")

    lms = {}
    for file in files:
        lang = file.stem
        lang_n_min, lang_n_max, lang_alpha = resolve_lang_hyperparams(
            lang=lang,
            n_min=n_min,
            n_max=n_max,
            alpha=alpha,
            enable_cjk_tuning=enable_cjk_tuning,
            cjk_n_min=cjk_n_min,
            cjk_n_max=cjk_n_max,
            cjk_alpha=cjk_alpha,
            ko_alpha=ko_alpha,
        )
        print(
            f"Training {lang} "
            f"(n_min={lang_n_min}, n_max={lang_n_max}, alpha={lang_alpha})"
        )
        lm = CharNgramLM(n_min=lang_n_min, n_max=lang_n_max, alpha=lang_alpha)
        with file.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                line = maybe_clean_text(line, use_data_cleaning=use_data_cleaning)
                if not line:
                    continue
                line = ("^" * lm.n_max) + line + "$"
                lm.train_text(line)
        lms[lang] = lm
        print(f"Finished training {lang}")

    for lm in lms.values():
        for ch in SPECIAL_CHARS:
            lm.vocab.discard(ch)
        lm.vocab_size = len(lm.vocab)
        lm.log_vocab_size = math.log(lm.vocab_size) if lm.vocab_size else 0.0

    return lms


def predict_for_context(context, lms, use_data_cleaning):
    if not lms:
        raise ValueError("Language models are empty. Train/load models first.")

    context = maybe_clean_text(context, use_data_cleaning=use_data_cleaning)
    detected = detect_script(context)

    if detected and detected in lms:
        candidate_langs = [detected]
    else:
        latin_langs = [lang for lang in ("en", "fr", "de", "it") if lang in lms]
        candidate_langs = latin_langs if latin_langs else list(lms.keys())

    best_lang = None
    best_score = float("-inf")

    for lang in candidate_langs:
        lm = lms[lang]
        score = lm.score_context(("^" * lm.n_max) + context)
        if score > best_score:
            best_score = score
            best_lang = lang

    if best_lang is None:
        best_lang = next(iter(lms))

    pred_context = ("^" * lms[best_lang].n_max) + context
    preds = lms[best_lang].predict_top3(pred_context)
    return "".join(preds), best_lang


def load_eval_triplet(eval_input, eval_answer, eval_lang, max_eval_rows):
    contexts = read_lines_keep_empty(eval_input)
    answers = read_lines_keep_empty(eval_answer)
    langs = read_lines_keep_empty(eval_lang)

    if not (len(contexts) == len(answers) == len(langs)):
        raise ValueError(
            "Mismatched lengths among eval files: "
            f"input={len(contexts)}, answer={len(answers)}, lang={len(langs)}"
        )

    if max_eval_rows and max_eval_rows > 0:
        contexts = contexts[:max_eval_rows]
        answers = answers[:max_eval_rows]
        langs = langs[:max_eval_rows]

    return contexts, answers, langs


def evaluate_model(lms, contexts, answers, langs, use_data_cleaning):
    top1_correct = 0
    top3_correct = 0
    per_lang_counts = defaultdict(lambda: {"top1": 0, "top3": 0, "total": 0})

    eval_start = time.perf_counter()
    for idx, (context, gold_char, lang) in enumerate(zip(contexts, answers, langs)):
        if (idx + 1) % 5000 == 0:
            print(f"Evaluating {idx + 1} examples")

        pred_str, _ = predict_for_context(
            context=context,
            lms=lms,
            use_data_cleaning=use_data_cleaning,
        )
        pred_str = pred_str.lower()
        gold_char = gold_char.lower()

        is_top1 = bool(pred_str) and pred_str[0] == gold_char
        is_top3 = gold_char in pred_str[:3]

        top1_correct += int(is_top1)
        top3_correct += int(is_top3)

        stats = per_lang_counts[lang]
        stats["total"] += 1
        stats["top1"] += int(is_top1)
        stats["top3"] += int(is_top3)

    eval_end = time.perf_counter()

    total = len(contexts)
    per_lang = {}
    for lang, stats in sorted(per_lang_counts.items()):
        lang_total = stats["total"]
        per_lang[lang] = {
            "top1_correct": stats["top1"],
            "top3_correct": stats["top3"],
            "total": lang_total,
            "top1_acc": (stats["top1"] / lang_total) if lang_total else 0.0,
            "top3_acc": (stats["top3"] / lang_total) if lang_total else 0.0,
        }

    cjk_scores = [per_lang[lang]["top3_acc"] for lang in CJK_LANGS if lang in per_lang]
    top3_cjk_macro = sum(cjk_scores) / len(cjk_scores) if cjk_scores else 0.0

    return {
        "num_examples": total,
        "top1_correct": top1_correct,
        "top3_correct": top3_correct,
        "top1_acc": (top1_correct / total) if total else 0.0,
        "top3_acc": (top3_correct / total) if total else 0.0,
        "top3_cjk_macro": top3_cjk_macro,
        "per_lang": per_lang,
        "eval_time_sec": eval_end - eval_start,
    }


def make_config_key(
    n_min,
    n_max,
    alpha,
    use_data_cleaning,
    enable_cjk_tuning,
    cjk_n_min,
    cjk_n_max,
    cjk_alpha,
    ko_alpha,
):
    ko_alpha_key = "none" if ko_alpha is None else round(float(ko_alpha), 8)
    return (
        n_min,
        n_max,
        round(float(alpha), 8),
        bool(use_data_cleaning),
        bool(enable_cjk_tuning),
        cjk_n_min,
        cjk_n_max,
        round(float(cjk_alpha), 8),
        ko_alpha_key,
    )


def evaluate_config_with_cache(
    cache,
    train_dir,
    contexts,
    answers,
    langs,
    n_min,
    n_max,
    alpha,
    use_data_cleaning,
    enable_cjk_tuning,
    cjk_n_min,
    cjk_n_max,
    cjk_alpha,
    ko_alpha,
):
    key = make_config_key(
        n_min=n_min,
        n_max=n_max,
        alpha=alpha,
        use_data_cleaning=use_data_cleaning,
        enable_cjk_tuning=enable_cjk_tuning,
        cjk_n_min=cjk_n_min,
        cjk_n_max=cjk_n_max,
        cjk_alpha=cjk_alpha,
        ko_alpha=ko_alpha,
    )
    if key in cache:
        print(
            "Using cached metrics for "
            f"(n_min={n_min}, n_max={n_max}, alpha={alpha}, "
            f"clean={use_data_cleaning}, cjk_tuned={enable_cjk_tuning})"
        )
        return cache[key]

    print(
        "Running config "
        f"(n_min={n_min}, n_max={n_max}, alpha={alpha}, "
        f"clean={use_data_cleaning}, cjk_tuned={enable_cjk_tuning})"
    )
    train_start = time.perf_counter()
    lms = train_language_models(
        train_dir=train_dir,
        n_min=n_min,
        n_max=n_max,
        alpha=alpha,
        use_data_cleaning=use_data_cleaning,
        enable_cjk_tuning=enable_cjk_tuning,
        cjk_n_min=cjk_n_min,
        cjk_n_max=cjk_n_max,
        cjk_alpha=cjk_alpha,
        ko_alpha=ko_alpha,
    )
    train_end = time.perf_counter()

    metrics = evaluate_model(
        lms=lms,
        contexts=contexts,
        answers=answers,
        langs=langs,
        use_data_cleaning=use_data_cleaning,
    )
    metrics["train_time_sec"] = train_end - train_start
    metrics["config"] = {
        "n_min": n_min,
        "n_max": n_max,
        "alpha": alpha,
        "use_data_cleaning": use_data_cleaning,
        "enable_cjk_tuning": enable_cjk_tuning,
        "cjk_n_min": cjk_n_min,
        "cjk_n_max": cjk_n_max,
        "cjk_alpha": cjk_alpha,
        "ko_alpha": ko_alpha,
    }

    cache[key] = metrics
    return metrics


def lang_top3(metrics, lang):
    return metrics["per_lang"].get(lang, {}).get("top3_acc", 0.0)


def _fmt_num(value):
    return f"{value:g}"

def plot_topk_vs_nmax(nmax_values, top1_values, top3_values, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nmax_values, top1_values, marker="o", linewidth=2, label="Top-1 Accuracy")
    ax.plot(nmax_values, top3_values, marker="s", linewidth=2, label="Top-3 Accuracy")
    ax.set_title("Top-1 and Top-3 Accuracy vs n_max")
    ax.set_xlabel("n_max (n_min=2, alpha=0.3)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(nmax_values)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_top3_vs_alpha(alpha_values, top3_values, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alpha_values, top3_values, marker="o", linewidth=2, color="#2a9d8f")
    ax.set_title("Effect of Alpha on Top-3 Accuracy")
    ax.set_xlabel("alpha (n_min=2, n_max=5)")
    ax.set_ylabel("Top-3 Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(alpha_values)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_top3_heatmap(nmax_values, alpha_values, matrix, output_path):
    heatmap = np.array(matrix)
    fig_width = max(8, 0.95 * len(alpha_values) + 4)
    fig_height = max(5, 0.8 * len(nmax_values) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heatmap, cmap="YlGnBu", aspect="auto")

    ax.set_title("Top-3 Accuracy Heatmap")
    ax.set_xlabel("alpha")
    ax.set_ylabel("n_max (n_min=2)")
    ax.set_xticks(np.arange(len(alpha_values)))
    ax.set_yticks(np.arange(len(nmax_values)))
    ax.set_xticklabels([_fmt_num(a) for a in alpha_values])
    ax.set_yticklabels([str(n) for n in nmax_values])

    for i in range(len(nmax_values)):
        for j in range(len(alpha_values)):
            ax.text(
                j,
                i,
                f"{heatmap[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Top-3 Accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_cjk_tuning_comparison(base_metrics, tuned_metrics, output_path):
    labels = ["overall", "zh", "ja", "ko"]
    base_vals = [
        base_metrics["top3_acc"],
        lang_top3(base_metrics, "zh"),
        lang_top3(base_metrics, "ja"),
        lang_top3(base_metrics, "ko"),
    ]
    tuned_vals = [
        tuned_metrics["top3_acc"],
        lang_top3(tuned_metrics, "zh"),
        lang_top3(tuned_metrics, "ja"),
        lang_top3(tuned_metrics, "ko"),
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, base_vals, width, label="No CJK tuning", color="#457b9d")
    ax.bar(x + width / 2, tuned_vals, width, label="With CJK tuning", color="#e76f51")

    ax.set_title("Top-3 Accuracy: CJK Tuning Ablation")
    ax.set_xlabel("Evaluation subset")
    ax.set_ylabel("Top-3 Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_cleaning_comparison(no_clean_metrics, clean_metrics, output_path):
    labels = ["overall", "zh", "ja", "ko"]
    no_clean_vals = [
        no_clean_metrics["top3_acc"],
        lang_top3(no_clean_metrics, "zh"),
        lang_top3(no_clean_metrics, "ja"),
        lang_top3(no_clean_metrics, "ko"),
    ]
    clean_vals = [
        clean_metrics["top3_acc"],
        lang_top3(clean_metrics, "zh"),
        lang_top3(clean_metrics, "ja"),
        lang_top3(clean_metrics, "ko"),
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, no_clean_vals, width, label="No data cleaning", color="#6d597a")
    ax.bar(x + width / 2, clean_vals, width, label="With data cleaning", color="#2a9d8f")

    ax.set_title("Top-3 Accuracy: Data Cleaning Ablation")
    ax.set_xlabel("Evaluation subset")
    ax.set_ylabel("Top-3 Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_metrics_table(cache, output_csv):
    rows = []
    for metrics in cache.values():
        cfg = metrics["config"]
        rows.append(
            {
                "n_min": cfg["n_min"],
                "n_max": cfg["n_max"],
                "alpha": cfg["alpha"],
                "use_data_cleaning": cfg["use_data_cleaning"],
                "enable_cjk_tuning": cfg["enable_cjk_tuning"],
                "cjk_n_min": cfg["cjk_n_min"],
                "cjk_n_max": cfg["cjk_n_max"],
                "cjk_alpha": cfg["cjk_alpha"],
                "ko_alpha": cfg["ko_alpha"],
                "num_examples": metrics["num_examples"],
                "top1_acc": metrics["top1_acc"],
                "top3_acc": metrics["top3_acc"],
                "top3_cjk_macro": metrics["top3_cjk_macro"],
                "top3_zh": lang_top3(metrics, "zh"),
                "top3_ja": lang_top3(metrics, "ja"),
                "top3_ko": lang_top3(metrics, "ko"),
                "train_time_sec": metrics["train_time_sec"],
                "eval_time_sec": metrics["eval_time_sec"],
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            by=["enable_cjk_tuning", "use_data_cleaning", "n_max", "alpha"],
            ascending=[False, False, True, True],
        )
    table.to_csv(output_csv, index=False)

# ===============================
# Train Mode
# ===============================


def train(args):
    start_time = time.perf_counter()
    lms = train_language_models(
        train_dir=args.train_dir,
        n_min=args.n_min,
        n_max=args.n_max,
        alpha=args.alpha,
        use_data_cleaning=(not args.no_data_cleaning),
        enable_cjk_tuning=(not args.disable_cjk_tuning),
        cjk_n_min=args.cjk_n_min,
        cjk_n_max=args.cjk_n_max,
        cjk_alpha=args.cjk_alpha,
        ko_alpha=args.ko_alpha,
    )
    os.makedirs(args.work_dir, exist_ok=True)
    save_model(lms, os.path.join(args.work_dir, "model.pkl"))

    end_time = time.perf_counter()
    print("Training complete")
    print(f"Training time: {end_time - start_time:.2f} seconds")


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
        contexts = read_lines_keep_empty(args.test_data)
        ids = None
        is_csv = False

    rows = []
    inference_start = time.perf_counter()
    for idx, context in enumerate(contexts):
        if (idx + 1) % 5000 == 0:
            print(f"Predicting {idx + 1}th entry")

        pred_str, _ = predict_for_context(
            context=context,
            lms=lms,
            use_data_cleaning=(not args.no_data_cleaning),
        )
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
    if contexts:
        print(
            f"Average time per example: "
            f"{(inference_end - inference_start) / len(contexts):.6f} seconds"
        )
    print(f"Total test time: {total_end - total_start:.2f} seconds")


# ===============================
# Plot Mode
# ===============================


def plot(args):
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    nmax_values = parse_int_list(args.plot_nmax_values)
    alpha_values = parse_float_list(args.plot_alpha_values)

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    contexts, answers, langs = load_eval_triplet(
        eval_input=args.eval_input,
        eval_answer=args.eval_answer,
        eval_lang=args.eval_lang,
        max_eval_rows=args.max_eval_rows,
    )
    print(f"Loaded {len(contexts)} eval examples")

    cache = {}

    sweep_use_data_cleaning = not args.no_data_cleaning
    sweep_use_cjk_tuning = args.plot_sweep_use_cjk_tuning

    # 1) Top-3 and Top-1 as n_max varies
    nmax_top1_scores = []
    nmax_top3_scores = []
    for nmax in nmax_values:
        metrics = evaluate_config_with_cache(
            cache=cache,
            train_dir=args.train_dir,
            contexts=contexts,
            answers=answers,
            langs=langs,
            n_min=args.n_min,
            n_max=nmax,
            alpha=args.plot_sweep_alpha,
            use_data_cleaning=sweep_use_data_cleaning,
            enable_cjk_tuning=sweep_use_cjk_tuning,
            cjk_n_min=args.cjk_n_min,
            cjk_n_max=args.cjk_n_max,
            cjk_alpha=args.cjk_alpha,
            ko_alpha=args.ko_alpha,
        )
        nmax_top1_scores.append(metrics["top1_acc"])
        nmax_top3_scores.append(metrics["top3_acc"])

    plot_topk_vs_nmax(
        nmax_values=nmax_values,
        top1_values=nmax_top1_scores,
        top3_values=nmax_top3_scores,
        output_path=plots_dir / "top1_top3_vs_nmax.png",
    )

    # 2) Effect of alpha on top-3 accuracy
    alpha_top3_scores = []
    for alpha in alpha_values:
        metrics = evaluate_config_with_cache(
            cache=cache,
            train_dir=args.train_dir,
            contexts=contexts,
            answers=answers,
            langs=langs,
            n_min=args.n_min,
            n_max=args.plot_sweep_n_max,
            alpha=alpha,
            use_data_cleaning=sweep_use_data_cleaning,
            enable_cjk_tuning=sweep_use_cjk_tuning,
            cjk_n_min=args.cjk_n_min,
            cjk_n_max=args.cjk_n_max,
            cjk_alpha=args.cjk_alpha,
            ko_alpha=args.ko_alpha,
        )
        alpha_top3_scores.append(metrics["top3_acc"])

    plot_top3_vs_alpha(
        alpha_values=alpha_values,
        top3_values=alpha_top3_scores,
        output_path=plots_dir / "top3_vs_alpha.png",
    )

    # 3) Top-3 heatmap across n_max and alpha
    heatmap = []
    for nmax in nmax_values:
        row = []
        for alpha in alpha_values:
            metrics = evaluate_config_with_cache(
                cache=cache,
                train_dir=args.train_dir,
                contexts=contexts,
                answers=answers,
                langs=langs,
                n_min=args.n_min,
                n_max=nmax,
                alpha=alpha,
                use_data_cleaning=sweep_use_data_cleaning,
                enable_cjk_tuning=sweep_use_cjk_tuning,
                cjk_n_min=args.cjk_n_min,
                cjk_n_max=args.cjk_n_max,
                cjk_alpha=args.cjk_alpha,
                ko_alpha=args.ko_alpha,
            )
            row.append(metrics["top3_acc"])
        heatmap.append(row)

    plot_top3_heatmap(
        nmax_values=nmax_values,
        alpha_values=alpha_values,
        matrix=heatmap,
        output_path=plots_dir / "top3_heatmap_nmax_alpha.png",
    )

    # 4) CJK tuning ablation
    cjk_base = evaluate_config_with_cache(
        cache=cache,
        train_dir=args.train_dir,
        contexts=contexts,
        answers=answers,
        langs=langs,
        n_min=args.n_min,
        n_max=args.plot_sweep_n_max,
        alpha=args.plot_sweep_alpha,
        use_data_cleaning=True,
        enable_cjk_tuning=False,
        cjk_n_min=args.cjk_n_min,
        cjk_n_max=args.cjk_n_max,
        cjk_alpha=args.cjk_alpha,
        ko_alpha=args.ko_alpha,
    )
    cjk_tuned = evaluate_config_with_cache(
        cache=cache,
        train_dir=args.train_dir,
        contexts=contexts,
        answers=answers,
        langs=langs,
        n_min=args.n_min,
        n_max=args.plot_sweep_n_max,
        alpha=args.plot_sweep_alpha,
        use_data_cleaning=True,
        enable_cjk_tuning=True,
        cjk_n_min=args.cjk_n_min,
        cjk_n_max=args.cjk_n_max,
        cjk_alpha=args.cjk_alpha,
        ko_alpha=args.ko_alpha,
    )
    plot_cjk_tuning_comparison(
        base_metrics=cjk_base,
        tuned_metrics=cjk_tuned,
        output_path=plots_dir / "cjk_tuning_top3.png",
    )

    # 5) Data cleaning ablation
    no_clean = evaluate_config_with_cache(
        cache=cache,
        train_dir=args.train_dir,
        contexts=contexts,
        answers=answers,
        langs=langs,
        n_min=args.n_min,
        n_max=args.plot_sweep_n_max,
        alpha=args.plot_sweep_alpha,
        use_data_cleaning=False,
        enable_cjk_tuning=True,
        cjk_n_min=args.cjk_n_min,
        cjk_n_max=args.cjk_n_max,
        cjk_alpha=args.cjk_alpha,
        ko_alpha=args.ko_alpha,
    )
    with_clean = evaluate_config_with_cache(
        cache=cache,
        train_dir=args.train_dir,
        contexts=contexts,
        answers=answers,
        langs=langs,
        n_min=args.n_min,
        n_max=args.plot_sweep_n_max,
        alpha=args.plot_sweep_alpha,
        use_data_cleaning=True,
        enable_cjk_tuning=True,
        cjk_n_min=args.cjk_n_min,
        cjk_n_max=args.cjk_n_max,
        cjk_alpha=args.cjk_alpha,
        ko_alpha=args.ko_alpha,
    )
    plot_cleaning_comparison(
        no_clean_metrics=no_clean,
        clean_metrics=with_clean,
        output_path=plots_dir / "cleaning_ablation_top3.png",
    )

    write_metrics_table(cache=cache, output_csv=plots_dir / "experiment_metrics.csv")
    print(f"Saved figures and metrics to {plots_dir}")


# ===============================
# Main
# ===============================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test", "plot"])
    parser.add_argument("--work_dir", default="../work")
    parser.add_argument("--train_dir", default="../data/train")
    parser.add_argument("--test_data", default="../kaggle-data/test.csv")
    parser.add_argument("--test_output", default="../submission.csv")

    parser.add_argument("--n_min", type=int, default=2)
    parser.add_argument("--n_max", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.3)

    parser.add_argument("--disable_cjk_tuning", action="store_true")
    parser.add_argument("--cjk_n_min", type=int, default=1)
    parser.add_argument("--cjk_n_max", type=int, default=3)
    parser.add_argument("--cjk_alpha", type=float, default=1.5)
    parser.add_argument("--ko_alpha", type=float, default=None)
    parser.add_argument("--no_data_cleaning", action="store_true")

    parser.add_argument("--eval_input", default="../data/open-dev/input.txt")
    parser.add_argument("--eval_answer", default="../data/open-dev/answer.txt")
    parser.add_argument("--eval_lang", default="../data/open-dev/lang.txt")
    parser.add_argument("--plots_dir", default="../plots")
    parser.add_argument("--max_eval_rows", type=int, default=0)
    parser.add_argument("--plot_nmax_values", default="3,4,5,6,7")
    parser.add_argument("--plot_alpha_values", default="0.05,0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--plot_sweep_alpha", type=float, default=0.3)
    parser.add_argument("--plot_sweep_n_max", type=int, default=5)
    parser.add_argument("--plot_sweep_use_cjk_tuning", action="store_true")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
