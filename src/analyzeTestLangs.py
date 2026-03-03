import csv
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
import langid

# =========================
# CONFIG
# =========================

TEST_CSV = Path("../kaggle-data/test.csv")
LATIN_CONF_THRESHOLD = 0.80
HIGH_CONF_THRESHOLD = 0.95

# Restrict to likely Latin languages in your dataset
langid.set_languages(["en", "fr", "de", "it", "tr", "fi", "hu", "vi", "pt", "es", "nl"])

# =========================
# DATA STRUCTURES
# =========================

script_distribution = Counter()
latin_predictions = Counter()
latin_low_conf = []

samples = defaultdict(list)
high_conf_samples = defaultdict(list)

total_lines = 0

# =========================
# UNICODE SCRIPT DETECTOR
# =========================


def detect_script(text):
    counts = Counter()
    total_letters = 0

    for ch in text:
        if ch.isalpha():
            total_letters += 1
            try:
                name = unicodedata.name(ch)

                if "LATIN" in name:
                    counts["LATIN"] += 1
                elif "GREEK" in name:
                    counts["GREEK"] += 1
                elif "CYRILLIC" in name:
                    counts["CYRILLIC"] += 1
                elif "ARABIC" in name:
                    counts["ARABIC"] += 1
                elif "HANGUL" in name:
                    counts["HANGUL"] += 1
                elif "HIRAGANA" in name or "KATAKANA" in name:
                    counts["JA"] += 1
                elif "CJK" in name:
                    counts["ZH"] += 1
                elif "DEVANAGARI" in name:
                    counts["HI"] += 1

            except ValueError:
                continue

    if total_letters == 0:
        return None

    if not counts:
        return None

    script, count = counts.most_common(1)[0]

    if count / total_letters < 0.5:
        return "MIXED"

    return script


# =========================
# MAIN ANALYSIS
# =========================

with TEST_CSV.open("r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header safely

    for row in reader:
        if len(row) < 2:
            continue

        line = row[1].strip()
        if not line:
            continue

        total_lines += 1

        script = detect_script(line)
        script_distribution[script] += 1

        if script == "LATIN":
            lang, confidence = langid.classify(line)
            latin_predictions[lang] += 1

            # Store up to 5 general samples per language
            if len(samples[lang]) < 5:
                samples[lang].append((confidence, line))

            # Store high-confidence samples
            if confidence >= HIGH_CONF_THRESHOLD and len(high_conf_samples[lang]) < 5:
                high_conf_samples[lang].append((confidence, line))

            # Track low-confidence lines
            if confidence < LATIN_CONF_THRESHOLD:
                latin_low_conf.append((line, lang, confidence))


# =========================
# OUTPUT SUMMARY
# =========================

print("\n===== TOTAL LINES =====")
print(total_lines)

print("\n===== SCRIPT DISTRIBUTION =====")
for script, count in script_distribution.items():
    print(f"{script}: {count}")

print("\n===== LATIN LANGUAGE DISTRIBUTION =====")
for lang, count in latin_predictions.items():
    print(f"{lang}: {count}")

print("\nLOW CONFIDENCE LATIN LINES:", len(latin_low_conf))


# =========================
# SAMPLE OUTPUT
# =========================

print("\n===== SAMPLE LATIN PREDICTIONS =====")

for lang in sorted(samples.keys()):
    print(f"\n--- {lang} ---")
    for conf, line in samples[lang]:
        print(f"{conf:.3f} | {line[:120]}")


print("\n===== HIGH CONFIDENCE LATIN SAMPLES (>= 0.95) =====")

for lang in sorted(high_conf_samples.keys()):
    print(f"\n+++ {lang} +++")
    for conf, line in high_conf_samples[lang]:
        print(f"{conf:.3f} | {line[:120]}")


# =========================
# SAVE LOW CONF LINES
# =========================

with open("latin_low_confidence.txt", "w", encoding="utf-8") as out:
    for line, lang, conf in latin_low_conf[:500]:
        out.write(f"{lang} | {conf:.3f} | {line}\n")

print("\nSaved low confidence Latin lines to latin_low_confidence.txt")
