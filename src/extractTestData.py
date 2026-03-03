import csv
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

csv_path = Path("../kaggle-data/test.csv")
output_dir = Path("../kaggle-data/")
output_dir.mkdir(parents=True, exist_ok=True)

SCRIPT_LANG_MAP = {
    "HIRAGANA": "ja",
    "KATAKANA": "ja",
    "CJK": "zh",
    "HANGUL": "ko",
    "ARABIC": "ar",
    "CYRILLIC": "ru",
    "DEVANAGARI": "hi",
}

AMBIGUOUS_SCRIPTS = {"LATIN", "GREEK"}


def detect_language_from_unicode(text, debug):
    script_counts = Counter()
    total_letters = 0

    for i, ch in enumerate(text):
        if ch.isalpha():
            total_letters += 1
            try:
                name = unicodedata.name(ch)
                if debug and i < 5:
                    print(name)
                for script in SCRIPT_LANG_MAP:
                    if script in name:
                        script_counts[script] += 1
                        break
                for amb in AMBIGUOUS_SCRIPTS:
                    if amb in name:
                        script_counts[amb] += 1
            except ValueError:
                continue

    if total_letters == 0:
        return None

    if not script_counts:
        return None

    script, count = script_counts.most_common(1)[0]

    if debug:
        print(
            f"Line: {text[:30]}... Detected script: {script} with count {count}/{total_letters}"
        )

    if script in AMBIGUOUS_SCRIPTS:
        return None

    if count / total_letters >= 0.4:
        return SCRIPT_LANG_MAP.get(script)

    return None


# Group lines by detected language
grouped = defaultdict(list)

with csv_path.open("r", encoding="utf-8") as f:
    reader = csv.reader(f)
    reader.__next__()  # Skip header
    for i, row in enumerate(reader):
        if not row:
            continue

        line = row[1].strip()
        if not line:
            continue

        lang = detect_language_from_unicode(line, i % 2 == 0 and i < 20)
        if lang:
            grouped[lang].append(line)
        else:
            grouped["unknown"].append(line)

print("Detected languages and counts:")
for lang, lines in grouped.items():
    print(f"{lang}: {len(lines)} lines")

for lang, lines in grouped.items():
    output_path = output_dir / f"{lang}.txt"

    with output_path.open("w", encoding="utf-8") as out:
        for line in lines:
            out.write(line + "\n")

print("Done. Wrote detected lines to language files.")
