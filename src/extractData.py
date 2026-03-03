from pathlib import Path
from collections import defaultdict

input_file = Path("../data/open-dev/input.txt")
answer_file = Path("../data/open-dev/answer.txt")
lang_file = Path("../data/open-dev/lang.txt")

# Read files
with input_file.open("r", encoding="utf-8") as f:
    input_lines = [line.rstrip("\n") for line in f]

with answer_file.open("r", encoding="utf-8") as f:
    answer_lines = [line.strip() for line in f]

with lang_file.open("r", encoding="utf-8") as f:
    lang_lines = [line.strip() for line in f]

# Safety check
if len(input_lines) != len(lang_lines):
    raise ValueError("input.txt and lang.txt do not have the same number of lines!")

# Group lines by language
grouped = defaultdict(list)

for sentence, lang, answer in zip(input_lines, lang_lines, answer_lines):
    grouped[lang].append(sentence + answer)

# Write to corresponding files
for lang, sentences in grouped.items():
    output_path = Path(f"../data/open-dev/{lang}.txt")

    with output_path.open("w", encoding="utf-8") as out:
        for s in sentences:
            out.write(s + "\n")

print("Done. Files created per language.")
