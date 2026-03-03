import csv
import unicodedata
from pathlib import Path

test_csv = Path("../kaggle-data/test.csv")
langs = set()

with test_csv.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        context = row["context"]
        
        lang = unicodedata.name(context[0]).split()[0]
        lang_counter = {}
        for char in context:
            try:
                char_lang = unicodedata.name(char).split()[0]
                if char_lang != lang:
                    lang = char_lang
                    break
            except ValueError:
                pass
        langs.add(lang)

print("Languages in test set:", len(langs))
print(langs)
