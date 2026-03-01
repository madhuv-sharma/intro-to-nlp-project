import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANG_MAP = {
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
    "ja": "jpn_Jpan",
    "hi": "hin_Deva",
    "ar": "arb_Arab",
}

# TARGET_LANGS = ["zh", "de", "ko", "ru", "ja", "hi", "ar", "fr", "it"]
TARGET_LANGS = ["hi"]

BATCH_SIZE = 128 if DEVICE == "cuda" else 32


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    model.eval()
    tokenizer.src_lang = "eng_Latn"

    print("Loading English lines...")
    with open("en.txt", "r", encoding="utf-8") as f:
        english_lines = [line.rstrip("\n") for line in f]

    print(f"Total lines: {len(english_lines)}")

    for lang in TARGET_LANGS:

        print(f"\nTranslating to {lang}...")
        target_code = tokenizer.convert_tokens_to_ids(LANG_MAP[lang])

        output_lines = []

        with torch.inference_mode():
            for i in range(0, len(english_lines), BATCH_SIZE):

                batch = english_lines[i : i + BATCH_SIZE]

                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True
                ).to(DEVICE)

                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=target_code,
                    max_new_tokens=384,
                    num_beams=1,
                    do_sample=False,
                )

                translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                output_lines.extend(translations)

                if (i // BATCH_SIZE) % 40 == 0:
                    print(f"{lang}: {i}/{len(english_lines)}")

        # Write language file
        with open(f"{lang}.txt", "w", encoding="utf-8") as f:
            for line in output_lines:
                f.write(line + "\n")

        print(f"{lang}.txt written.")

    print("All languages completed.")


if __name__ == "__main__":
    main()
