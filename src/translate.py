import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANGUAGE_NAMES = {
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "ru": "Russian",
    "zh": "Simplified Chinese",
    "ko": "Korean",
    "ja": "Japanese",
    "hi": "Hindi",
    "ar": "Arabic",
}

TARGET_LANGS = ["zh", "de", "ko", "ru", "ja", "hi", "ar", "fr", "it"]

BATCH_SIZE = 24 if DEVICE == "cuda" else 2
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 192


def _build_prompt(tokenizer, line, language_name):
    system_prompt = (
        "You are a translation engine. Translate from English to the requested language. "
        "Return only the translation."
    )
    user_prompt = (
        f"Translate this English sentence to {language_name}.\n" f"English: {line}"
    )

    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return f"{system_prompt}\n\n{user_prompt}\nTranslation:"


def _get_dtype():
    if DEVICE != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=_get_dtype(),
        trust_remote_code=True,
    ).to(DEVICE)
    model.eval()

    print("Loading English lines...")
    with open("en.txt", "r", encoding="utf-8") as f:
        english_lines = [line.rstrip("\n") for line in f]

    print(f"Total lines: {len(english_lines)}")

    for lang in TARGET_LANGS:
        language_name = LANGUAGE_NAMES[lang]
        print(f"\nTranslating to {lang} ({language_name})...")

        output_lines = []

        with torch.inference_mode():
            for i in range(0, len(english_lines), BATCH_SIZE):
                batch = english_lines[i : i + BATCH_SIZE]
                prompts = [
                    _build_prompt(tokenizer, line, language_name) for line in batch
                ]

                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS,
                ).to(DEVICE)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                generated = outputs[:, inputs["input_ids"].shape[1] :]
                translations = tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
                output_lines.extend([t.strip() for t in translations])

                if (i // BATCH_SIZE) % 20 == 0:
                    print(f"{lang}: {i}/{len(english_lines)}")

        with open(f"{lang}.txt", "w", encoding="utf-8") as f:
            for line in output_lines:
                f.write(line + "\n")

        print(f"{lang}.txt written.")

    print("All languages completed.")


if __name__ == "__main__":
    main()
