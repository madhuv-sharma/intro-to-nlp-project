import os
import re
import json

BASE_DIR = "../missions"
OUTPUT_FILE = "space_data.json"

LANGS = ["zh", "de", "ko", "ru", "ja", "hi", "ar", "fr", "en", "it"]

timestamp_pattern = re.compile(r"\[(-?\d{2}:\d{2}:\d{2}:\d{2})\]")
dialogue_pattern = re.compile(r"^([A-Za-z0-9][A-Za-z0-9 ()\-']{0,40}):\s+(.*)")

dataset = []
files_processed = 0

for mission in os.listdir(BASE_DIR):
    print(f"Processing mission: {mission}...")

    file_names = ["ATG", "CM", "PAO", "TEC", "en"]

    for file_name in file_names:
        file_path = os.path.join(BASE_DIR, mission, "transcripts", file_name)
        if not os.path.exists(file_path):
            # print(f"File not {file_name} found for mission {mission}, skipping.")
            continue

        files_processed += 1
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            current_timestamp = None
            current_speaker = None
            current_text_lines = []

            for line in f:
                line = line.rstrip()

                # Timestamp line
                ts_match = timestamp_pattern.match(line)
                if ts_match:
                    current_timestamp = ts_match.group(1)
                    continue

                # Blank line = finalize current dialogue
                if not line.strip():
                    if current_timestamp and current_speaker and current_text_lines:
                        full_text = " ".join(current_text_lines).strip()

                        # [glossary:term] -> term
                        full_text = re.sub(r"\[glossary:([^\]]+)\]", r"\1", full_text)
                        # [time:00:00:00|description] -> description
                        full_text = re.sub(
                            r"\[time:[^|]+\|([^\]]+)\]", r"\1", full_text
                        )
                        # [time:02:42:18] -> 02:42:18
                        full_text = re.sub(
                            r"\[time:(\d{2}:\d{2}:\d{2})\]", r"\1", full_text
                        )
                        # <del>Censored</del> -> (remove)
                        full_text = re.sub(r"<del>Censored</del>", "", full_text)

                        full_text = re.sub(r"_page\s*:.*", "", full_text)
                        full_text = re.sub(r"_comment\s*:.*", "", full_text)
                        full_text = re.sub(r"_note\s*:.*", "", full_text)

                        dataset.append(
                            {
                                "mission": mission,
                                "timestamp": current_timestamp,
                                "speaker": current_speaker,
                                "text": {"en": full_text},
                            }
                        )

                    current_speaker = None
                    current_text_lines = []
                    continue

                if line[:1].isspace():  # continuation line
                    if current_speaker:
                        current_text_lines.append(line.strip())
                    continue

                # First dialogue line (has speaker)
                dlg_match = dialogue_pattern.match(line)
                if dlg_match:
                    current_speaker = dlg_match.group(1)
                    first_text = dlg_match.group(2)
                    current_text_lines = [first_text]
                else:
                    # Continuation line
                    if current_speaker:
                        current_text_lines.append(line.strip())

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    json.dump(dataset, out, indent=2, ensure_ascii=False)
    print("Read and processed {} files.".format(files_processed))
