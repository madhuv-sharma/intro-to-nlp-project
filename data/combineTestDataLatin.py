from pathlib import Path

train_dir = "./train v1"
test_dir = "../kaggle-data"
output_dir = "./train latin"

latin_langs = {"en", "fr", "de", "it"}


def combine_data():
    for file in Path(train_dir).glob("*.txt"):
        with open(file, encoding="utf-8") as f:
            file_name = file.name
            if file.stem in latin_langs:
                file_name = "latin.txt"
            with open(Path(output_dir) / file_name, "w", encoding="utf-8") as h:
                for line in f:
                    h.write(line)
                try:
                    with open(Path(test_dir) / file.name, "r", encoding="utf-8") as g:
                        for line in g:
                            h.write(line)
                except FileNotFoundError:
                    print(f"File {file.name} not found in {test_dir}, skipping.")

    print("Finished combining train and test data for latin languages.")
    print("Now appending unknown.txt to latin.txt if it exists.")
    try:
        with open(Path(test_dir) / "unknown.txt", "r", encoding="utf-8") as g:
            with open(Path(output_dir) / "latin.txt", "a", encoding="utf-8") as h:
                for line in g:
                    h.write(line)
    except FileNotFoundError:
        print(f"File unknown.txt not found in {test_dir}, skipping.")


if __name__ == "__main__":
    combine_data()
