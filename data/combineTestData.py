from pathlib import Path

train_dir = "./train v1"
test_dir = "../kaggle-data"
output_dir = "./train v2"


def combine_data():
    for file in Path(train_dir).glob("*.txt"):
        with open(file, encoding="utf-8") as f:
            with open(Path(output_dir) / file.name, "w", encoding="utf-8") as h:
                for line in f:
                    h.write(line)
                try:
                    with open(Path(test_dir) / file.name, "r", encoding="utf-8") as g:
                        for line in g:
                            h.write(line)
                except FileNotFoundError:
                    print(f"File {file.name} not found in {test_dir}, skipping.")


if __name__ == "__main__":
    combine_data()
