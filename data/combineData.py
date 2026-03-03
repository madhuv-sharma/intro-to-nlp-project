from pathlib import Path

train_v0_dir = "train v0"
open_dev_dir = "open-dev"
output_dir = "train v1"


def combine_data():
    for file in Path(train_v0_dir).glob("*.txt"):
        with open(file, encoding="utf-8") as f:
            with open(Path(open_dev_dir) / file.name, "r", encoding="utf-8") as g:
                with open(Path(output_dir) / file.name, "w", encoding="utf-8") as h:
                    for line in f:
                        h.write(line)
                    for line in g:
                        h.write(line)


if __name__ == "__main__":
    combine_data()
