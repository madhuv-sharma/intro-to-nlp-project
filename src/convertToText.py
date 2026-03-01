import json

with open("space_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

with open("en.txt", "w", encoding="utf-8") as output_file:
    for item in data:
        line = item["text"]["en"].strip()
        if not line:
            continue
        output_file.write(item["text"]["en"] + "\n")
