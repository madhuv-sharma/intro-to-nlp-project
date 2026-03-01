import csv

with open("./metaData.csv", "r") as f:
    reader = csv.reader(f)
    reader.__next__()
    langs = set()
    for row in reader:
        langs.add(row[1])
print(len(langs))
print(langs)
