#!/usr/bin/env python
from argparse import ArgumentParser
from collections import Counter
import os


parser = ArgumentParser()
parser.add_argument("fpred")
parser.add_argument("fgold")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()


def load_pred(fname, force_limit=None):
    loaded = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            line = line[:-1].lower()
            if force_limit is not None:
                line = line[:force_limit]
            loaded.append(line)
        return loaded


pred = load_pred(args.fpred, force_limit=3)
gold = load_pred(args.fgold)

flang = os.path.join(os.path.dirname(args.fgold), "lang.txt")
lang = load_pred(flang)

if len(pred) < len(gold):
    pred.extend([""] * (len(gold) - len(pred)))

correct = Counter()
total = Counter()
for i, (p, g, l) in enumerate(zip(pred, gold, lang)):
    right = g in p
    correct[l] += right
    total[l] += 1
    if args.verbose:
        print(
            "Input {}: {}, {} is {} in {}".format(
                i, "right" if right else "wrong", g, "in" if right else "not in", p
            )
        )


for k, v in correct.items():
    print(f"Success rate for {k}: {v}/{total[k]} = {v/total[k]}")

avg = sum(correct.values()) / sum(total.values())
print(f"Average success rate: {avg}")
