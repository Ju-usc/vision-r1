# -*- coding: utf-8 -*-
"""
Create tiny dev‑stage splits: 80 train / 10 dev / 10 test
========================================================
Run *after* you have the full‑size:
  • `data/recipe_pairs.jsonl`
  • `data/recipe_pairs_raw_vec.jsonl`

It writes three paired JSONL files per split so the indices stay aligned:
```
data/dev_stage/
    train_pairs.jsonl
    dev_pairs.jsonl
    test_pairs.jsonl
    train_pairs_vec.jsonl
    dev_pairs_vec.jsonl
    test_pairs_vec.jsonl
```
The “vec” files retain only the rows matching the small splits.
"""
from __future__ import annotations

import argparse, json, itertools
from pathlib import Path
import random


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def dump_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")


def main(rec_path: Path, vec_path: Path, out_dir: Path, seed: int = 42):
    recs = load_jsonl(rec_path)
    vecs = load_jsonl(vec_path)
    assert len(recs) == len(vecs), "recipe and vector files must align 1‑to‑1"

    random.seed(seed)
    idxs = list(range(len(recs)))
    random.shuffle(idxs)

    split_sizes = {"train": 80, "dev": 10, "test": 10}
    cursor = 0
    for split, n in split_sizes.items():
        take = idxs[cursor : cursor + n]
        cursor += n
        dump_jsonl([recs[i] for i in take], out_dir / f"{split}_pairs.jsonl")
        dump_jsonl([vecs[i]  for i in take], out_dir / f"{split}_pairs_vec.jsonl")
        print(f"{split}: {n} examples written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", default="data/recipe_pairs.jsonl", type=Path)
    ap.add_argument("--vectors", default="data/recipe_pairs_raw_vec.jsonl", type=Path)
    ap.add_argument("--out_dir", default="data/dev_stage", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    main(args.recipes, args.vectors, args.out_dir, args.seed)
