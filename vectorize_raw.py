# -*- coding: utf-8 -*-
"""
Add **per‑ingredient** and **per‑step** sentence‑BERT vectors to each recipe
==========================================================================
Unlike `vectorize_recipes.py` (which mean‑pools to one vector per section),
this script keeps the raw list of embeddings so you can reason about
individual items later.

Input JSONL (one per line) – produced by `preprocess_recipe_csv.py`:
```
{"image_path": "/abs/.../123.jpg", "recipe_xml": "<recipe>…</recipe>"}
```

Output JSONL – same fields plus:
* `ing_vecs`   – list[list[float]]  (len == #ingredients)
* `step_vecs`  – list[list[float]]  (len == #steps)

Run
----
```bash
python vectorize_recipes_raw.py \
  --in_jsonl  data/recipe_pairs.jsonl \
  --out_jsonl data/recipe_pairs_raw_vec.jsonl \
  --model all-MiniLM-L6-v2 \
  --batch_size 256
```
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import xml.etree.ElementTree as ET

ING_RE  = re.compile(r"<ingredient>(.*?)</ingredient>", re.S)
STEP_RE = re.compile(r"<step>\d+\. (.*?)</step>", re.S)


def extract(xml: str) -> Tuple[List[str], List[str]]:
    """Return (ingredients, steps) from the XML string."""
    ings  = [m.strip() for m in ING_RE.findall(xml)]
    steps = [m.strip() for m in STEP_RE.findall(xml)]
    if ings and steps:
        return ings, steps
    # fallback XML parser
    try:
        root = ET.fromstring(xml)
        ings = [n.text.strip() for n in root.find('ingredients').findall('ingredient')]
        steps = [n.text.strip() for n in root.find('instructions').findall('step')]
        return ings, steps
    except Exception:
        return [], []


def vectorize(in_path: Path, out_path: Path, model_name: str, batch_size: int):
    model = SentenceTransformer(model_name)

    # --- read all recipes
    with in_path.open(encoding="utf-8") as f:
        recipes = [json.loads(line) for line in f]

    # --- collect every sentence for batch encoding
    all_sentences: List[str] = []
    index_meta: List[Tuple[int, str]] = []  # (recipe_idx, 'ing'|'step')

    for idx, rec in enumerate(recipes):
        ings, steps = extract(rec["recipe_xml"])
        rec["_ings_src"]  = ings  # store temporarily for order
        rec["_steps_src"] = steps
        for ing in ings:
            all_sentences.append(ing)
            index_meta.append((idx, "ing"))
        for st in steps:
            all_sentences.append(st)
            index_meta.append((idx, "step"))

    # --- initialise holders
    for rec in recipes:
        rec["ing_vecs"]  = []
        rec["step_vecs"] = []

    # --- encode in batches
    for start in tqdm(range(0, len(all_sentences), batch_size), desc="embedding"):
        batch = all_sentences[start:start + batch_size]
        embs  = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        for emb, (idx, kind) in zip(embs, index_meta[start:start + batch_size]):
            target = "ing_vecs" if kind == "ing" else "step_vecs"
            recipes[idx][target].append(emb.tolist())

    # sanity: drop temp fields
    for rec in recipes:
        rec.pop("_ings_src", None)
        rec.pop("_steps_src", None)

    # write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in recipes:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(recipes)} recipes → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add raw per-ingredient / per-step vectors to recipes")
    ap.add_argument("--in_jsonl",  type=Path, default=Path("data/recipe_pairs.jsonl"))
    ap.add_argument("--out_jsonl", type=Path, default=Path("data/recipe_pairs_raw_vec.jsonl"))
    ap.add_argument("--model",     type=str,  default="all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    vectorize(args.in_jsonl, args.out_jsonl, args.model, args.batch_size)
