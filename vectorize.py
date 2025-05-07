# -*- coding: utf-8 -*-
"""
Vectorize recipe JSONL → JSONL with sentence‑transformer embeddings
==================================================================
Takes the output of `preprocess_recipe_csv.py` (each line has
`image_path`, `recipe_xml`) and adds two **fixed‑size vectors** per recipe:
* `ing_emb`   – mean pooling of each `<ingredient>` sentence embedding
* `steps_emb` – mean pooling of each `<step>` sentence embedding

Usage
-----
```bash
python vectorize_recipes.py \
  --in_jsonl data/recipe_pairs.jsonl \
  --out_jsonl data/recipe_pairs_vec.jsonl \
  --model all-MiniLM-L6-v2 \
  --batch_size 256
```
Written embeddings are plain lists of floats so they stay JSON serialisable.
You can load them later with NumPy:
```python
import json, numpy as np
vec = np.array(json.loads(line)['ing_emb'])
```
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import xml.etree.ElementTree as ET
from tqdm import tqdm

ING_RE = re.compile(r"<ingredient>(.*?)</ingredient>", re.S)
STEP_RE = re.compile(r"<step>\d+\. (.*?)</step>", re.S)


def parse_xml(xml: str) -> tuple[list[str], list[str]]:
    """Return (ingredients, steps) lists from XML string."""
    try:
        # fast regex first; fall back to ET only if needed
        ings = [m.strip() for m in ING_RE.findall(xml)]
        steps = [m.strip() for m in STEP_RE.findall(xml)]
        if ings and steps:
            return ings, steps
        root = ET.fromstring(xml)
        ings = [i.text.strip() for i in root.find('ingredients').findall('ingredient')]
        steps = [s.text.strip() for s in root.find('instructions').findall('step')]
        return ings, steps
    except Exception:
        return [], []


def mean_pool(embs: List[np.ndarray]) -> List[float]:
    if not embs:
        return []
    return np.mean(np.stack(embs), axis=0).tolist()


def vectorize(in_path: Path, out_path: Path, model_name: str, batch_size: int = 256):
    model = SentenceTransformer(model_name)

    with in_path.open(encoding="utf-8") as fin:
        entries = [json.loads(l) for l in fin]

    # Collect all sentences for efficient batching
    sentences: List[str] = []
    meta: List[tuple[int, str]] = []  # (entry_idx, 'ing'|'step')
    for idx, entry in enumerate(entries):
        ings, steps = parse_xml(entry['recipe_xml'])
        for ing in ings:
            sentences.append(ing)
            meta.append((idx, 'ing'))
        for st in steps:
            sentences.append(st)
            meta.append((idx, 'step'))

    # Prepare holders
    ing_vecs: Dict[int, List[np.ndarray]] = {}
    step_vecs: Dict[int, List[np.ndarray]] = {}

    # Batch encode
    for start in tqdm(range(0, len(sentences), batch_size), desc="Embedding"):
        batch_sents = sentences[start:start + batch_size]
        batch_embs = model.encode(batch_sents, show_progress_bar=False, convert_to_numpy=True)
        for emb, (idx, kind) in zip(batch_embs, meta[start:start + batch_size]):
            (ing_vecs if kind == 'ing' else step_vecs).setdefault(idx, []).append(emb)

    # Add mean‑pooled vectors to entries
    for idx, entry in enumerate(entries):
        entry['ing_emb'] = mean_pool(ing_vecs.get(idx, []))
        entry['steps_emb'] = mean_pool(step_vecs.get(idx, []))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as fout:
        for entry in entries:
            json.dump(entry, fout, ensure_ascii=False)
            fout.write('\n')
    print(f"Wrote {len(entries)} entries → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add sentence‑BERT vectors to recipe JSONL")
    parser.add_argument('--in_jsonl',  type=Path, default=Path('data/recipe_pairs.jsonl'))
    parser.add_argument('--out_jsonl', type=Path, default=Path('data/recipe_pairs_vec.jsonl'))
    parser.add_argument('--model',     type=str,   default='all-MiniLM-L6-v2')
    parser.add_argument('--batch_size', type=int,  default=256)
    args = parser.parse_args()

    vectorize(args.in_jsonl, args.out_jsonl, args.model, args.batch_size)
