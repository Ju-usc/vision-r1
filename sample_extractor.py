# -*- coding: utf-8 -*-
"""
Quick helper to fetch a unified sample from the two JSONL files
==============================================================
* **recipes_file**  – output of `preprocess_recipe_csv.py` (`recipe_pairs.jsonl`)
* **vectors_file**  – output of `vectorize_recipes_raw.py` (`recipe_pairs_raw_vec.jsonl`)

The function `get_sample(idx)` returns a dictionary:
```
{
  'title':                  str,
  'ingredients':            List[str],
  'instructions':           List[str],
  'ingredients_embeddings': List[List[float]],  # one per ingredient
  'instructions_embeddings':List[List[float]],  # one per step
  'image':                  PIL.Image.Image
}
```
Usage
-----
```python
from sample_extractor import get_sample
s = get_sample(42, recipes_path='data/recipe_pairs.jsonl',
                     vectors_path='data/recipe_pairs_raw_vec.jsonl')
print(s['title'])
print(len(s['ingredients']), 'ingredients')
```
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import xml.etree.ElementTree as ET

ING_RE  = re.compile(r"<ingredient>(.*?)</ingredient>", re.S)
STEP_RE = re.compile(r"<step>\d+\. (.*?)</step>", re.S)

def _parse_xml(xml: str) -> tuple[List[str], List[str]]:
    ings  = [m.strip() for m in ING_RE.findall(xml)]
    steps = [m.strip() for m in STEP_RE.findall(xml)]
    if ings and steps:
        return ings, steps
    try:  # fall back robust XML parser
        root = ET.fromstring(xml)
        ings  = [n.text.strip() for n in root.find('ingredients').findall('ingredient')]
        steps = [n.text.strip() for n in root.find('instructions').findall('step')]
        return ings, steps
    except Exception:
        return [], []


def _load_nth_line(path: Path, idx: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"Index {idx} out of range for {path}")


def get_sample(idx: int,
               recipes_path: str | Path = "data/recipe_pairs.jsonl",
               vectors_path: str | Path = "data/recipe_pairs_raw_vec.jsonl") -> Dict[str, Any]:
    """Return unified sample at position *idx* (0‑based)."""
    recipes_path = Path(recipes_path)
    vectors_path = Path(vectors_path)

    rec  = _load_nth_line(recipes_path, idx)
    vecs = _load_nth_line(vectors_path,   idx)

    # parse XML
    ings, steps = _parse_xml(rec["recipe_xml"])

    # load image
    img = Image.open(rec["image_path"]).convert("RGB")

    return {
        "title": rec.get("title") or _parse_title(rec["recipe_xml"]),
        "ingredients": ings,
        "instructions": steps,
        "ingredients_embeddings": vecs.get("ing_vecs", []),
        "instructions_embeddings": vecs.get("step_vecs", []),
        "image": img,
    }


def _parse_title(xml: str) -> str:
    m = re.search(r"<title>(.*?)</title>", xml, re.S)
    if m:
        return m.group(1).strip()
    try:
        root = ET.fromstring(xml)
        return root.find('title').text.strip()
    except Exception:
        return "Unknown"

# demo block ---------------------------------------------------------------
if __name__ == "__main__":
    s = get_sample(0)
    print("Title:", s['title'])
    print("#ingredients:", len(s['ingredients']))
    print("#steps:", len(s['instructions']))
    print("First embedding dim:", len(s['ingredients_embeddings'][0]) if s['ingredients_embeddings'] else 'N/A')
    s['image'].show()
