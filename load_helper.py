# -*- coding: utf-8 -*-
"""
recipe_dataset_utils.py  –  I/O helpers for the tiny 80/10/10 splits
===================================================================
Usage
-----
```python
from recipe_dataset_utils import get_dev_stage_datasets
train_ds, dev_ds, test_ds = get_dev_stage_datasets(
    root="data/dev_stage",                # where subset_recipes.py wrote files
    map_to_prompt_fn=format_data          # your existing formatter
)
```
Functions exported
------------------
* **parse_recipe_xml(xml_str) -> (title, ings, steps)**
* **merge_pair(rec_line, vec_line) -> dict**  – one ready row
* **load_split(split, root) -> Dataset**      – raw rows w/ image + vectors
* **get_dev_stage_datasets(root, map_to_prompt_fn)**  – returns train/dev/test

All PIL images are loaded in RGB.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Callable

from datasets import Dataset, Features, Sequence, Value
from PIL import Image

__all__ = [
    "parse_recipe_xml",
    "merge_pair",
    "load_split",
    "get_dev_stage_datasets",
]

# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------

def parse_recipe_xml(xml_str: str) -> Tuple[str, List[str], List[str]]:
    """Return (title, ingredients list, steps list) from the <recipe> XML."""
    root = ET.fromstring(xml_str)
    title = (root.findtext("title") or "Unknown").strip()
    ings  = [n.text.strip() for n in root.find("ingredients").iterfind("ingredient")]
    steps = [n.text.strip() for n in root.find("instructions").iterfind("step")]
    return title, ings, steps

# ---------------------------------------------------------------------------
# JSONL merging / loading helpers
# ---------------------------------------------------------------------------

def merge_pair(rec_line: str, vec_line: str):
    """Combine one recipe JSONL row with its vector row."""
    rec  = json.loads(rec_line)
    vec  = json.loads(vec_line)

    title, ings, steps = parse_recipe_xml(rec["recipe_xml"])

    return {
        "image"                   : Image.open(rec["image_path"]).convert("RGB"),
        "title"                   : title,
        "ingredients"             : ings,
        "instructions"            : steps,
        "ingredients_embeddings"  : vec["ing_vecs"],
        "instructions_embeddings" : vec["step_vecs"],
    }


def _feature_schema() -> Features:
    """Fixed schema so HF avoids guessing float64 arrays."""
    return Features({
        "image"                   : Value("binary"),
        "title"                   : Value("string"),
        "ingredients"             : Sequence(Value("string")),
        "instructions"            : Sequence(Value("string")),
        "ingredients_embeddings"  : Sequence(Sequence(Value("float32"))),
        "instructions_embeddings" : Sequence(Sequence(Value("float32"))),
    })


def load_split(split: str, root: str | Path = "data/dev_stage") -> Dataset:
    """Load *train*, *dev*, or *test* raw rows into a HF Dataset."""
    root = Path(root)
    rec_path = root / f"{split}_pairs.jsonl"
    vec_path = root / f"{split}_pairs_vec.jsonl"

    with rec_path.open(encoding="utf-8") as fr, vec_path.open(encoding="utf-8") as fv:
        rows = [merge_pair(r, v) for r, v in zip(fr, fv)]

    return Dataset.from_list(rows, features=_feature_schema())


# ---------------------------------------------------------------------------
# Public convenience: tiny 80/10/10 splits ready for training/eval
# ---------------------------------------------------------------------------

def get_dev_stage_datasets(
    root: str | Path = "data/dev_stage",
    map_to_prompt_fn: Callable | None = None,
):
    """Return (train, dev, test) datasets.

    If *map_to_prompt_fn* is provided, it will be applied to every split to
    produce the `{"prompt":…, "answer":…}` structure your trainer needs.
    """
    train_ds = load_split("train", root)
    dev_ds   = load_split("dev",   root)
    test_ds  = load_split("test",  root)

    if map_to_prompt_fn is not None:
        train_ds = train_ds.map(map_to_prompt_fn, remove_columns=train_ds.column_names)
        dev_ds   = dev_ds.map(map_to_prompt_fn,   remove_columns=dev_ds.column_names)
        test_ds  = test_ds.map(map_to_prompt_fn,  remove_columns=test_ds.column_names)

    return train_ds, dev_ds, test_ds
