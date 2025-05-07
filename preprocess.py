# -*- coding: utf-8 -*-
"""
CSV ▸ JSONL pre‑processor for the food‑recipe dataset
====================================================
* Handles blank first header.
* Appends `.jpg` if missing.
* Falls back to `Cleaned_Ingredients`.
* Outputs **absolute** image paths.
Run:
```bash
python preprocess_recipe_csv.py --verbose 5
```
Override headers/paths with flags.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

# ── regex / unit tweaks ────────────────────────────────────────────────────
UNIT_REWRITE = {
    r"tsp[.]?": "tsp",
    r"tbsp[.]?": "tbsp",
    r"oz[.]?": "oz",
    r"fl[.]? oz": "fl oz",
    r"g[.]?": "g",
    r"kg[.]?": "kg",
    r"ml[.]?": "ml",
    r"l[.]?": "l",
}
UNIT_PATTS = [(re.compile(p, re.I), r) for p, r in UNIT_REWRITE.items()]
STEP_SPLIT_RE = re.compile(r"\r?\n|\u2022|;|\t|\d+\.\s")
ING_SPLIT_RE = re.compile(r"[,|]+")
IMG_EXT_RE  = re.compile(r"\.jpe?g$", re.I)

# ── helpers ────────────────────────────────────────────────────────────────

def normalise_units(txt: str) -> str:
    for patt, repl in UNIT_PATTS:
        txt = patt.sub(repl, txt)
    return txt.strip()


def split_ingredients(raw: str) -> List[str]:
    return [normalise_units(p) for p in ING_SPLIT_RE.split(raw) if p.strip()]


def split_steps(raw: str) -> List[str]:
    raw = re.sub(r"^\s*\d+\.\s*", "", raw, flags=re.MULTILINE)
    return [s.strip() for s in STEP_SPLIT_RE.split(raw) if s.strip()]


def to_xml(title: str, ings: List[str], steps: List[str]) -> str:
    parts = [
        "<recipe>", f"  <title>{title}</title>", "  <ingredients>",
        *[f"    <ingredient>{i}</ingredient>" for i in ings],
        "  </ingredients>", "  <instructions>",
        *[f"    <step>{idx}. {st}</step>" for idx, st in enumerate(steps, 1)],
        "  </instructions>", "</recipe>",
    ]
    return "\n".join(parts)

# ── per‑row conversion ─────────────────────────────────────────────────────

def clean(row: Dict[str, str]) -> Dict[str, str]:
    """Lower‑case keys for fuzzy match."""
    return {k.lower().strip(): v for k, v in row.items()}


def row_to_example(row: Dict[str, str], img_dir: Path, cols: Tuple[str, str, str, str]):
    img_col, title_col, ing_col, dir_col = [c.lower() for c in cols]
    row_lc = clean(row)

    img_name = row_lc.get(img_col, "").strip().strip('"\'')
    if not img_name:
        return None, "missing image field"
    if not IMG_EXT_RE.search(img_name):
        img_name += ".jpg"
    img_path = (img_dir / img_name).resolve()
    if not img_path.exists():
        return None, "image not found on disk"

    raw_ing = row_lc.get(ing_col) or row_lc.get("cleaned_ingredients", "")
    raw_dir = row_lc.get(dir_col, "")
    if not raw_ing.strip():
        return None, "no ingredients"
    if not raw_dir.strip():
        return None, "no directions"

    ings  = split_ingredients(raw_ing)
    steps = split_steps(raw_dir)
    if not ings or not steps:
        return None, "failed split"

    title = (row_lc.get(title_col) or "Unknown").strip()
    return {"image_path": str(img_path), "recipe_xml": to_xml(title, ings, steps)}, ""

# ── core routine ───────────────────────────────────────────────────────────

def preprocess(csv_p: Path, img_dir: Path, out_p: Path, verbose: int, cols):
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with csv_p.open(newline="", encoding="utf-8") as fin:
        first = fin.readline(); fin.seek(0)
        hdr   = next(csv.reader([first]))
        if not hdr[0]:
            hdr[0] = "id"
            rdr = csv.DictReader(fin, fieldnames=hdr); next(rdr)
        else:
            rdr = csv.DictReader(fin)

        total = saved = shown = 0
        with out_p.open("w", encoding="utf-8") as fout:
            for row in rdr:
                total += 1
                ex, reason = row_to_example(row, img_dir, cols)
                if ex is None:
                    if verbose and shown < verbose:
                        print(f"Skip #{total}: {reason}")
                        shown += 1
                    continue
                json.dump(ex, fout, ensure_ascii=False); fout.write("\n"); saved += 1
    print(f"Saved {saved}/{total} → {out_p}")

# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description="Food CSV → JSONL (abs paths)")
    ap.add_argument("--csv",    type=Path, default=root / "Food Ingredients and Recipe Dataset with Image Name Mapping.csv")
    ap.add_argument("--images", type=Path, default=root / "Food Images/Food Images")
    ap.add_argument("--out",    type=Path, default=root / "data/recipe_pairs.jsonl")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--image_col", default="Image_Name")
    ap.add_argument("--title_col", default="Title")
    ap.add_argument("--ing_col",   default="Ingredients")
    ap.add_argument("--dir_col",   default="Instructions")
    args = ap.parse_args()

    preprocess(args.csv, args.images, args.out, args.verbose,
               (args.image_col, args.title_col, args.ing_col, args.dir_col))
