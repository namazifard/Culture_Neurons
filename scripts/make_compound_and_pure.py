#!/usr/bin/env python
"""
Builds
  • {lang}_compound.mask.pth   =  L_lang  ∧  C_culture
  • {cult}_pure.mask.pth       =  C_culture  \  ⋃_{lang∈culture} L_lang
Assumes:
  * language masks  activation_mask/{en,de,da,ru,fa}.mask.pth
  * culture  masks  activation_mask/{western,nordic,slavic,iranian,eastasia}.mask.pth
  * mapping Lang → Culture in LANG2CULT below
"""

import torch, os

# select the model

# mdl = TAG = "Llama-2-7b"
# MODEL = model = "meta-llama/Llama-2-7b-hf"

# mdl = TAG = "Llama-3.1-8b"
# MODEL = model = "meta-llama/Llama-3.1-8B"

mdl = TAG = "Qwen2.5-7b"
MODEL = model = "Qwen/Qwen2.5-7B"

# mdl = TAG = "Gemma-3-12b"
# MODEL = model = "google/gemma-3-12b-pt"

LANGS  = ['en', 'de', 'da', 'zh', 'ru', 'fa']
CULTS  = ["english", "german", "danish", "chinese", "russian", "persian"]
LANG2CULT = {         # adapt if your mapping differs
    "en": "english",
    "de": "german",
    "da": "danish",
    "zh": "chinese",
    "ru": "russian",
    "fa": "persian",
}

# ---------- compound masks (L ∧ C) ----------
for lang in LANGS:
    lmask = torch.load(f"activation_mask/{mdl}/{lang}.mask.pth")
    cmask = torch.load(f"activation_mask/{mdl}/{LANG2CULT[lang]}.mask.pth")
    compound = [l & c for l, c in zip(lmask, cmask)]
    out = f"activation_mask/{mdl}/{lang}_compound.mask.pth"
    torch.save(compound, out)
    print("saved", out)

# ---------- pure culture masks (C \ ⋃L) ----------
for cult in CULTS:
    cmask = torch.load(f"activation_mask/{mdl}/{cult}.mask.pth")

    # OR over all language masks that belong to this culture
    union_lang = [torch.zeros_like(layer) for layer in cmask]
    for lang, cl in LANG2CULT.items():
        if cl == cult:
            lmask = torch.load(f"activation_mask/{mdl}/{lang}.mask.pth")
            union_lang = [u | l for u, l in zip(union_lang, lmask)]

    pure = [c & ~u for c, u in zip(cmask, union_lang)]
    out = f"activation_mask/{mdl}/{cult}_pure.mask.pth"
    torch.save(pure, out)
    print("saved", out)
