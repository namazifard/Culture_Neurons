#!/usr/bin/env python
import numpy as np
import os

mdl = "llama-2-7b"
MODEL = 'meta-llama/Llama-2-7b-hf'
model = 'meta-llama/Llama-2-7b-hf'

# 5 evaluation languages (columns)
LANGS = ['en', 'de', 'da', 'zh', 'ru', 'fa']
# 5 cultures (for culture / pure / compound where rows are cultures)
CULTS = ["english", "german", "danish", "chinese", "russian", "persian"]

# map each method to its row‐labels and filename pattern
METHODS = {
    "language": {
        "rows": LANGS,
        "pattern": "results/llama-2-7b/ppl_mask_{row}.txt",
    },
    "culture": {
        "rows": CULTS,
        "pattern": "results/{llama-2-7b/ppl_mask_{row}.txt",
    },
    "pure_culture": {
        "rows": CULTS,
        "pattern": "results/llama-2-7b/ppl_mask_{row}_pure.txt",
    },
    "compound": {
        "rows": LANGS,
        "pattern": "results/llama-2-7b/ppl_mask_{row}_compound.txt",
    },
}

# load baseline
base = np.loadtxt("results/llama-2-7b/ppl_mask_baseline.txt")

# make sure output dir exists
os.makedirs("results", exist_ok=True)

for method, info in METHODS.items():
    rows = info["rows"]
    pat  = info["pattern"]
    mat  = []
    for r in rows:
        fn = pat.format(row=r)
        vec = np.loadtxt(fn)
        mat.append(vec - base)
    M = np.vstack(mat)  # shape (5,5)
    out = f"results/llama-2-7b/matrix_{method}.txt"
    np.savetxt(out, M, fmt="%.4f", delimiter="\t")
    print(f"Saved {out}")