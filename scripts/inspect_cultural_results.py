import numpy as np
import glob, os

# 1. baseline
base = np.loadtxt("results/ppl_baseline.txt")

def load_and_delta(pattern, label):
    for fn in sorted(glob.glob(pattern)):
        vec = np.loadtxt(fn)
        name = os.path.basename(fn).split("_")[2]
        delta = (vec - base).round(3)
        print(f"{label:<8s} {name:<10s} Δ:", delta)

print("Baseline:", base.round(3))
load_and_delta("results/ppl_mask_*[a-z].txt",      "Lang")    # language-only
load_and_delta("results/ppl_mask_*[a-z][a-z].txt", "Cult")    # culture-only
load_and_delta("results/ppl_mask_*_pure.txt",      "Pure")    # pure culture
load_and_delta("results/ppl_mask_*_compound.txt",  "Compound")# lang∧cult
