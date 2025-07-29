#!/usr/bin/env python
"""
Compute generic culture neurons  G = ⋂_i  P_i
where P_i are the *_pure.mask.pth files written by make_compound_and_pure.py
"""

# select the model

# mdl = TAG = "Llama-2-7b"
# MODEL = model = "meta-llama/Llama-2-7b-hf"

# mdl = TAG = "Llama-3.1-8b"
# MODEL = model = "meta-llama/Llama-3.1-8B"

mdl = TAG = "Qwen2.5-7b"
MODEL = model = "Qwen/Qwen2.5-7B"

# mdl = TAG = "Gemma-3-12b"
# MODEL = model = "google/gemma-3-12b-pt"

import torch, functools, operator, glob

pure_paths = sorted(glob.glob(f"activation_mask/{mdl}/*_pure.mask.pth"))
if not pure_paths:
    raise RuntimeError("No *_pure.mask.pth files found. Run make_compound_and_pure.py first.")

pure_masks = [torch.load(p) for p in pure_paths]
generic = [functools.reduce(operator.__and__, layer_tuple)
           for layer_tuple in zip(*pure_masks)]

out = f"activation_mask/{mdl}/generic_culture.mask.pth"
torch.save(generic, out)
print("saved", out, "total neurons:", sum(l.sum().item() for l in generic))
