#!/usr/bin/env python
import torch
import pandas as pd
import os

# select the model

# mdl = TAG = "Llama-2-7b"
# MODEL = model = "meta-llama/Llama-2-7b-hf"

# mdl = TAG = "Llama-3.1-8b"
# MODEL = model = "meta-llama/Llama-3.1-8B"

mdl = TAG = "Qwen2.5-7b"
MODEL = model = "Qwen/Qwen2.5-7B"

# mdl = TAG = "Gemma-3-12b"
# MODEL = model = "google/gemma-3-12b-pt"

# Entities
langs = ['en', 'de', 'da', 'zh', 'ru', 'fa']
cults = ["english", "german", "danish", "chinese", "russian", "persian"]

# Prepare storage
records = []

def neuron_count(layer):
    if isinstance(layer, torch.Tensor):
        if layer.dtype == torch.bool:
            return layer.sum().item()
        else:
            return layer.numel()
    else:
        return len(layer)

# 1) Language‐specific & Compound (lang∧cult)
for lang in langs:
    lang_mask  = torch.load(f"activation_mask/{mdl}/{lang}.mask.pth")
    comp_mask  = torch.load(f"activation_mask/{mdl}/{lang}_compound.mask.pth")
    records.append({
        "entity": lang,
        "language_specific": sum(neuron_count(l) for l in lang_mask),
        "compound":          sum(neuron_count(l) for l in comp_mask),
        "culture_specific":  None,
        "pure_culture":      None
    })

# 2) Culture‐specific & Pure‐culture
for cult in cults:
    cult_mask = torch.load(f"activation_mask/{mdl}/{cult}.mask.pth")
    pure_mask = torch.load(f"activation_mask/{mdl}/{cult}_pure.mask.pth")
    records.append({
        "entity":           cult,
        "language_specific": None,
        "compound":          None,
        "culture_specific":  sum(neuron_count(l) for l in cult_mask),
        "pure_culture":      sum(neuron_count(l) for l in pure_mask)
    })

# Build DataFrame
df = pd.DataFrame(records).set_index("entity")
df = df[[
    "language_specific",
    "compound",
    "culture_specific",
    "pure_culture"
]]

# Print to console
print("\nNeuron counts by category:\n")
print(df.to_markdown())

# Save CSV
os.makedirs("results", exist_ok=True)
csv_path = f"results/{mdl}/neuron_counts.csv"
df.to_csv(csv_path)
print(f"\nSaved counts to {csv_path}")