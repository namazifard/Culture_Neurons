import os
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

# Model lookup by tag
MODELS = {
    "Llama-2-7b":     "meta-llama/Llama-2-7b-hf",
    "Llama-3.1-8b":   "meta-llama/Llama-3.1-8B",
    "Qwen2.5-7b":     "Qwen/Qwen2.5-7B",
    "Gemma-3-12b":    "google/gemma-3-12b-pt",
}

LANGS        = ["en", "de", "da", "zh", "ru", "fa"]
TOK_EVAL     = 100_000     # eval set size
TOK_TRAIN    = 10_000_000  # train set size after eval
TARGET_TOTAL = TOK_EVAL + TOK_TRAIN
RAW_DIR      = "cultural_data/raw"
OUT_DIR      = "cultural_data"

def main(tag):
    # Check model tag validity
    if tag not in MODELS:
        raise ValueError(f"Unknown tag: {tag}. Choose from {list(MODELS.keys())}")
    model_name = MODELS[tag]
    print(f"Using model '{model_name}' for tag '{tag}'")

    # Set up Hugging Face cache
    CACHE = os.path.expanduser("~/hf_cache")
    os.environ["HF_HOME"] = CACHE
    os.environ["TRANSFORMERS_CACHE"] = CACHE

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    os.makedirs(f"{OUT_DIR}/valid",  exist_ok=True)
    os.makedirs(f"{OUT_DIR}/train", exist_ok=True)

    for lang in LANGS:
        # Gather all .txt files for this language
        txt_files = [
            os.path.join(root, fn)
            for root, _, files in os.walk(f"{RAW_DIR}/{lang}")
            for fn in files if fn.lower().endswith(".txt")
        ]
        if not txt_files:
            print(f"⚠️  No .txt files for {lang}, skipping.")
            continue

        print(f"→ {lang}: reading {len(txt_files)} files")
        eval_ids, train_ids, total = [], [], 0

        # Read and tokenize lines
        for path in tqdm(txt_files, unit="file", desc=lang):
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    for raw in fp:
                        line = raw.strip()
                        if not line:
                            continue
                        ids = tok(line, add_special_tokens=False).input_ids
                        for tid in ids:
                            total += 1
                            if total <= TOK_EVAL:
                                eval_ids.append(tid)
                            elif total <= TARGET_TOTAL:
                                train_ids.append(tid)
                            if total == TARGET_TOTAL:
                                break
                        if total == TARGET_TOTAL:
                            break
            except UnicodeDecodeError:
                print(f"⚠️  {path} is not valid UTF-8 — skipped")
                continue

            if total == TARGET_TOTAL:
                break

        if total < TARGET_TOTAL:
            print(f"⚠️  {lang}: only {total} tokens < {TARGET_TOTAL}; still saving")

        # Save IDs
        torch.save(torch.LongTensor(eval_ids),
                   f"{OUT_DIR}/valid/id.{lang}.eval.{tag}.cultural")
        torch.save(torch.LongTensor(train_ids),
                   f"{OUT_DIR}/train/id.{lang}.train.{tag}.cultural")

        print(f"✅ {lang}: {len(eval_ids)} eval + {len(train_ids)} train tokens saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare train/eval cultural data for all languages.")
    parser.add_argument("--tag", required=True, choices=list(MODELS.keys()),
                        help="Model tag, e.g. Gemma-3-12b")
    args = parser.parse_args()
    main(args.tag)
