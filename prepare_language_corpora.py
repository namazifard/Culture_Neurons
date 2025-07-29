import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

MODELS = {
    "Llama-2-7b":     "meta-llama/Llama-2-7b-hf",
    "Llama-3.1-8b":   "meta-llama/Llama-3.1-8B",
    "Qwen2.5-7b":     "Qwen/Qwen2.5-7B",
    "Gemma-3-12b":    "google/gemma-3-12b-pt",
}
LANGS         = ['en', 'de', 'da', 'zh', 'ru', 'fa']
WIKI_VERSION  = "20231101"

# Corpus sizes (adjust if needed)
TRAIN_TOKENS = 100_000_000
EVAL_TOKENS  = 1_000_000
SKIP_TOKENS  = 110_000_000   # For eval, to reduce overlap

def prepare_split(lang, tag, model_name, split, target_tokens, out_path, skip_tokens=0):
    if os.path.exists(out_path):
        print(f"✅ Already exists: {out_path}")
        return

    try:
        config = f"{WIKI_VERSION}.{lang}"
        print(f"Streaming Wikipedia: {config} ({split})")
        dataset = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)

        all_token_ids = []
        skipped = 0
        total_tokens = 0

        for example in tqdm(dataset, desc=f"{split.capitalize()} {lang}", unit="article"):
            tokens = tokenizer(example["text"], return_attention_mask=False, return_token_type_ids=False)["input_ids"]

            if skip_tokens and skipped < skip_tokens:
                skipped += len(tokens)
                continue

            all_token_ids.extend(tokens)
            total_tokens += len(tokens)

            if total_tokens >= target_tokens:
                break

        all_token_ids = all_token_ids[:target_tokens]
        token_tensor = torch.LongTensor(all_token_ids)
        torch.save(token_tensor, out_path)
        print(f"✅ Saved {len(token_tensor)} tokens to: {out_path}")

    except Exception as e:
        print(f"❌ Failed to process {lang.upper()} ({split}): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare both train and eval corpora for all languages for a given model tag.")
    parser.add_argument("--tag", required=True, choices=list(MODELS.keys()), help="Model tag (e.g. Gemma-3-12b)")
    args = parser.parse_args()

    tag = args.tag
    model_name = MODELS[tag]
    print(f"Using model: {model_name} ({tag})")

    os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache")
    os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/hf_cache")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    os.makedirs("data", exist_ok=True)
    os.makedirs("eval_data", exist_ok=True)

    for lang in LANGS:
        print(f"\n--- Processing {lang.upper()} ---")

        train_path = f"data/train/id.{lang}.train.{tag}.language"
        eval_path  = f"data/valid/id.{lang}.eval.{tag}.language"

        # Train
        prepare_split(
            lang, tag, model_name,
            split="train",
            target_tokens=TRAIN_TOKENS,
            out_path=train_path,
            skip_tokens=0
        )

        # Eval (after SKIP_TOKENS tokens)
        prepare_split(
            lang, tag, model_name,
            split="eval",
            target_tokens=EVAL_TOKENS,
            out_path=eval_path,
            skip_tokens=SKIP_TOKENS
        )
