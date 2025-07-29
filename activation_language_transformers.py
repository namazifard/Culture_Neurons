import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

os.environ["PYTORCH_SDP_ATTENTION"] = "0"
torch.cuda.empty_cache()

def find_layers(model_obj):
    if hasattr(model_obj, "language_model") and hasattr(model_obj.language_model, "layers"):
        return model_obj.language_model.layers
    if hasattr(model_obj, "model") and hasattr(model_obj.model, "layers"):
        return model_obj.model.layers
    if hasattr(model_obj, "layers"):
        return model_obj.layers
    if hasattr(model_obj, "base_model") and hasattr(model_obj.base_model, "layers"):
        return model_obj.base_model.layers
    if hasattr(model_obj, "decoder") and hasattr(model_obj.decoder, "layers"):
        return model_obj.decoder.layers
    raise RuntimeError("Could not find model layers for activation hooks.")

def run(model_name, language, tag, cache_dir):
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

    print(f"Loading model {model_name}...")
    model_obj = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    model_obj.eval()
    model_obj.cuda()
    device = model_obj.device if hasattr(model_obj, "device") else "cuda"

    print("Finding main layers...")
    layers = find_layers(model_obj)
    n_layers = len(layers)
    cfg = model_obj.config
    if hasattr(cfg, "intermediate_size"):
        hidden_size = cfg.intermediate_size
    elif hasattr(cfg, "text_config") and hasattr(cfg.text_config, "intermediate_size"):
        hidden_size = cfg.text_config.intermediate_size
    else:
        raise RuntimeError("Could not find hidden/intermediate size in config")

    print(f"Model has {n_layers} layers, each with {hidden_size} neurons.")

    over_zero = torch.zeros(n_layers, hidden_size, dtype=torch.int32, device="cuda")

    def make_hook(idx):
        def hook(module, input, output):
            tensor = output[0] if isinstance(output, tuple) else output
            count = (tensor > 0).sum(dim=(0, 1)).int()
            over_zero[idx][:len(count)] += count.to(over_zero.device)
        return hook

    print("Registering hooks...")
    hooks = []
    for i, lyr in enumerate(layers):
        mlp = None
        if hasattr(lyr, "mlp"):
            mlp = lyr.mlp
        elif hasattr(lyr, "ffn"):
            mlp = lyr.ffn
        elif hasattr(lyr, "feed_forward"):
            mlp = lyr.feed_forward
        else:
            mlp = lyr
        h = mlp.register_forward_hook(make_hook(i))
        hooks.append(h)

    ids_path = f"data/train/id.{language}.train.{tag}.language"
    ids = torch.load(ids_path)
    max_len = model_obj.config.max_position_embeddings if hasattr(model_obj.config, "max_position_embeddings") else 4096
    usable = (min(len(ids), 99_999_744) // max_len) * max_len
    input_ids = ids[:usable].reshape(-1, max_len)

    print(f"Processing {usable} tokens ({input_ids.shape[0]} batches of {max_len})...")

    for batch in tqdm(input_ids, desc="Batches", unit="batch"):
        with torch.no_grad():
            model_obj(batch.unsqueeze(0).to(device))

    for h in hooks:
        h.remove()

    out_path = f"data/activation/activation.{language}.train.{tag}.language"
    torch.save(dict(n=usable, over_zero=over_zero.cpu()), out_path)
    print(f"✓ Saved activation counts to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-l", "--language", help="Language code (en, de, da, zh, ru, fa)")
    parser.add_argument("-t", "--tag", required=True)
    parser.add_argument("--all", action="store_true", help="Run for all languages")
    parser.add_argument("--cache_dir", default=os.path.expanduser("~/hf_cache"))
    args = parser.parse_args()

    if args.all:
        LANGS = ["en", "de", "da", "zh", "ru", "fa"]
        for lang in LANGS:
            print(f"\n>>> Running activation for {lang} <<<\n")
            run(args.model, lang, args.tag, args.cache_dir)
    else:
        if not args.language:
            parser.error("Must provide --language or use --all")
        run(args.model, args.language, args.tag, args.cache_dir)
