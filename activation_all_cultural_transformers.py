import os

cultures = ["en", "de", "da", "zh", "ru", "fa"]

# Choose your model and tag here!
MODELS = {
    "llama-2-7b":  "meta-llama/Llama-2-7b-hf",
    "llama-3.1":   "meta-llama/Llama-3.1-8B",
    "qwen2.5":     "Qwen/Qwen2.5-7B",
    "gemma-3":     "google/gemma-3-12b-pt",
}
TAG = "gemma-3"
MODEL = MODELS[TAG]

cache_path = os.path.expanduser("~/hf_cache")

for culture in cultures:
    print(f"\n>>> Running activation for {culture} <<<\n")
    os.environ["HF_HOME"] = cache_path
    os.environ["TRANSFORMERS_CACHE"] = cache_path
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Call the function in the other script
    import activation_cultural_transformers
    activation_cultural_transformers.run(
        model=MODEL,
        culture=culture,
        tag=TAG,
        cache_dir=cache_path,
    )