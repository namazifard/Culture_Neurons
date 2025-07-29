import os
import activation_language_transformers

langs = ["en", "de", "da", "zh", "ru", "fa"]

MODELS = {
    "Llama-2-7b":  "meta-llama/Llama-2-7b-hf",
    "Llama-3.1-8b":   "meta-llama/Llama-3.1-8B",
    "Qwen2.5-7b":     "Qwen/Qwen2.5-7B",
    "Gemma-3-12b":     "google/gemma-3-12b-pt",
}

TAG = "Gemma-3-12b"
MODEL = MODELS[TAG]
cache_path = os.path.expanduser("~/hf_cache")

for language in langs:
    print(f"\n>>> Running activation for {language} <<<\n")
    os.environ["HF_HOME"] = cache_path
    os.environ["TRANSFORMERS_CACHE"] = cache_path
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    activation_language_transformers.run(
        model_name=MODEL,
        language=language,
        tag=TAG,
        cache_dir=cache_path,
    )
