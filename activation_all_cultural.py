import subprocess, os

cultures = ["en", "de", "da", "zh", "ru", "fa"]

# -------------------------------------------------------------------
#  ▼ choose ONE checkpoint + its matching ID-file tag
# -------------------------------------------------------------------
# Llama-2-7B
# TAG   = "llama-2-7b"
# MODEL = "meta-llama/Llama-2-7b-hf"

# Llama-3-8B
# TAG   = "Llama-3.1"
# MODEL = "meta-llama/Llama-3.1-8B"

# Qwen-2.5-7B
# TAG   = "qwen2.5"
# MODEL = "Qwen/Qwen2.5-7B"

# Gemma-3-12B
TAG = mdl   = "gemma-3"
MODEL = model = "google/gemma-3-12b-pt"
# -------------------------------------------------------------------

model = MODEL

cache_path = os.path.expanduser("~/hf_cache")

for culture in cultures:
    print(f"\n>>> Running activation for {culture} <<<\n")
    subprocess.run(
        [
            "python", "activation_cultural.py",
            "-m", model,
            "-c", culture,
            "-t", TAG           # ⬅ NEW: tells inner script which ID files
        ],
        env={
            **os.environ,
            "HF_HOME": cache_path,
            "TRANSFORMERS_CACHE": cache_path,
            "CUDA_VISIBLE_DEVICES": "0"
        }
    )