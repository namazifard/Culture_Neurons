import subprocess, os

langs = ["en", "de", "da", "zh", "ru", "fa"]

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
TAG = mdl  = "qwen2.5"
MODEL = model = "Qwen/Qwen2.5-7B"
# -------------------------------------------------------------------

cache = os.path.expanduser("~/hf_cache")

for lang in langs:
    print(f"\n>>> Running activation for {lang} <<<\n")
    subprocess.run(
        [
            "python", "activation_language.py",
            "-m", MODEL,        # checkpoint
            "-l", lang,         # language
            "-t", TAG           # ⬅ NEW: tells inner script which ID files
        ],
        env={
            **os.environ,
            "HF_HOME": cache,
            "TRANSFORMERS_CACHE": cache,
            "CUDA_VISIBLE_DEVICES": "0"
        },
        check=True
    )