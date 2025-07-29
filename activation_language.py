"""
Count (>0) activations per MLP neuron for a language token stream.

Supported checkpoints (vLLM 0.2.7 + Tesla V100):
  * meta-llama/Llama-2-7b-hf
  * meta-llama/Llama-3.1-8B
  * google/gemma-3-12b
  * Qwen/Qwen2.5-7B
"""

# ------------------------------------------------------------------
# 0. Imports and HF cache paths
# ------------------------------------------------------------------
import argparse, os, torch
from types import MethodType
from vllm import LLM, SamplingParams
import vllm.config as _vc
import vllm.model_executor.layers.rotary_embedding as _rope

cache = os.path.expanduser("~/hf_cache")
os.environ["HF_HOME"]            = cache
os.environ["TRANSFORMERS_CACHE"] = cache
os.environ["HF_DATASETS_CACHE"]  = os.path.join(cache, "datasets")

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

orig_convert_tokens_to_string = PreTrainedTokenizerFast.convert_tokens_to_string
def safe_convert_tokens_to_string(self, tokens):
    if tokens is None:
        print("WARNING: convert_tokens_to_string got None (skipped)")
        return ""
    return orig_convert_tokens_to_string(self, tokens)
PreTrainedTokenizerFast.convert_tokens_to_string = safe_convert_tokens_to_string

# ------------------------------------------------------------------
# 1. CLI arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model",  default="Qwen/Qwen2.5-7B",
                    help="HF repo or local path of checkpoint")
parser.add_argument("-l", "--language", default="en",
                    help="culture / language code (en,de,da,zh,ru,fa)")
parser.add_argument("-t", "--tag", default="qwen2.5",
                    help="identifier used in ID/output filenames")
args = parser.parse_args()

model_name_l   = args.model.lower()
is_llama_like  = ("llama" in model_name_l) or ("qwen" in model_name_l)

# ------------------------------------------------------------------
# Enable vLLM-0.2.7 to load Qwen-2 checkpoints
# ------------------------------------------------------------------
import importlib

# locate the module that contains the _MODELS dict
models_pkg = importlib.import_module("vllm.model_executor.models")
try:
    _MODELS = models_pkg._MODELS                # most wheels
except AttributeError:
    raise RuntimeError("Could not find _MODELS in vllm; Qwen2 patch failed.")

# register the new key → treat it like Llama
_MODELS["Qwen2ForCausalLM"] = ("llama", "LlamaForCausalLM")


# ------------------------------------------------------------------
# 2. Hot-patch vLLM 0.2.7 for missing rope_scaling["type"]
# ------------------------------------------------------------------
def _norm(rs):
    if isinstance(rs, dict) and "type" not in rs:
        return {"type": "yarn", **rs}
    return rs

# patch a) max-len checker
_orig_len = _vc._get_and_verify_max_len
def _safe_len(cfg, max_len=None):
    cfg.rope_scaling = _norm(getattr(cfg, "rope_scaling", None))
    return _orig_len(cfg, max_len)
_vc._get_and_verify_max_len = _safe_len
# ------------------------------------------------------------------
# replacement for the previous rope patch
# ------------------------------------------------------------------
_orig_rope = _rope.get_rope

def _safe_rope(*args, **kw):
    # ---- pull args/kwargs into local vars -------------------------
    if len(args) >= 4:                # positional style
        args       = list(args)
        rope_scale = args[3]
        max_pos    = args[1]
    else:                             # keyword style
        rope_scale = kw.get("rope_scaling")
        max_pos    = (kw.get("max_pos") or
                      kw.get("max_position") or
                      kw.get("max_position_embeddings"))

    # ---- add default type -----------------------------------------
    rope_scale = _norm(rope_scale)

    # ---- if Yarn, make equality hold exactly ----------------------
    if isinstance(rope_scale, dict) and rope_scale.get("type") == "yarn":
        factor = float(rope_scale.get("factor", 1.0))
        orig   = int(round(max_pos / factor))
        # recompute factor so orig * factor == max_pos exactly
        factor = max_pos / orig
        rope_scale.update({
            "factor": factor,
            "original_max_position_embeddings": orig
        })

    # ---- push back & call original --------------------------------
    if len(args) >= 4:
        args[3] = rope_scale
        return _orig_rope(*args, **kw)
    kw["rope_scaling"] = rope_scale
    return _orig_rope(*args, **kw)

_rope.get_rope = _safe_rope

model = LLM(
    model=args.model,
    dtype="float16",
    max_model_len=4096, # << NEW: pick a length you can afford
    # max_model_len=8192,          # << NEW: pick a length you can afford
    gpu_memory_utilization=0.90, # optional: leave a little head-room
    tensor_parallel_size=torch.cuda.device_count(),
    enforce_eager=True,
    load_format="safetensors",
    trust_remote_code=True
)

cfg        = model.llm_engine.model_config
hf_cfg     = cfg.hf_config
max_len    = cfg.max_model_len
n_layers   = hf_cfg.num_hidden_layers
hidden_i   = hf_cfg.intermediate_size if is_llama_like else hf_cfg.hidden_size * 4
over_zero  = torch.zeros(n_layers, hidden_i, dtype=torch.int32, device="cuda")

# ------------------------------------------------------------------
# 4. Hook factory (fp16 path + fp32 counting)
# ------------------------------------------------------------------
def factory(idx: int):
    def llama_fwd(self, x):
        up, _ = self.gate_up_proj(x)           # fp16
        mid = up.size(-1) // 2
        up[..., :mid] = torch.nn.SiLU()(up[..., :mid])
        over_zero[idx] += (up[..., :mid].float() > 0).sum((0, 1))
        x = up[..., :mid] * up[..., mid:]      # still fp16
        x, _ = self.down_proj(x)
        return x

    def bloom_fwd(self, x):
        x, _ = self.dense_h_to_4h(x)
        x    = self.gelu_impl(x)
        over_zero[idx] += (x.float() > 0).sum((0, 1))
        x, _ = self.dense_4h_to_h(x)
        return x
    return llama_fwd if is_llama_like else bloom_fwd

# ------------------------------------------------------------------
# 5. Locate layers & attach hooks (handles Qwen path)
# ------------------------------------------------------------------
root = model.llm_engine.driver_worker.model_runner.model
if hasattr(root, "model") and hasattr(root.model, "layers"):   # Llama, Qwen-2
    layers = root.model.layers
elif hasattr(root, "transformer") and hasattr(root.transformer, "h"):  # Qwen-1, Bloom
    layers = root.transformer.h
else:
    raise RuntimeError("Unsupported model structure; cannot find layers")

for idx, layer in enumerate(layers):
    layer.mlp.forward = MethodType(factory(idx), layer.mlp)

# ------------------------------------------------------------------
# 6. Load cultural IDs
# ------------------------------------------------------------------
language  = args.language
tag       = args.tag            # filename tag chosen by user
ids_path  = f"data/train/id.{language}.train.{tag}.language"
ids       = torch.load(ids_path)

usable    = (min(len(ids), 99_999_744) // max_len) * max_len
input_ids = ids[:usable].reshape(-1, max_len)

print("input_ids shape:", input_ids.shape)
print("Sample input_ids[0]:", input_ids[0])
print("input_ids dtype:", input_ids.dtype)

# ------------------------------------------------------------------
# 7. Trigger hooks with one-step generation
# ------------------------------------------------------------------
successes = 0
total = len(input_ids)
rows = input_ids.tolist()

for i, row in enumerate(rows):
    try:
        model.generate(prompt_token_ids=[row], sampling_params=SamplingParams(max_tokens=1), use_tqdm=False)
        successes += 1
    except Exception as e:
        print(f"Skipping row {i} due to error: {e}")

    # Print progress every 100 rows
    if (i+1) % 100 == 0 or (i+1) == total:
        percent = (i+1) * 100.0 / total
        print(f"[{i+1}/{total}] ({percent:.1f}%) processed, successes: {successes}, failed: {(i+1)-successes}")

print(f"\n✓ Finished all {total} rows, succeeded on {successes}, failed: {total-successes}")

# ------------------------------------------------------------------
# 8. Save activation counts
# ------------------------------------------------------------------
out_path = f"data/activation/activation.{language}.train.{tag}.language"
torch.save(dict(n=usable, over_zero=over_zero.cpu()), out_path)
print(f"✓ Saved activation counts to {out_path}")