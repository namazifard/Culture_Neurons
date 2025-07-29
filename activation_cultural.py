# activation_cultural.py

"""
Count (>0) activations per MLP neuron for a cultural token stream.

Supported checkpoints (requires vLLM main branch + Transformers v4.49.0-Gemma-3 patch):
  * meta-llama/Llama-2-7b-hf
  * meta-llama/Llama-3.1-8B
  * Qwen/Qwen-7B
  * Qwen/Qwen2-7B
  * google/gemma-3-12b-pt
"""

# ------------------------------------------------------------------
# 0. Imports and HF cache paths
# ------------------------------------------------------------------
import argparse
import os
import torch
from types import MethodType
from vllm import LLM, SamplingParams
import vllm.config as _vc
import vllm.model_executor.layers.rotary_embedding as _rope
from transformers import AutoConfig

# Set Hugging Face cache directories
cache = os.path.expanduser("~/hf_cache")
os.environ["HF_HOME"] = cache
os.environ["TRANSFORMERS_CACHE"] = cache
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache, "datasets")

# ------------------------------------------------------------------
# 1. CLI arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model",  default="google/gemma-3-12b-pt",
                    help="HF repo or local path of checkpoint")
parser.add_argument("-c", "--culture", default="en",
                    help="culture / language code (en,de,da,zh,ru,fa)")
parser.add_argument("-t", "--tag", default="gemma-3",
                    help="identifier used in ID/output filenames")
args = parser.parse_args()

model_name_l   = args.model.lower()
is_llama_like  = ("llama" in model_name_l) or ("qwen" in model_name_l)

# ------------------------------------------------------------------
# 2. Hot-patch vLLM for missing rope_scaling["type"]
# ------------------------------------------------------------------
def _norm(rs):
    if isinstance(rs, dict) and "type" not in rs:
        return {"type": "yarn", **rs}
    return rs

_orig_len = _vc._get_and_verify_max_len

# Patch max-len checker\ n_orig_len = _vc._get_and_verify_max_len
def _safe_len(cfg, max_len=None):
    cfg.rope_scaling = _norm(getattr(cfg, "rope_scaling", None))
    return _orig_len(cfg, max_len)
_vc._get_and_verify_max_len = _safe_len

# Patch rotary embedding
_orig_rope = _rope.get_rope

def _safe_rope(*args, **kw):
    if len(args) >= 4:
        args       = list(args)
        rope_scale = args[3]
        max_pos    = args[1]
    else:
        rope_scale = kw.get("rope_scaling")
        max_pos    = (kw.get("max_pos") or kw.get("max_position") or kw.get("max_position_embeddings"))

    rope_scale = _norm(rope_scale)
    if isinstance(rope_scale, dict) and rope_scale.get("type") == "yarn":
        factor = float(rope_scale.get("factor", 1.0))
        orig   = int(round(max_pos / factor))
        factor = max_pos / orig
        rope_scale.update({
            "factor": factor,
            "original_max_position_embeddings": orig
        })

    if len(args) >= 4:
        args[3] = rope_scale
        return _orig_rope(*args, **kw)
    kw["rope_scaling"] = rope_scale
    return _orig_rope(*args, **kw)

_rope.get_rope = _safe_rope

# ------------------------------------------------------------------
# 3. Monkey-patch detokenization to skip missing tokens
# ------------------------------------------------------------------
import vllm.transformers_utils.tokenizer as _tk
_orig_detok = _tk.detokenize_incrementally

def _safe_detokenize_incrementally(tokenizer, output, is_first, read_offset=0):
    if getattr(output, "tokens", None) is None:
        return "", read_offset
    return _orig_detok(tokenizer, output, is_first, read_offset)

_tk.detokenize_incrementally = _safe_detokenize_incrementally

# --- PATCH GEMMA-3 CONFIG IF NEEDED ---
try:
    hf_config = AutoConfig.from_pretrained(args.model)
except Exception as e:
    print("WARNING: Could not load model config with transformers:", e)
    hf_config = None

if hf_config is not None and "gemma" in args.model.lower():
    # vLLM expects 'num_attention_heads' but Gemma uses 'num_heads'
    if not hasattr(hf_config, "num_attention_heads") and hasattr(hf_config, "num_heads"):
        print(f"Patching: Setting num_attention_heads = {hf_config.num_heads} for Gemma config")
        hf_config.num_attention_heads = hf_config.num_heads
    # Sometimes vLLM expects 'multi_query'
    if not hasattr(hf_config, "multi_query") and hasattr(hf_config, "multi_query_attention"):
        hf_config.multi_query = hf_config.multi_query_attention

# ------------------------------------------------------------------
# 4. Initialize the LLM
# ------------------------------------------------------------------
model = LLM(
    model=args.model,
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.90,
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
# 5. Hook factory (fp16 path + fp32 counting)
# ------------------------------------------------------------------

def factory(idx: int):
    def llama_fwd(self, x):
        up, _ = self.gate_up_proj(x)
        mid = up.size(-1) // 2
        up[..., :mid] = torch.nn.SiLU()(up[..., :mid])
        over_zero[idx] += (up[..., :mid].float() > 0).sum((0, 1))
        x = up[..., :mid] * up[..., mid:]
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
# 6. Attach hooks to each MLP layer
# ------------------------------------------------------------------
root = model.llm_engine.driver_worker.model_runner.model
if hasattr(root, "model") and hasattr(root.model, "layers"):
    layers = root.model.layers
elif hasattr(root, "transformer") and hasattr(root.transformer, "h"):
    layers = root.transformer.h
else:
    raise RuntimeError("Unsupported model structure; cannot find layers")

for idx, layer in enumerate(layers):
    layer.mlp.forward = MethodType(factory(idx), layer.mlp)

# ------------------------------------------------------------------
# 7. Load cultural IDs
# ------------------------------------------------------------------
culture   = args.culture
tag       = args.tag
ids_path  = f"cultural_data/train/id.{culture}.train.{tag}.cultural"
ids       = torch.load(ids_path)

usable    = (min(len(ids), 99_999_744) // max_len) * max_len
input_ids = ids[:usable].reshape(-1, max_len)

# ------------------------------------------------------------------
# 8. Trigger hooks with one-step generation
# ------------------------------------------------------------------
model.generate(
    prompt_token_ids=input_ids.tolist(),
    sampling_params=SamplingParams(max_tokens=1)
)

# ------------------------------------------------------------------
# 9. Save activation counts
# ------------------------------------------------------------------
out_path = f"cultural_data/activation/activation.{culture}.train.{tag}.cultural"
torch.save(dict(n=usable, over_zero=over_zero.cpu()), out_path)
print(f"✓ Saved activation counts to {out_path}")
