"""
Build a per-layer boolean mask for a single language from the LAPE
neuron-index file. Works with BLOOM-7B and LLAMA-family models.

Usage:
  python mask_utils.py \
      --neurons activation_mask/bloom-7b.language \
      --model   bigscience/bloom-7b1 \
      --lang_id 0 \
      --out     activation_mask/en.mask.pth
"""

import argparse, torch
from transformers import AutoModelForCausalLM, AutoConfig

# def load_layers(model):
#     """Return (iterable_of_layers, fn -> hidden_size) for Bloom / Llama."""
#     arch = model.config.architectures[0].lower()
#     if "bloom" in arch:
#         layers = model.transformer.h
#         get_hidden = lambda lyr: lyr.mlp.dense_h_to_4h.weight.shape[1]
#     elif "llama" in arch:
#         layers = model.model.layers
#         get_hidden = lambda lyr: lyr.mlp.up_proj.weight.shape[1]
#     else:
#         raise ValueError(f"Unsupported architecture: {arch}")
#     return layers, get_hidden

def load_layers(model):
    """Return (iterable_of_layers, fn -> hidden_size) for Bloom / Llama / Qwen."""
    arch = model.config.architectures[0].lower()

    if "bloom" in arch:
        layers = model.transformer.h
        get_hidden = lambda lyr: lyr.mlp.dense_h_to_4h.weight.shape[1]
    elif "llama" in arch:
        layers = model.model.layers
        get_hidden = lambda lyr: lyr.mlp.up_proj.weight.shape[0]
    elif "qwen2" in arch or "qwen3" in arch or arch.startswith("qwen"):
        # Qwen-1, 2, 3 all keep Llama-style layer layout
        layers = model.model.layers
        mlp0   = layers[0].mlp
        if hasattr(mlp0, "up_proj"):                         # Qwen-1
            get_hidden = lambda lyr: lyr.mlp.up_proj.weight.shape[1]
        elif hasattr(mlp0, "gate_up_proj"):                  # Qwen-2/3
            get_hidden = lambda lyr: lyr.mlp.gate_up_proj.weight.shape[1] // 2
        else:
            raise ValueError("Unrecognised MLP layout in Qwen family")

    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    return layers, get_hidden

def build_masks(neuron_file, model_name, lang_id, out_file):
    # neuron_list[lang][layer]   according to the repo README
    neuron_list = torch.load(neuron_file)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16)
    model.to("cpu")

    layers, get_hidden = load_layers(model)
    masks = []

    # for layer_idx, layer in enumerate(layers):
    #     hdim = get_hidden(layer)
    #     mask = torch.zeros(hdim, dtype=torch.bool)
    #     try:
    #         indices = neuron_list[lang_id][layer_idx]    # Tensor of ints
    #         mask[indices] = True
    #     except IndexError:
    #         # no indices for this layer/lang — leave mask all-False
    #         pass
    #     masks.append(mask)

    for layer_idx, layer in enumerate(layers):
        hdim = get_hidden(layer)
        mask = torch.zeros(hdim, dtype=torch.bool)
        try:
            indices = neuron_list[lang_id][layer_idx]
            # Only use indices in [0, hdim)
            indices = indices[(indices >= 0) & (indices < hdim)]
            mask[indices] = True
        except IndexError:
            pass
        masks.append(mask)
    
    # for layer_idx, layer in enumerate(layers):
    # hdim = get_hidden(layer)
    # mask = torch.zeros(hdim, dtype=torch.bool)
    # try:
    #     indices = neuron_list[lang_id][layer_idx]    # Tensor of ints
    #     # Print diagnostic info
    #     print(f"[Layer {layer_idx}] hidden_dim: {hdim}, indices: {indices.tolist()}")
    #     if len(indices) > 0:
    #         print(f"   min(idx): {min(indices.tolist())}, max(idx): {max(indices.tolist())}")
    #         # Warn about out-of-bounds
    #         if any(idx >= hdim for idx in indices.tolist()):
    #             print(f"   WARNING: Some indices >= hdim ({hdim})! Will be ignored.")
    #         # Filter OOB indices:
    #         indices = indices[indices < hdim]
    #         mask[indices] = True
    # except Exception as e:
    #     print(f"Exception in layer {layer_idx}: {e}")
    # masks.append(mask)

    torch.save(masks, out_file)
    print(f"Saved mask → {out_file}")
    total = sum(m.sum().item() for m in masks)
    print(f"Total neurons set in mask: {total}")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", required=True)
    ap.add_argument("--model",   required=True)
    ap.add_argument("--lang_id", type=int, required=True)
    ap.add_argument("--out",     required=True)
    args = ap.parse_args()
    build_masks(args.neurons, args.model, args.lang_id, args.out)

# """
# Build a per-layer boolean mask for a single language from the LAPE
# neuron-index file. Works with BLOOM-7B and LLAMA-family models.

# Usage:
#   python mask_utils.py \
#       --neurons activation_mask/bloom-7b.language \
#       --model   bigscience/bloom-7b1 \
#       --lang_id 0 \
#       --out     activation_mask/en.mask.pth
# """

# import argparse, torch
# from transformers import AutoModelForCausalLM

# def load_layers(model):
#     """Return (iterable_of_layers, fn -> hidden_size) for Bloom / Llama / Qwen."""
#     arch = model.config.architectures[0].lower()

#     if "bloom" in arch:
#         layers = model.transformer.h
#         get_hidden = lambda lyr: lyr.mlp.dense_h_to_4h.weight.shape[1]
#     elif "llama" in arch:
#         layers = model.model.layers
#         get_hidden = lambda lyr: lyr.mlp.up_proj.weight.shape[1]
#     elif "qwen2" in arch or "qwen3" in arch or arch.startswith("qwen"):
#         # Qwen-1, 2, 3 all keep Llama-style layer layout
#         layers = model.model.layers
#         mlp0   = layers[0].mlp
#         if hasattr(mlp0, "up_proj"):                         # Qwen-1
#             get_hidden = lambda lyr: lyr.mlp.up_proj.weight.shape[1]
#         elif hasattr(mlp0, "gate_up_proj"):                  # Qwen-2/3
#             get_hidden = lambda lyr: lyr.mlp.gate_up_proj.weight.shape[1] // 2
#         else:
#             raise ValueError("Unrecognised MLP layout in Qwen family")
#     else:
#         raise ValueError(f"Unsupported architecture: {arch}")

#     return layers, get_hidden

# def build_masks(neuron_file, model_name, lang_id, out_file):
#     neuron_list = torch.load(neuron_file)
#     print(f"Loaded neuron list from {neuron_file}: {len(neuron_list)} languages.")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name, torch_dtype=torch.float16)
#     model.to("cpu")

#     layers, get_hidden = load_layers(model)
#     print(f"Model: {model_name}, layers: {len(layers)}")
#     masks = []
#     for layer_idx, layer in enumerate(layers):
#         hdim = get_hidden(layer)
#         mask = torch.zeros(hdim, dtype=torch.bool)
#         try:
#             indices = neuron_list[lang_id][layer_idx]    # Tensor of ints
#             print(f"[Layer {layer_idx}] mask size: {hdim}, {len(indices)} indices, min: {indices.min().item() if len(indices) > 0 else 'n/a'}, max: {indices.max().item() if len(indices) > 0 else 'n/a'}")
#             # Filter out-of-bounds indices
#             indices = indices[indices < hdim]
#             if len(indices) > 0 and indices.max().item() >= hdim:
#                 print(f"WARNING: Found out-of-bounds index in layer {layer_idx}")
#             mask[indices] = True
#         except Exception as e:
#             print(f"Exception in layer {layer_idx}: {e}")
#         masks.append(mask)

#     torch.save(masks, out_file)
#     print(f"Saved mask → {out_file}")
#     total = sum(m.sum().item() for m in masks)
#     print(f"Total neurons set in mask: {total}")
#     for i, m in enumerate(masks):
#         print(f"Layer {i}: {m.sum().item()} neurons activated, mask size: {len(m)}")

# # ---------- CLI ----------
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--neurons", required=True)
#     ap.add_argument("--model",   required=True)
#     ap.add_argument("--lang_id", type=int, required=True)
#     ap.add_argument("--out",     required=True)
#     args = ap.parse_args()
#     build_masks(args.neurons, args.model, args.lang_id, args.out)
