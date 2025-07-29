import torch
import torch.nn.functional as F
import os

# # for qwen2.5
# n, over_zero = [], []
# for lang in ['en', 'de', 'da', 'zh', 'ru', 'fa']:
#     data = torch.load(f'data/activation/activation.{lang}.train.{mdl}.language')
#     n.append(data['n'])
#     # Only take the first 3584 neurons
#     over_zero.append(data['over_zero'][:,:3584])

# n = torch.tensor(n)
# over_zero = torch.stack(over_zero, dim=-1)

# num_layers, intermediate_size, lang_num = over_zero.shape


## select the model

mdl = TAG = "Llama-2-7b"
MODEL = model = "meta-llama/Llama-2-7b-hf"

# mdl = TAG = "Qwen2.5-7b"
# MODEL = model = "Qwen/Qwen2.5-7B"

# mdl = TAG = "Llama-3.1-8b"
# MODEL = model = "meta-llama/Llama-3.1-8B"

# mdl = TAG = "Gemma-3-12b"
# MODEL = model = "google/gemma-3-12b-pt"

# Llamas
n, over_zero = [], []
for lang in ['en', 'de', 'da', 'zh', 'ru', 'fa']:
    data = torch.load(f'data/activation/activation.{lang}.train.{mdl}.language')
    n.append(data['n'])
    over_zero.append(data['over_zero'])

n = torch.tensor(n)
over_zero = torch.stack(over_zero, dim=-1)

num_layers, intermediate_size, lang_num = over_zero.size()

def activation():
    top_rate = 0.01
    filter_rate = 0.95
    activation_bar_ratio = 0.95
    activation_probs = over_zero / n # layer x inter x lang_num
    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    largest = False
    
    if torch.isnan(entropy).sum():
        print(torch.isnan(entropy).sum())
        raise ValueError
    
    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * filter_rate)).values.item()
    print(top_prob_value)
    # dismiss the neruon if no language has an activation value over top 90%
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    flattened_entropy = entropy.flatten()
    top_entropy_value = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index] # n x lang
    # for r, c in zip(row_index, col_index):
    #     print(r, c, activation_probs[r][c])

    print(selected_probs.size(0), torch.bincount(selected_probs.argmax(dim=-1)))
    selected_probs = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * activation_bar_ratio)).values.item()
    print((selected_probs > activation_bar).sum(dim=1).tolist())
    lang, indice = torch.where(selected_probs > activation_bar)

    merged_index = torch.stack((row_index, col_index), dim=-1)
    final_indice = []
    for _, index in enumerate(indice.split(torch.bincount(lang).tolist())):
        lang_index = [tuple(row.tolist()) for row in merged_index[index]]
        lang_index.sort()
        layer_index = [[] for _ in range(num_layers)]
        for l, h in lang_index:
            layer_index[l].append(h)
        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()
        final_indice.append(layer_index)
    os.makedirs("activation_mask", exist_ok=True)
    torch.save(final_indice, f"activation_mask/{mdl}.language")  

activation()

# import torch
# import os
# from transformers import AutoModelForCausalLM

# # ------------ Model/tag config ------------
# mdl = TAG = "Qwen2.5-7b"
# MODEL = model = "Qwen/Qwen2.5-7B"
# # mdl = TAG = "Gemma-3-12b"
# # MODEL = model = "google/gemma-3-12b-pt"

# LANGS = ['en', 'de', 'da', 'zh', 'ru', 'fa']
# n, over_zero = [], []

# # ------------ Load activations ------------
# for lang in LANGS:
#     path = f'data/activation/activation.{lang}.train.{mdl}.language'
#     print(f"Loading {path}")
#     data = torch.load(path)
#     n.append(data['n'])
#     over_zero.append(data['over_zero'])

# n = torch.tensor(n)
# over_zero = torch.stack(over_zero, dim=-1)
# num_layers, intermediate_size, lang_num = over_zero.size()

# # ------------ Load model for sanity checks ------------
# print(f"Loading model {MODEL} for dimension check...")
# model = AutoModelForCausalLM.from_pretrained(MODEL)
# if hasattr(model, "model"):
#     layers = model.model.layers
# else:
#     layers = model.layers

# mlp0 = layers[0].mlp
# if hasattr(mlp0, "gate_up_proj"):  # Qwen2/3, Gemma-3, etc
#     hidden_sizes = [lyr.mlp.gate_up_proj.weight.shape[1] // 2 for lyr in layers]
# elif hasattr(mlp0, "up_proj"):
#     hidden_sizes = [lyr.mlp.up_proj.weight.shape[1] for lyr in layers]
# else:
#     raise RuntimeError("Unrecognised MLP structure in model")

# print("Per-layer hidden sizes (from actual model):", hidden_sizes)
# if any(hs != intermediate_size for hs in hidden_sizes):
#     print(f"WARNING: Your activation tensor shape ({intermediate_size}) does not match model hidden sizes! This could cause bugs.")

# # ------------ Main selection logic ------------
# def activation():
#     top_rate = 0.01
#     filter_rate = 0.95
#     activation_bar_ratio = 0.95
#     activation_probs = over_zero / n  # [layer x inter x lang]
#     normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
#     normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
#     log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
#     entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
#     largest = False

#     if torch.isnan(entropy).sum():
#         print(torch.isnan(entropy).sum())
#         raise ValueError

#     flattened_probs = activation_probs.flatten()
#     top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * filter_rate)).values.item()
#     print(f"Filter top_prob_value (for filtering): {top_prob_value}")
#     # dismiss the neuron if no language has an activation value over top 95%
#     top_position = (activation_probs > top_prob_value).sum(dim=-1)
#     entropy[top_position == 0] = -torch.inf if largest else torch.inf

#     flattened_entropy = entropy.flatten()
#     top_entropy_value = round(len(flattened_entropy) * top_rate)
#     _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
#     row_index = index // entropy.size(1)
#     col_index = index % entropy.size(1)
#     selected_probs = activation_probs[row_index, col_index]  # [n x lang]

#     print(f"Selected probs shape: {selected_probs.shape}, bincount: {torch.bincount(selected_probs.argmax(dim=-1))}")
#     selected_probs = selected_probs.transpose(0, 1)
#     activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * activation_bar_ratio)).values.item()
#     print(f"Activation_bar: {activation_bar}, over_bar count:", (selected_probs > activation_bar).sum(dim=1).tolist())
#     lang, indice = torch.where(selected_probs > activation_bar)

#     merged_index = torch.stack((row_index, col_index), dim=-1)
#     final_indice = []
#     for lang_id, index in enumerate(indice.split(torch.bincount(lang).tolist())):
#         lang_index = [tuple(row.tolist()) for row in merged_index[index]]
#         lang_index.sort()
#         layer_index = [[] for _ in range(num_layers)]
#         for l, h in lang_index:
#             layer_index[l].append(h)
#         # ----------- Debug: Check against model hidden size ----------
#         for l, h in enumerate(layer_index):
#             if len(h) > 0 and max(h) >= hidden_sizes[l]:
#                 print(f"WARNING: Lang {lang_id} Layer {l} has index {max(h)} >= hidden_size {hidden_sizes[l]}")
#                 # Remove out-of-bounds indices
#                 h = [x for x in h if x < hidden_sizes[l]]
#             layer_index[l] = torch.tensor(h).long()
#         final_indice.append(layer_index)
#     os.makedirs("activation_mask", exist_ok=True)
#     torch.save(final_indice, f"activation_mask/{mdl}.language")
#     print(f"✓ Saved mask indices to activation_mask/{mdl}.language")
#     # Diagnostics: print layer counts
#     for lang_idx, lang_layers in enumerate(final_indice):
#         for layer_idx, inds in enumerate(lang_layers):
#             print(f"Lang {lang_idx}, Layer {layer_idx}: {len(inds)} indices (max={inds.max().item() if len(inds)>0 else 'n/a'})")

# activation()