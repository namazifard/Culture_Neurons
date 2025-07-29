This repository contains the dataset and code for the paper "Isolating Culture Neurons in Multilingual Large Language Models".

> **Abstract:** Language and culture are deeply intertwined, yet it is so far unclear how and where multilingual large language models encode culture. Here, we extend upon an established methodology for  identifying language-specific neurons and extend it to localize and isolate culture-specific neurons, carefully disentangling their overlap and interaction with language-specific neurons. To facilitate our experiments, we introduce MUREL, a curated dataset of 85.2 million tokens spanning six different cultures. Our localization and intervention experiments show that LLMs encode different cultures in distinct neuron populations, predominantly in upper layers, and that these culture neurons can be modulated independently from language-specific neurons or those specific to other cultures. These findings suggest that cultural knowledge and propensities in multilingual language models can be isolated and editing, promoting fairness, inclusivity, and alignment.

## Reproducing

1. Preparing data:
    ```bash
    pip install torch transformers datasets
    cd LAPE
    python prepare_language_corpora.py --tag mdl # Llama-2-7b, Llama-3.1-8b, Gemma-3-12b, Qwen2.5-7b
    python prepare_culture_corpora.py --tag mdl
    ```
    Output: data/train/id.{LANG}.train.{mdl}

2. Calculating Activation:
    ```bash
    pip install torch transformers accelerate
    # for all languages
    python activation_language_transformers.py -m google/gemma-3-12b-pt -t Gemma-3-12b --all
    
    python activation_language_transformers.py -m google/gemma-3-12b-pt -l en -t Gemma-3-12b
    python activation_cultural_transformers.py -m google/gemma-3-12b-pt -c de -t Gemma-3-12b
    ```
    
3. Dependencies:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   source ~/.bashrc
   which pyenv
   pyenv --version
   ```

4. Identifying language- and culture-specific neurons:
   ```bash
   python identify.py
   ```
5. Generate per-language masks
```bash
langs=(en de da zh ru fa)
for idx in "${!langs[@]}"; do
  lang=${langs[idx]}
  python mask_utils.py \
    --neurons activation_mask/Llama-2-7b.language \
    --model   meta-llama/Llama-2-7b-hf \
    --lang_id $idx \
    --out     activation_mask/Llama-2-7b/${lang}.mask.pth
done
```
6. Generate per-culture masks
```bash
cults=(english german danish chinese russian persian)
for idx in "${!cults[@]}"; do
  cult=${cults[idx]}
  python mask_utils.py \
    --neurons activation_mask/Llama-2-7b.cultural \
    --model   meta-llama/Llama-2-7b-hf \
    --lang_id $idx \
    --out     activation_mask/Llama-2-7b/${cult}.mask.pth
done
```

# baseline
```bash
python ppl.py -m meta-llama/Llama-2-7b-hf | tee results/Llama-2-7b/ppl_mask_baseline.txt
```
# (A) zero language neurons
```bash
for lang in en de da zh ru fa; do
  python ppl.py -m meta-llama/Llama-2-7b-hf \
    -a activation_mask/Llama-2-7b/${lang}.mask.pth \
  | tee results/Llama-2-7b/ppl_mask_${lang}.txt
done
```
# (B) zero culture neurons
```bash
for cult in english german danish chinese russian persian; do
  python ppl.py -m meta-llama/Llama-2-7b-hf \
    -a activation_mask/Llama-2-7b/${cult}.mask.pth \
  | tee results/Llama-2-7b/ppl_mask_${cult}.txt
done
```
# (C) zero pure culture neurons
```bash
for cult in english german danish chinese russian persian; do
  python ppl.py -m meta-llama/Llama-2-7b-hf \
    -a activation_mask/Llama-2-7b/${cult}_pure.mask.pth \
  | tee results/Llama-2-7b/ppl_mask_${cult}_pure.txt
done
```
# (D) zero language∧culture neurons
```bash
for lang in en de da zh ru fa; do
  python ppl.py -m meta-llama/Llama-2-7b-hf \
    -a activation_mask/Llama-2-7b/${lang}_compound.mask.pth \
  | tee results/Llama-2-7b/ppl_mask_${lang}_compound.txt
done
```

7.
```bash
python scripts/count_neurons.py
```
```bash
python scripts/combine_results.py
```
```bash
python scripts/plot_all_heatmaps.py
```
