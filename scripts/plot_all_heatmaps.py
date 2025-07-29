import numpy as np
import matplotlib.pyplot as plt

mdl = "llama-2-7b"
MODEL = 'meta-llama/Llama-2-7b-hf'
model = 'meta-llama/Llama-2-7b-hf'

base = np.loadtxt(f"results/{mdl}/ppl_mask_baseline.txt")
langs = ['en', 'de', 'da', 'zh', 'ru', 'fa']
cults = ["english", "german", "danish", "chinese", "russian", "persian"]

ls = np.vstack([np.loadtxt(f"results/{mdl}/ppl_mask_{l}.txt") - base for l in langs])
cs = np.vstack([np.loadtxt(f"results/{mdl}/ppl_mask_{c}.txt") - base for c in cults])
pc = np.vstack([np.loadtxt(f"results/{mdl}/ppl_mask_{c}_pure.txt") - base for c in cults])
lc = np.vstack([np.loadtxt(f"results/{mdl}/ppl_mask_{l}_compound.txt") - base for l in langs])

def plot_heat(matrix, rows, cols, title, fname):
    plt.figure()
    im = plt.imshow(matrix, cmap="Reds")
    plt.xticks(range(len(cols)), cols)
    plt.yticks(range(len(rows)), rows)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center")
    # plt.xlabel("Evaluation language")
    # plt.ylabel(title)
    # plt.title(f"Δ-log-PPL: {title}")
    # plt.colorbar(im, label="Δ-log-PPL")
    # plt.colorbar(im, label="")
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")
    plt.tight_layout()
    plt.savefig(f"results/{mdl}/{fname}.pdf",
                format="pdf",
                dpi=1200,
                bbox_inches="tight")

plot_heat(ls, langs, langs, "Language-specific neurons", "ppl_language")
plot_heat(cs, langs, langs, "Culture-specific neurons", "ppl_culture")
plot_heat(pc, langs, langs, "Pure culture-specific neurons", "ppl_pure_culture")
plot_heat(lc, langs, langs, "Language∧Culture neurons", "ppl_language_culture")
