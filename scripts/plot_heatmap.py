import numpy as np, matplotlib.pyplot as plt
order_c = ["english","german","danish","russian","persian"]
langs   = ["en","de","da","ru","fa"]
base = np.loadtxt(f"results/{mdl}/ppl_baseline.txt")

# build delta matrix
mat = []
for cult in order_c:
    vec = np.loadtxt(f"results/ppl_mask_{cult}_pure.txt")
    mat.append(vec - base)
mat = np.vstack(mat)

fig, ax = plt.subplots(figsize=(4,4))
cax = ax.matshow(mat, cmap="Reds")
ax.set_xticks(range(5)); ax.set_xticklabels(langs)
ax.set_yticks(range(5)); ax.set_yticklabels(order_c)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", color="white")
plt.colorbar(cax, label="Δ-log-PPL")
plt.xlabel("Eval language"); plt.ylabel("Culture masked")
plt.tight_layout()
plt.savefig("results/heatmap_pure_culture.png", dpi=300)
