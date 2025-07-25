import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import string

# Noms des fichiers panels (à adapter au besoin)
filenames = [
    "Figure_6a.png", "Figure_6b.png", "Figure_6c.png", "Figure_6d.png",
    "Figure_6e.png", "Figure_6f.png", "Figure_6g.png", "Figure_6h.png",
    "Figure_6i.png", "Figure_6j.png", "Figure_6k.png", "Figure_6l.png"
]

# Paramètres de la grille
nrows, ncols = 4, 3
panel_labels = list(string.ascii_uppercase[:12])  # ['A', ..., 'L']

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 18))

for idx, ax in enumerate(axes.flat):
    img = mpimg.imread(filenames[idx])
    ax.imshow(img)
    ax.axis('off')  # Pas d'axes
    # Ajout du label (A, B, ...) en haut à gauche de chaque panel
    ax.text(
        0.02, 0.96, panel_labels[idx],
        transform=ax.transAxes, fontsize=22, fontweight="bold",
        va="top", ha="left", color="black",
        bbox=dict(facecolor='white', edgecolor='none', pad=1.0, alpha=0.7)
    )

plt.tight_layout(w_pad=0.5, h_pad=1.0)
plt.savefig("Figure_6_all_panels.png", dpi=300, bbox_inches="tight")
plt.show()
