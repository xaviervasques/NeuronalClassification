import matplotlib.pyplot as plt
from PIL import Image

# Paths to your images (replace with your actual file names if different)
image_paths = [
    "figure_heatmap_mean_profiles_etype.png",
    "figure_heatmap_mae_per_feature_etype.png",
    "figure_heatmap_mae_per_feature_vs_test_etype.png",
    "figure_heatmap_mean_profiles_meetype.png",
    "figure_heatmap_mae_per_feature_meetype.png",
    "figure_heatmap_mae_per_feature_vs_test_meetype.png"
]

# Titles for each panel
panel_titles = [
    "A. e-type: Mean Feature Profiles",
    "B. e-type: MAE (Synthetic vs Train)",
    "C. e-type: MAE (Synthetic vs Test)",
    "D. mee-type: Mean Feature Profiles",
    "E. mee-type: MAE (Synthetic vs Train)",
    "F. mee-type: MAE (Synthetic vs Test)"
]

fig, axes = plt.subplots(2, 3, figsize=(30, 14))

for idx, (img_path, title) in enumerate(zip(image_paths, panel_titles)):
    row, col = divmod(idx, 3)
    img = Image.open(img_path)
    axes[row, col].imshow(img)
    axes[row, col].axis('off')
    axes[row, col].set_title(title, fontsize=20, pad=18)

plt.tight_layout()
plt.savefig("figure7_merged_synthetic_profiles.png", dpi=200)
plt.show()
