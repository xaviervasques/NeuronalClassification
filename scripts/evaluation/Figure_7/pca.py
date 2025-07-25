import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Data Loading
df_train = pd.read_excel("e-type_train_real_rescaled.xlsx")
df_test = pd.read_excel("e-type_test_real_rescaled.xlsx")
df_synth = pd.read_csv("e-type_ddpm_yeo_35.csv", delimiter=",")

# Remove NaN rows **separately in each dataset**
features = [col for col in df_train.columns if col.startswith("feature_")]
df_train_clean = df_train.dropna(subset=features)
df_test_clean = df_test.dropna(subset=features)
df_synth_clean = df_synth.dropna(subset=features)

# Add origin label
df_train_clean["origin"] = "train"
df_test_clean["origin"] = "test"
df_synth_clean["origin"] = "synthetic"

# PCA on real neurons only (train+test)
df_real = pd.concat([df_train_clean, df_test_clean], ignore_index=True)
X_real = df_real[features]

pca = PCA(n_components=2)
X_real_pca = pca.fit_transform(X_real)

# Projection of synthetic neurons in the real neuron PCA space
X_synth = df_synth_clean[features]
X_synth_pca = pca.transform(X_synth)

# Assemble DataFrame for plotting
df_real["PC1"] = X_real_pca[:, 0]
df_real["PC2"] = X_real_pca[:, 1]
df_synth_clean["PC1"] = X_synth_pca[:, 0]
df_synth_clean["PC2"] = X_synth_pca[:, 1]

# Combine for plot
df_plot = pd.concat([df_real, df_synth_clean], ignore_index=True)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
colors = {"synthetic": "#238b45", "train": "#08519c", "test": "#d73027"}

for origin in ["synthetic", "train", "test"]:
    subset = df_plot[df_plot["origin"] == origin]
    ax.scatter(
        subset["PC1"], subset["PC2"],
        label=origin.capitalize(),
        alpha=0.35,
        s=30,
        color=colors[origin]
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA (fit on real neurons) – Real (train/test) vs Synthetic Neurons (DDPM-Yeo, 35 synthetic neurons)")
ax.legend()
plt.tight_layout()
ax.text(-0.23, 1.18, "A", fontsize=24, fontweight="bold", transform=ax.transAxes, va="top", ha="left")
plt.show()
