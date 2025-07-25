import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load real neurons for reference
df_train = pd.read_excel("mee-type_train_real_rescaled.xlsx")
df_test = pd.read_excel("mee-type_test_real_rescaled.xlsx")
df_synth = pd.read_csv("mee-type_flow_maxabs_10000.csv", delimiter=",")

features = [col for col in df_train.columns if col.startswith("feature_")]
df_train_clean = df_train.dropna(subset=features)
df_test_clean = df_test.dropna(subset=features)
df_synth_clean = df_synth.dropna(subset=features)

# Compute robust thresholds from real data (train+test)
df_real = pd.concat([df_train_clean, df_test_clean], ignore_index=True)
med = df_real[features].median()
mad = (df_real[features] - med).abs().median()  # Median Absolute Deviation

# Define outlier threshold (ex: > 5 MAD from median)
n_mad = 5
min_allowed = med - n_mad * mad
max_allowed = med + n_mad * mad

# Identify outliers in synthetic neurons
mask_outlier = (df_synth_clean[features] < min_allowed) | (df_synth_clean[features] > max_allowed)
to_remove = mask_outlier.any(axis=1)
print(f"Removing {to_remove.sum()} synthetic neurons out of {len(df_synth_clean)} as outliers.")

# Remove outliers
df_synth_no_outlier = df_synth_clean.loc[~to_remove].copy()
df_synth_no_outlier["origin"] = "synthetic"

# === Continue comme avant ===
df_train_clean["origin"] = "train"
df_test_clean["origin"] = "test"

# PCA fit only on real neurons
df_real = pd.concat([df_train_clean, df_test_clean], ignore_index=True)
X_real = df_real[features]
pca = PCA(n_components=2)
X_real_pca = pca.fit_transform(X_real)

# Projection of synthetic neurons (without outliers)
X_synth = df_synth_no_outlier[features]
X_synth_pca = pca.transform(X_synth)

# Prepare DataFrame for plot
df_real["PC1"] = X_real_pca[:, 0]
df_real["PC2"] = X_real_pca[:, 1]
df_synth_no_outlier["PC1"] = X_synth_pca[:, 0]
df_synth_no_outlier["PC2"] = X_synth_pca[:, 1]
df_plot = pd.concat([df_real, df_synth_no_outlier], ignore_index=True)

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
ax.set_title("PCA (fit on real neurons) – Real (train/test) vs Synthetic Neurons (Flow-MaxAbs, 10000 synthetic neurons, outliers removed)")
ax.legend()
plt.tight_layout()
ax.text(-0.23, 1.18, "A", fontsize=24, fontweight="bold", transform=ax.transAxes, va="top", ha="left")
plt.show()
