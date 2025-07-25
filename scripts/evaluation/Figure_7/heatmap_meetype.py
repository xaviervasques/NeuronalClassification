import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Chargement des données ---
df_train = pd.read_excel("mee-type_train_real_rescaled.xlsx")
df_test = pd.read_excel("mee-type_test_real_rescaled.xlsx")
df_synth = pd.read_csv("mee-type_smote_maxabs_5000.csv", delimiter=",")

df_train["origin"] = "train"
df_test["origin"] = "test"
df_synth["origin"] = "synthetic"

df_all = pd.concat([df_train, df_test, df_synth], ignore_index=True)
features = [col for col in df_all.columns if col.startswith("feature_")]

# --- Calcul des moyennes par mee-type et origine ---
mean_profiles = df_all.groupby(["e-type", "origin"])[features].mean().reset_index()
mean_profiles["row_id"] = mean_profiles["e-type"].astype(str) + " (" + mean_profiles["origin"] + ")"
mean_profiles.set_index("row_id", inplace=True)

# ----------------------------------------
# 1. HEATMAP PROFILS MOYENS
# ----------------------------------------
plt.figure(figsize=(18, max(6, 0.4*mean_profiles.shape[0])))
sns.heatmap(mean_profiles[features], cmap="viridis", cbar=True)
plt.title("Feature mean profiles by mee-type", fontsize=40)
plt.ylabel("mee-type - Train, Test, Synthetic", fontsize=22)
plt.xlabel("Feature", fontsize=22)
plt.tight_layout()
plt.savefig("figure_heatmap_mean_profiles.png", dpi=300)
plt.close()

# ----------------------------------------
# 2. DISTANCE EUCLIDIENNE synthetic vs train/test
# ----------------------------------------
e_types = mean_profiles["e-type"].unique()
distances_train = []
distances_test = []
for e_type in e_types:
    try:
        synth = mean_profiles.loc[f"{e_type} (synthetic)", features].values
        train = mean_profiles.loc[f"{e_type} (train)", features].values
        test = mean_profiles.loc[f"{e_type} (test)", features].values
        distances_train.append(np.linalg.norm(synth - train))
        distances_test.append(np.linalg.norm(synth - test))
    except KeyError:
        distances_train.append(np.nan)
        distances_test.append(np.nan)

x = np.arange(len(e_types))
width = 0.35
plt.figure(figsize=(12, 4))
plt.bar(x - width/2, distances_train, width, label='Synthetic vs Train')
plt.bar(x + width/2, distances_test, width, label='Synthetic vs Test')
plt.xticks(x, e_types, rotation=45)
plt.ylabel('Euclidean distance')
plt.title('Distance between synthetic and real mean profiles (by mee-type)', fontsize=22)
plt.legend()
plt.tight_layout()
plt.savefig("figure_barplot_distance_profiles.png", dpi=300)
plt.close()

# ----------------------------------------
# 3. MAE PAR FEATURE ET PAR E-TYPE (synthetic vs train)
# ----------------------------------------
mae_matrix = []
for e_type in e_types:
    try:
        synth = mean_profiles.loc[f"{e_type} (synthetic)", features].values
        train = mean_profiles.loc[f"{e_type} (train)", features].values
        mae = np.abs(synth - train)
        mae_matrix.append(mae)
    except KeyError:
        mae_matrix.append([np.nan]*len(features))

mae_matrix = np.array(mae_matrix, dtype=float)

plt.figure(figsize=(18, max(5, 0.4*len(e_types))))
sns.heatmap(mae_matrix, xticklabels=features, yticklabels=e_types, cmap="rocket", cbar=True)
plt.title("Mean absolute error (synthetic vs train) per feature and mee-type", fontsize=22)
plt.xlabel("Feature", fontsize=22)
plt.ylabel("mee-type", fontsize=22)
plt.tight_layout()
plt.savefig("figure_heatmap_mae_per_feature.png", dpi=300)
plt.close()

# MAE PER FEATURE AND E-TYPE (synthetic vs test)
mae_matrix_test = []
for e_type in e_types:
    try:
        synth = mean_profiles.loc[f"{e_type} (synthetic)", features].values
        test = mean_profiles.loc[f"{e_type} (test)", features].values
        mae = np.abs(synth - test)
        mae_matrix_test.append(mae)
    except KeyError:
        mae_matrix_test.append([np.nan]*len(features))

mae_matrix_test = np.array(mae_matrix_test, dtype=float)

plt.figure(figsize=(18, max(5, 0.4*len(e_types))))
sns.heatmap(mae_matrix_test, xticklabels=features, yticklabels=e_types, cmap="rocket", cbar=True)
plt.title("Mean absolute error (synthetic vs test) per feature and mee-type", fontsize=22)
plt.xlabel("Feature", fontsize=22)
plt.ylabel("mee-type", fontsize=22)
plt.tight_layout()
plt.savefig("figure_heatmap_mae_per_feature_vs_test.png", dpi=300)
plt.close()

# Save MAE matrix for synthetic vs test
df_mae_test = pd.DataFrame(mae_matrix_test, index=e_types, columns=features)
df_mae_test.to_csv("mae_matrix_synthetic_vs_test.csv")



# Sauvegarder les distances euclidiennes
df_dist = pd.DataFrame({
    'mee-type': e_types,
    'euclidean_distance_train': distances_train,
    'euclidean_distance_test': distances_test
})
df_dist.to_csv("distances_euclidiennes_profiles.csv", index=False)

# Sauvegarder la MAE par feature et par e-type
df_mae = pd.DataFrame(mae_matrix, index=e_types, columns=features)
df_mae.to_csv("mae_matrix_synthetic_vs_train.csv")
