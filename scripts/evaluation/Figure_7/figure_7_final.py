import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----- 1. Load data for e-type -----
df_train_e = pd.read_excel("e-type_train_real_rescaled.xlsx")
df_test_e = pd.read_excel("e-type_test_real_rescaled.xlsx")
df_synth_e = pd.read_csv("e-type_smote_yeo_5000.csv", delimiter=",")

df_train_e["origin"] = "train"
df_test_e["origin"] = "test"
df_synth_e["origin"] = "synthetic"
df_all_e = pd.concat([df_train_e, df_test_e, df_synth_e], ignore_index=True)
features_e = [col for col in df_all_e.columns if col.startswith("feature_")]
mean_profiles_e = df_all_e.groupby(["e-type", "origin"])[features_e].mean().reset_index()
mean_profiles_e["row_id"] = mean_profiles_e["e-type"].astype(str) + " (" + mean_profiles_e["origin"] + ")"
mean_profiles_e.set_index("row_id", inplace=True)

# ----- 2. Load data for mee-type -----
df_train_m = pd.read_excel("mee-type_train_real_rescaled.xlsx")
df_test_m = pd.read_excel("mee-type_test_real_rescaled.xlsx")
df_synth_m = pd.read_csv("mee-type_smote_maxabs_5000.csv", delimiter=",")

df_train_m["origin"] = "train"
df_test_m["origin"] = "test"
df_synth_m["origin"] = "synthetic"
df_all_m = pd.concat([df_train_m, df_test_m, df_synth_m], ignore_index=True)
features_m = [col for col in df_all_m.columns if col.startswith("feature_")]
mean_profiles_m = df_all_m.groupby(["e-type", "origin"])[features_m].mean().reset_index()
mean_profiles_m["row_id"] = mean_profiles_m["e-type"].astype(str) + " (" + mean_profiles_m["origin"] + ")"
mean_profiles_m.set_index("row_id", inplace=True)

# ----- 3. Load MAE matrices -----
mae_train_e = pd.read_csv('mae_matrix_synthetic_vs_train_etype.csv', index_col=0)
mae_test_e = pd.read_csv('mae_matrix_synthetic_vs_test_etype.csv', index_col=0)
mae_train_m = pd.read_csv('mae_matrix_synthetic_vs_train_meetype.csv', index_col=0)
mae_test_m = pd.read_csv('mae_matrix_synthetic_vs_test_meetype.csv', index_col=0)

# ----- 4. Plot 6-panel figure -----
fig, axs = plt.subplots(2, 3, figsize=(30, 16))

# Panel 1: e-type feature mean profiles
sns.heatmap(mean_profiles_e[features_e], cmap='viridis', ax=axs[0,0], cbar=True)
axs[0,0].set_title('A. e-type: Feature mean profiles', fontsize=22)
axs[0,0].set_ylabel('e-type - Train, Test, Synthetic', fontsize=16)
axs[0,0].set_xlabel('Feature', fontsize=16)

# Panel 2: e-type MAE (synthetic vs train)
sns.heatmap(mae_train_e, cmap='rocket', ax=axs[0,1], cbar=True)
axs[0,1].set_title('B. e-type: MAE (synthetic vs train)', fontsize=22)
axs[0,1].set_ylabel('e-type', fontsize=16)
axs[0,1].set_xlabel('Feature', fontsize=16)

# Panel 3: e-type MAE (synthetic vs test)
sns.heatmap(mae_test_e, cmap='rocket', ax=axs[0,2], cbar=True)
axs[0,2].set_title('C. e-type: MAE (synthetic vs test)', fontsize=22)
axs[0,2].set_ylabel('e-type', fontsize=16)
axs[0,2].set_xlabel('Feature', fontsize=16)

# Panel 4: mee-type feature mean profiles
sns.heatmap(mean_profiles_m[features_m], cmap='viridis', ax=axs[1,0], cbar=True)
axs[1,0].set_title('D. mee-type: Feature mean profiles', fontsize=22)
axs[1,0].set_ylabel('mee-type - Train, Test, Synthetic', fontsize=16)
axs[1,0].set_xlabel('Feature', fontsize=16)

# Panel 5: mee-type MAE (synthetic vs train)
sns.heatmap(mae_train_m, cmap='rocket', ax=axs[1,1], cbar=True)
axs[1,1].set_title('E. mee-type: MAE (synthetic vs train)', fontsize=22)
axs[1,1].set_ylabel('mee-type', fontsize=16)
axs[1,1].set_xlabel('Feature', fontsize=16)

# Panel 6: mee-type MAE (synthetic vs test)
sns.heatmap(mae_test_m, cmap='rocket', ax=axs[1,2], cbar=True)
axs[1,2].set_title('F. mee-type: MAE (synthetic vs test)', fontsize=22)
axs[1,2].set_ylabel('mee-type', fontsize=16)
axs[1,2].set_xlabel('Feature', fontsize=16)

plt.tight_layout()
plt.savefig('figure_six_panels_heatmap.png', dpi=350)
plt.show()
