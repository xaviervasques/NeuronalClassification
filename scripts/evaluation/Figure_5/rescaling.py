import pandas as pd
from sklearn.preprocessing import PowerTransformer, MaxAbsScaler

# Charger le fichier Excel
df = pd.read_excel("./mee-type_train_real.xlsx")

# Sélectionner uniquement les colonnes numériques pour le rescaling
num_cols = df.select_dtypes(include=['number']).columns

# Initialiser et appliquer la transformation Yeo-Johnson
#scaler = PowerTransformer(method='yeo-johnson')
scaler = MaxAbsScaler()
df_rescaled = df.copy()
df_rescaled[num_cols] = scaler.fit_transform(df[num_cols])

# Sauvegarder dans un nouveau fichier Excel
df_rescaled.to_excel("./mee-type_train_real_rescaled.xlsx", index=False)
