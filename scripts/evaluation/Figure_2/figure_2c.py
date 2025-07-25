import pandas as pd

# Charger l'excel
excel = pd.ExcelFile("results_figure_2.xlsx")
sheet_names = [
    ("e-type_without_reducer", "e-type (no reducer)"),
    ("e-type_with_reducer", "e-type (with reducer)"),
    ("mee-type_without_reducer", "mee-type (no reducer)"),
    ("mee-type_with_reducer", "mee-type (with reducer)"),
]

def top_pipelines(df, n=3):
    # Tri par test score décroissant et gap croissant
    return df.sort_values(['mean_test_score', 'consistency'], ascending=[False, True]).head(n)

# Extraction et mise en forme des top pipelines
top_tables = {}
for sheet, label in sheet_names:
    df = excel.parse(sheet)
    # Sélection des colonnes importantes
    cols = ['scaler']
    if 'reducer' in df.columns:
        cols.append('reducer')
    cols += ['classifier', 'mean_test_score', 'std_test_score', 'mean_train_score', 'consistency']
    top3 = top_pipelines(df)
    top_tables[label] = top3[cols].reset_index(drop=True)

# Export CSV pour chaque condition (optionnel)
for label, table in top_tables.items():
    fname = label.replace(' ', '_').replace('(', '').replace(')', '') + '_top3.csv'
    table.to_csv(fname, index=False)
    print(f"Saved {fname}")

# Affichage Markdown pour manuscrit
for label, table in top_tables.items():
    print(f"\n### {label}\n")
    print(table.to_markdown(index=False, floatfmt=".3f"))
