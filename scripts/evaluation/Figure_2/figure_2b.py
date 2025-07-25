import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
excel = pd.ExcelFile("results_figure_2.xlsx")
sheet_names = [
    ("e-type_without_reducer", "e-type (no reducer)"),
    ("e-type_with_reducer", "e-type (with reducer)"),
    ("mee-type_without_reducer", "mee-type (no reducer)"),
    ("mee-type_with_reducer", "mee-type (with reducer)"),
]

# Préparation du DataFrame pour plot groupé
all_scores = []
for sheet, label in sheet_names:
    df = excel.parse(sheet)
    for score in df['mean_test_score']:
        all_scores.append({"Condition": label, "Mean Test Accuracy": score})

df_panelB = pd.DataFrame(all_scores)

# Palette cohérente avec Panel A
palette = ['royalblue', 'navy', 'darkorange', 'orangered']

plt.figure(figsize=(10, 6))
sns.violinplot(
    x='Condition',
    y='Mean Test Accuracy',
    data=df_panelB,
    palette=palette,
    cut=0
)
plt.title("Panel B. Distribution of Mean Test Accuracy Across Pipelines", fontsize=15)
plt.ylabel("Mean Test Accuracy", fontsize=13)
plt.xlabel("")
plt.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.show()
