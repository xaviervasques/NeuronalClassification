import pandas as pd
import matplotlib.pyplot as plt

# --- CHARGEMENT DES DONNÉES ---
excel = pd.ExcelFile("results_figure_2.xlsx")

# Configuration : couleurs, symboles, labels pour chaque condition
config = [
    ("e-type_without_reducer", 'royalblue', 'o', 'e-type (no reducer)'),
    ("e-type_with_reducer", 'navy', 's', 'e-type (with reducer)'),
    ("mee-type_without_reducer", 'darkorange', 'o', 'mee-type (no reducer)'),
    ("mee-type_with_reducer", 'orangered', 's', 'mee-type (with reducer)'),
]

plt.figure(figsize=(9, 7))

# --- SCATTERPLOT : UN GROUPE PAR CONDITION ---
for sheet, color, marker, label in config:
    df = excel.parse(sheet)
    plt.scatter(
        df['mean_test_score'], df['consistency'],
        label=label, alpha=0.7, c=color, marker=marker,
        edgecolor='k', s=54
    )

# --- LIGNES SEUILS ---
plt.axhline(0.03, color='green', linestyle='--', linewidth=2, label='Generalization threshold (0.03)')
plt.axhline(0.07, color='red', linestyle=':', linewidth=2, label='Max allowed gap (0.07)')

# --- FORMATTAGE FIGURE ---
plt.xlabel('Mean Test Accuracy', fontsize=14)
plt.ylabel('Train-Test Gap (Consistency)', fontsize=14)
plt.title('Panel A. Performance Landscape of All Pipelines\n(Mouse Visual Cortex Neuron Classification)', fontsize=15)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()
