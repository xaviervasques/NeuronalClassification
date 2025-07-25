# Synthetic Data Generation for Neuronal Classification

This repository contains the full pipeline for reproducing the results in the manuscript:

**"Synthetic Data Generation for Classifying Electrophysiological and Morpho-Electrophysiological Neurons from Mouse Visual Cortex"**

## 📁 Repository Structure

- `scripts/`: Python scripts for classification, generation, and evaluation of synthetic data
- `data/`: Raw CSV data used in classification benchmarks
- `notebooks/`: Jupyter notebooks for post-analysis and visualization
- `figures/`: Final figures used in the paper
- `docs/`: Manuscript and supplementary materials

## 📦 Setup

Install required packages:

```bash
pip install -r requirements.txt
# or for conda
conda env create -f environment.yml
```

## 🧠 Neuronal Data

The datasets (e-type and mee-type neurons) are derived from the Allen Cell Types Database and preprocessed for use in this pipeline.

## 🚀 Reproduce Classification Benchmarks

```bash
cd scripts/classification/
python classification_etype_without_reducer.py
python classification_meetype_with_reducer.py
```

## 🔄 Generate Synthetic Neurons

```bash
cd scripts/generation/Generation_scripts_scaling_before_sampling/
python generation_etype.py  # example
```

## 📊 Evaluate Synthetic Data

```bash
cd scripts/evaluation/
python heatmap_meetype.py  # or any other evaluation script
```

## 📜 License

MIT License

## 📬 Contact

For questions or collaborations: [xaviervasques@institutduneurone.fr](mailto:xaviervasques@institutduneurone.fr)
