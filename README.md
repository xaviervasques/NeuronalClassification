# NeuronalClassification

Benchmark of synthetic data augmentation methods for neuronal e-type prediction using the Allen Cell Types dataset.

This repository contains:
- Baseline supervised classification pipelines for two tasks:
  - **E→e-type**: predict Allen electrophysiology-defined e-types from intrinsic electrophysiological features only.
  - **M+E→e-type**: predict the **same e-types** from concatenated morphology + electrophysiology features.
- Synthetic data generation (SMOTE, VAE, GAN, masked autoregressive flow, DDPM).
- Evaluation of classifiers trained on augmented data.

## 1. Installation

### 1.1 Create environment
We recommend Python **3.9–3.11**.

```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install --upgrade pip
```

### 1.2 Install dependencies

```bash
pip install -r requirements.txt
```

**GPU note (optional):**
- On Google Colab or Linux/CUDA, TensorFlow will automatically use the GPU if available.
- If you use a local NVIDIA setup, install a GPU-enabled TensorFlow build compatible with your CUDA drivers.

## 2. Data

The scripts expect curated CSV files in `data/`.

Included example files (curated feature matrices and splits):
- `data/e-type.csv` : electrophysiology feature table with e-type labels.
- `data/mee-type.csv` : joint morphology + electrophysiology feature table with the same labels.
- `data/X_train_e.csv`, `y_train_e.csv`, `X_test_e.csv`, `y_test_e.csv`
- `data/X_train_mee.csv`, `y_train_mee.csv`, `X_test_mee.csv`, `y_test_mee.csv`

If you want to regenerate these from the raw Allen Cell Types release, follow the curation steps described in the manuscript:
dropping technical variables, re-referencing *_t_* features to stimulus onset, and applying family-specific scaling.

## 3. Baseline classification (real data)

### 3.1 E→e-type (electrophysiology-only)

```bash
cd classification
python classification_E.py
```

### 3.2 M+E→e-type (morphology + electrophysiology)

```bash
cd classification
python "classification_M+E.py"
```

These scripts:
- load the corresponding dataset,
- run the full pipeline grid-search / cross-validation,
- save per-pipeline scores and selected baselines to the `results/` folder (created automatically).

## 4. Synthetic data generation

Two generator entry points are provided in `generation/`.

### 4.1 CPU / standard run

Edit the configuration block at the bottom of `generation/generation.py`
(e.g., choose `INPUT_CSV="../data/e-type.csv"` or `../data/mee-type.csv`,
set `SYN_PER_CLASS_GRID`, epochs, etc.), then run:

```bash
cd generation
python generation.py
```

This produces, for each method and augmentation level:
- augmented training sets (real + synthetic),
- untouched real test sets,
- one JSON config per run.

Outputs are written to `generation/augmented_outputs/` by default.

### 4.2 GPU-optimized run

For CUDA/Colab runs, use:

```bash
cd generation
python generation_gpu_enabled.py
```

This version enables mixed precision on GPU and disables XLA to avoid common Colab/CUDA issues.

## 5. Classification on augmented data

After generation, evaluate all augmented training sets with:

```bash
cd generation
python classify_on_synthetic.py
```

This script:
- detects all augmented datasets produced by `generation.py`,
- retrains the selected baseline pipelines,
- reports hold-out accuracy and macro metrics for each method × augmentation level.

Results are saved as CSV files in `generation/augmented_results/`.

## 6. Reproducibility

- Random seed is set in each script (default: `SEED=42`).
- All augmentation is applied **only to the training set**.
- The augmentation schedule is class-conditional and shared across methods:
  `{0, 100, 500, 1000, 5000, 10000}` synthetic samples per class.

## 7. Citation

If you use this code for research, please cite the associated manuscript and the Allen Cell Types dataset papers (Gouwens et al., 2019, 2020).

---

Questions or issues: please open a GitHub issue on the repository.
