import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from time import time
import ast

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

print("\n[START] Baseline E→e-type pipeline\n")

# ----------------------------
# 0) Load data
# ----------------------------
file_path = '../data/e-type.csv'   # expects a column 'e-type' (Exc_1..Inh_13)
print(f"[LOAD] Reading: {file_path}")
data = pd.read_csv(file_path, delimiter=';')
print(f"[LOAD] Shape: {data.shape[0]} rows × {data.shape[1]} cols\n")

# ----------------------------
# 1) Feature curation: drop non-biological/technical vars
# ----------------------------
TECH_DROP = [
    # Electrophysiology (setup/quality; non-phenotypic)
    "electrode_0_pa",        # Feature_3
    "seal_gohm",             # Feature_21
    "vm_for_sag",            # if present in any export, drop

    # Morphology (pipeline/constant; non-phenotypic)
    "neuron_reconstruction_type",  # Feature_57
    "scale_factor_x",              # Feature_66
    "scale_factor_y",              # Feature_67
    "scale_factor_z",              # Feature_68
]
drop_cols = [c for c in TECH_DROP if c in data.columns]
if drop_cols:
    print(f"[CURATE] Dropping technical columns ({len(drop_cols)}): {drop_cols}")
    data = data.drop(columns=drop_cols)
else:
    print("[CURATE] No technical columns to drop.")
print(f"[CURATE] Remaining columns: {len(data.columns)}\n")

# ----------------------------
# 2) Split features/labels & basic NA handling
# ----------------------------
assert 'e-type' in data.columns, "Expected a column named 'e-type' for labels."
X = data.drop(columns=['e-type'])
y = data['e-type'].astype(str)

# Simple impute with column means for numeric
X = X.apply(pd.to_numeric, errors='ignore')
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
before_na = X[num_cols].isna().sum().sum()
X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
after_na = X[num_cols].isna().sum().sum()
print(f"[NA] Missing numeric values (before→after): {before_na} → {after_na}")

# Drop any constant numeric columns (safety net)
const_cols = [c for c in X.select_dtypes(include=[np.number]).columns if X[c].nunique(dropna=False) <= 1]
if const_cols:
    print(f"[CURATE] Dropping constant columns ({len(const_cols)}): {const_cols}")
    X = X.drop(columns=const_cols)
else:
    print("[CURATE] No constant numeric columns to drop.")
print()

# ----------------------------
# 3) Stimulus-relative re-referencing for time columns
# ----------------------------
# If you have exact onsets from metadata, replace these placeholders:
STIM_ONSETS = {"short_square": 1.0, "long_square": 1.0, "ramp": 1.0}
print(f"[TIME] Using stimulus onsets (s): {STIM_ONSETS}")

def re_reference_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_cols = 0
    for c in df.columns:
        if not (np.issubdtype(df[c].dtype, np.number)):
            continue
        if c.endswith('_t_short_square'):
            df[c] = df[c] - STIM_ONSETS['short_square']; n_cols += 1
        elif c.endswith('_t_long_square'):
            df[c] = df[c] - STIM_ONSETS['long_square']; n_cols += 1
        elif c.endswith('_t_ramp'):
            df[c] = df[c] - STIM_ONSETS['ramp']; n_cols += 1
    print(f"[TIME] Re-referenced {n_cols} timing columns to stimulus onset.\n")
    return df

X = re_reference_time(X)

# ----------------------------
# 4) Column groups & ColumnTransformer (family-specific scaling)
# ----------------------------
def cols_endwith(suffixes):
    return [c for c in X.columns if any(c.endswith(sfx) for sfx in suffixes)]

TIME_COLS = cols_endwith(['_t_short_square','_t_long_square','_t_ramp'])
VOLT_COLS = [c for c in X.columns if c.endswith('_v') or c in ['vrest','rheobase','max_firing_rate']]
RATIO_COLS = [c for c in X.columns if c in ['adaptation','sag_ratio','upstroke_downstroke_ratio','ISI_CV','avg_isi']]
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
OTHER_COLS = [c for c in num_cols if c not in set(TIME_COLS + VOLT_COLS + RATIO_COLS)]

print(f"[COLUMNS] TIME={len(TIME_COLS)}, VOLT={len(VOLT_COLS)}, RATIO={len(RATIO_COLS)}, OTHER={len(OTHER_COLS)}")
if len(TIME_COLS)==0:
    print("[WARN] No *_t_* timing columns found. Confirm feature names.\n")
else:
    print()

preprocessor = ColumnTransformer(
    transformers=[
        ('time', RobustScaler(), TIME_COLS),     # robust to outliers; preserves ms-scale deltas
        ('volt', StandardScaler(), VOLT_COLS),
        ('ratio','passthrough', RATIO_COLS),     # indices/ratios already scale-free
        ('other', StandardScaler(), OTHER_COLS)
    ],
    remainder='drop',
    n_jobs=None
)

# ----------------------------
# 5) Stratified 80/20 split with class coverage check
# ----------------------------
print("[SPLIT] Stratified 80/20 split by e-type …")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X, y))
X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

print(f"[SPLIT] Train: {X_train.shape}, Test: {X_test.shape}")
assert set(y_test.unique()) == set(y.unique()), "Some classes missing in test split."
print(f"[SPLIT] Class coverage OK. Unique classes: {sorted(y.unique())}\n")

# Save splits (parity/debug)
X_train.to_csv('X_train_e.csv', index=False)
X_test.to_csv('X_test_e.csv', index=False)
y_train.to_csv('y_train_e.csv', index=False)
y_test.to_csv('y_test_e.csv', index=False)
print("[SAVE] Wrote split CSVs: X_train_e.csv, X_test_e.csv, y_train_e.csv, y_test_e.csv\n")

# ----------------------------
# 6) Reducers (optional) and classifiers
# ----------------------------
reducers = {
    'none': 'passthrough',
    'pca': PCA(n_components=2, random_state=42),
    'ica': FastICA(n_components=2, random_state=42, max_iter=1000),
    'truncatedsvd': TruncatedSVD(n_components=2, random_state=42),
    'isomap': Isomap(n_components=2, n_neighbors=10),
    'lle': LocallyLinearEmbedding(n_components=2, n_neighbors=10, method="standard", random_state=42),
    'nca': NeighborhoodComponentsAnalysis(n_components=2, init="pca", random_state=42)
    # Note: TSNE omitted; no transform() for pipeline use
}

classifiers = {
    'svm_linear': SVC(kernel='linear', probability=False, random_state=42),
    'svm_rbf':    SVC(kernel='rbf', probability=False, random_state=42),
    'svm_sigmoid':SVC(kernel='sigmoid', probability=False, random_state=42),
    'logreg':     LogisticRegression(solver='sag', multi_class='auto', max_iter=1000, random_state=42),
    'rf':         RandomForestClassifier(random_state=42),
    'et':         ExtraTreesClassifier(random_state=42),
    'lda_cls':    LDA(),
    'gnb':        GaussianNB(),
    'dt':         DecisionTreeClassifier(random_state=42),
    'mlp':        MLPClassifier(
                    max_iter=2000,
                    #early_stopping=True,
                    #n_iter_no_change=20,
                    #validation_fraction=0.1,
                    random_state=42)
}

# Smaller, sane grids (extend as needed)
param_grid = {
    'svm_linear': {'classifier__C':[0.1,1,10]},
    'svm_rbf':    {'classifier__C':[0.1,1,10], 'classifier__gamma':['scale','auto']},
    'svm_sigmoid':{'classifier__C':[0.1,1,10], 'classifier__gamma':['scale','auto']},
    'logreg':     {'classifier__penalty':['l1','l2'], 'classifier__C':[0.1,1,10], 'classifier__solver': ['liblinear']},
    'rf':         {'classifier__n_estimators':[50, 100,200], 'classifier__max_depth':[None,10, 20],
                   'classifier__min_samples_split':[2,10, 20], 'classifier__min_samples_leaf':[1,5, 10]},
    'et':         {'classifier__n_estimators':[100, 200,300,500], 'classifier__max_depth':[None,10, 20],
                   'classifier__min_samples_split':[2,10, 20], 'classifier__min_samples_leaf':[1,5, 10]},
    'lda_cls':    {},
    'gnb':        {},
    'dt':         {'classifier__max_depth':[10,20,None], 'classifier__min_samples_split':[2,10, 20],
                   'classifier__min_samples_leaf':[1,5, 10]},
    # Reducer neighbor grids (only when reducer uses n_neighbors)
    'isomap':     {'reducer__n_neighbors':[5,10,15]},
    'lle':        {'reducer__n_neighbors':[5,10,15]},
    'nca':        {},
    'mlp': {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (128,), (256,), (128, 64), (256, 128)],
        'classifier__alpha': [1e-4, 1e-3, 1e-2],
        'classifier__learning_rate_init': [1e-3, 3e-4],
        'classifier__activation': ['relu', 'tanh']
    }
}

# Multi-metric scoring
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1_macro': make_scorer(f1_score, average='macro'),
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
    'recall_macro': make_scorer(recall_score, average='macro', zero_division=0)
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ----------------------------
# 7) Run grid over reducers & classifiers (preprocessor fixed)
# ----------------------------
full_results_csv_path = './e-type_baseline_results.csv'
Path(full_results_csv_path).write_text('')  # reset file
print(f"[GRID] Results will be saved to: {full_results_csv_path}\n")

rows = []
total = len(reducers) * len(classifiers)
k = 0
t0_all = time()

for reducer_name, reducer in reducers.items():
    for clf_name, clf in classifiers.items():
        k += 1
        print(f"[GRID] ({k}/{total}) reducer={reducer_name}, classifier={clf_name} …")
        try:
            pipe = Pipeline([
                ('pre', preprocessor),
                ('reducer', reducer),
                ('classifier', clf)
            ])
            # Merge grids relevant to this classifier/reducer
            grid_params = {}
            if clf_name in param_grid: grid_params.update(param_grid[clf_name])
            if reducer_name in param_grid: grid_params.update(param_grid[reducer_name])

            grid = GridSearchCV(
                estimator=pipe,
                param_grid=grid_params if grid_params else [{}],
                scoring=scoring,
                refit='accuracy',
                cv=cv,
                n_jobs=-1,
                return_train_score=True
            )
            t0 = time()
            grid.fit(X_train, y_train)
            dt = time() - t0

            res = pd.DataFrame(grid.cv_results_)
            keep = [
                'mean_fit_time','mean_score_time',
                'mean_test_accuracy','std_test_accuracy',
                'mean_test_f1_macro','mean_test_precision_macro','mean_test_recall_macro',
                'mean_train_accuracy','std_train_accuracy',
                'params','rank_test_accuracy'
            ]
            res = res[keep].copy()
            res['reducer'] = reducer_name
            res['classifier'] = clf_name
            rows.append(res)

            print(f"[OK] {reducer_name} + {clf_name} "
                  f"| best_acc={grid.best_score_:.3f} | time={dt:.1f}s | params={grid.best_params_}\n")

        except Exception as e:
            print(f"[ERR] {reducer_name} + {clf_name}: {e}\n")

df_out = pd.concat(rows, ignore_index=True)
df_out.to_csv(full_results_csv_path, index=False)
print(f"[SAVE] Wrote CV results: {full_results_csv_path} (rows={len(df_out)})\n")

# ----------------------------
# 8) Holdout evaluation for best baseline (no reducer preferred)
# ----------------------------
print("[SELECT] Picking best pipeline among reducer='none' by mean_test_accuracy …")
df_all = pd.read_csv(full_results_csv_path)
df_none = df_all[df_all['reducer'] == 'none'].sort_values('mean_test_accuracy', ascending=False)
best = df_none.iloc[0]
best_clf = best['classifier']
# safer than eval()
best_params = ast.literal_eval(best['params']) if isinstance(best['params'], str) else best['params']
print(f"[SELECT] Best classifier (no reducer): {best_clf} with params: {best_params}")

# Rebuild the exact best pipeline and fit on train, evaluate on test
best_reducer = 'passthrough'
clf_obj = (classifiers[best_clf]
           .set_params(**{k.split('__',1)[1]: v for k,v in best_params.items() if k.startswith('classifier__')}))
pipe_best = Pipeline([('pre', preprocessor), ('reducer', best_reducer), ('classifier', clf_obj)])
print("[FIT] Training best 'no reducer' pipeline on train set …")
pipe_best.fit(X_train, y_train)

y_pred = pipe_best.predict(X_test)
test_acc  = accuracy_score(y_test, y_pred)
test_f1   = f1_score(y_test, y_pred, average='macro')
test_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
test_rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)

print("\n=== Holdout test (no reducer baseline) ===")
print(f"Accuracy: {test_acc:.3f} | F1_macro: {test_f1:.3f} | "
      f"Precision_macro: {test_prec:.3f} | Recall_macro: {test_rec:.3f}\n")

# Save a compact test-metrics CSV too (handy)
pd.DataFrame([{
    "reducer": "none",
    "classifier": best_clf,
    **{k: v for k, v in best_params.items() if k.startswith("classifier__")},
    "test_accuracy": test_acc,
    "test_f1_macro": test_f1,
    "test_precision_macro": test_prec,
    "test_recall_macro": test_rec
}]).to_csv("e-type_baseline_test_metrics.csv", index=False)
print("[SAVE] Wrote holdout metrics: e-type_baseline_test_metrics.csv")

# ----------------------------
# 9) Diagnostics: class counts and basic variance guard
# ----------------------------
train_counts = y_train.value_counts().sort_index()
test_counts  = y_test.value_counts().sort_index()
pd.concat([train_counts.rename('train'), test_counts.rename('test')], axis=1)\
  .to_csv('e-type_class_counts.csv')
print("[SAVE] Wrote class counts: e-type_class_counts.csv")

# Variance check after transform (fit on train only)
Xtr_trans = preprocessor.fit_transform(X_train)
vars_ = np.var(Xtr_trans, axis=0)
n_bad = int((vars_ <= 1e-8).sum())
if n_bad > 0:
    raise ValueError(f"{n_bad} features have near-zero variance after preprocessing. "
                     f"Check timing re-reference and scalers.")
else:
    print("[QA] Variance check passed: no near-zero-variance features after preprocessing.")

print(f"\n[DONE] Total grid combos: {total} | Total CV rows: {len(df_out)} | "
      f"Elapsed: {time()-t0_all:.1f}s\n")
