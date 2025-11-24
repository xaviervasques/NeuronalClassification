# classify_on_augmented.py
import re
import ast
import glob
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

def log(msg):
    print(msg, flush=True)

# ----------------------------
# Column groups & ColumnTransformer (same logic as baseline)
# ----------------------------
def cols_endwith(frame, suffixes):
    cols = frame.columns if isinstance(frame, pd.DataFrame) else frame
    return [c for c in cols if any(c.endswith(sfx) for sfx in suffixes)]

def make_preprocessor(X):
    TIME_COLS = cols_endwith(X, ['_t_short_square','_t_long_square','_t_ramp'])
    VOLT_COLS = [c for c in X.columns if c.endswith('_v') or c in ['vrest','rheobase','max_firing_rate']]
    RATIO_COLS = [c for c in X.columns if c in ['adaptation','sag_ratio','upstroke_downstroke_ratio','ISI_CV','avg_isi','sag']]
    num_cols   = X.select_dtypes(include=[np.number]).columns.tolist()
    OTHER_COLS = [c for c in num_cols if c not in set(TIME_COLS + VOLT_COLS + RATIO_COLS)]

    log(f"[COLUMNS] TIME={len(TIME_COLS)}, VOLT={len(VOLT_COLS)}, RATIO={len(RATIO_COLS)}, OTHER={len(OTHER_COLS)}")
    if len(TIME_COLS)==0:
        log("[WARN] No *_t_* timing columns found. Proceeding without time group.")

    pre = ColumnTransformer(
        transformers=[
            ('time', RobustScaler(), TIME_COLS),
            ('volt', StandardScaler(), VOLT_COLS),
            ('ratio','passthrough', RATIO_COLS),
            ('other', StandardScaler(), OTHER_COLS)
        ],
        remainder='drop',
        n_jobs=None
    )
    return pre

# ----------------------------
# Models & grids (same as your baseline)
# ----------------------------
reducers = {
    'none': 'passthrough',
    'pca': PCA(n_components=2, random_state=42),
    'ica': FastICA(n_components=2, random_state=42, max_iter=1000),
    'truncatedsvd': TruncatedSVD(n_components=2, random_state=42),
    'isomap': Isomap(n_components=2, n_neighbors=10),
    'lle': LocallyLinearEmbedding(n_components=2, n_neighbors=10, method="standard", random_state=42),
    'nca': NeighborhoodComponentsAnalysis(n_components=2, init="pca", random_state=42)
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
                    random_state=42)
}

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
    'isomap':     {'reducer__n_neighbors':[5,10,15]},
    'lle':        {'reducer__n_neighbors':[5,10,15]},
    'nca':        {},
    'mlp': {
        'classifier__hidden_layer_sizes': [(256, 128)],
        'classifier__alpha': [1e-2],
        'classifier__learning_rate_init': [1e-3],
        'classifier__activation': ['tanh']
    }
}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1_macro': make_scorer(f1_score, average='macro'),
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
    'recall_macro': make_scorer(recall_score, average='macro', zero_division=0)
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ----------------------------
# Train/Eval on one augmented set
# ----------------------------
def run_one_augmented(train_csv, test_csv):
    # Parse method/size from filename e.g., aug_e_train_aug_smote_5000.csv
    m = re.search(r"train_aug_([a-zA-Z0-9]+)_(\d+)\.csv$", Path(train_csv).name)
    method = m.group(1) if m else "unknown"
    size   = m.group(2) if m else "NA"

    log(f"\n[RUN] Using TRAIN={train_csv} | TEST={test_csv} | method={method} | size={size}")

    # Load
    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)

    # Split features/labels (already curated & inverse-transformed)
    assert 'e-type' in df_tr.columns and 'e-type' in df_te.columns, "Expected 'e-type' in both CSVs."
    X_train = df_tr.drop(columns=['e-type'])
    y_train = df_tr['e-type'].astype(str)
    X_test  = df_te.drop(columns=['e-type'])
    y_test  = df_te['e-type'].astype(str)

    # Preprocessor
    preprocessor = make_preprocessor(X_train)

    # Grid over reducers & classifiers (same as baseline)
    full_results_csv_path = f'./results_aug_{method}_{size}.csv'
    Path(full_results_csv_path).write_text('')  # reset file
    log(f"[GRID] Results will be saved to: {full_results_csv_path}")

    rows = []
    total = len(reducers) * len(classifiers)
    k = 0
    t0_all = time()

    for reducer_name, reducer in reducers.items():
        for clf_name, clf in classifiers.items():
            k += 1
            log(f"[GRID] ({k}/{total}) reducer={reducer_name}, classifier={clf_name} …")
            try:
                pipe = Pipeline([
                    ('pre', preprocessor),
                    ('reducer', reducer),
                    ('classifier', clf)
                ])

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

                log(f"[OK] {reducer_name} + {clf_name} "
                    f"| best_acc={grid.best_score_:.3f} | time={dt:.1f}s | params={grid.best_params_}")

            except Exception as e:
                log(f"[ERR] {reducer_name} + {clf_name}: {e}")

    df_out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not df_out.empty:
        df_out.to_csv(full_results_csv_path, index=False)
        log(f"[SAVE] Wrote CV results: {full_results_csv_path} (rows={len(df_out)})")
    else:
        log("[WARN] No CV rows saved (all grids failed?).")

    # ----------------------------
    # Select best among reducer='none' and evaluate on real test
    # ----------------------------
    if df_out.empty:
        return

    log("[SELECT] Picking best pipeline among reducer='none' by mean_test_accuracy …")
    df_none = df_out[df_out['reducer'] == 'none'].sort_values('mean_test_accuracy', ascending=False)
    if df_none.empty:
        log("[WARN] No 'none' reducer rows; taking overall best instead.")
        df_none = df_out.sort_values('mean_test_accuracy', ascending=False)

    best = df_none.iloc[0]
    best_clf = best['classifier']
    best_params = ast.literal_eval(best['params']) if isinstance(best['params'], str) else best['params']
    log(f"[SELECT] Best classifier (no reducer preferred): {best_clf} with params: {best_params}")

    clf_obj = (classifiers[best_clf]
               .set_params(**{k.split('__',1)[1]: v for k,v in best_params.items() if k.startswith('classifier__')}))
    best_reducer = 'passthrough'
    pipe_best = Pipeline([('pre', preprocessor), ('reducer', best_reducer), ('classifier', clf_obj)])
    log("[FIT] Training best 'no reducer' pipeline on augmented train …")
    pipe_best.fit(X_train, y_train)

    y_pred = pipe_best.predict(X_test)
    test_acc  = accuracy_score(y_test, y_pred)
    test_f1   = f1_score(y_test, y_pred, average='macro')
    test_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    test_rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)

    log("\n=== Holdout test (real test set) ===")
    log(f"Accuracy: {test_acc:.3f} | F1_macro: {test_f1:.3f} | Precision_macro: {test_prec:.3f} | Recall_macro: {test_rec:.3f}")

    # Save compact metrics
    pd.DataFrame([{
        "method": method,
        "size": size,
        "reducer": "none",
        "classifier": best_clf,
        **{k: v for k, v in best_params.items() if k.startswith("classifier__")},
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "test_precision_macro": test_prec,
        "test_recall_macro": test_rec
    }]).to_csv(f"metrics_aug_{method}_{size}.csv", index=False)
    log(f"[SAVE] Wrote holdout metrics: metrics_aug_{method}_{size}.csv")

# ----------------------------
# Discover and run all augmented sets
# ----------------------------
if __name__ == "__main__":
    # You can switch between e-type and mee-type by changing the prefix here:
    PREFIX = "aug_e"   # or "aug_mee"

    train_glob = f"./{PREFIX}_train_aug_*.csv"
    test_file  = f"./{PREFIX}_test_real.csv"

    # safety checks
    if not Path(test_file).exists():
        raise FileNotFoundError(f"Could not find test file: {test_file}")

    train_files = sorted(glob.glob(train_glob))
    if not train_files:
        raise FileNotFoundError(f"No augmented train files found with pattern: {train_glob}")

    log(f"[FOUND] {len(train_files)} augmented train files.")
    log(f"[TEST ] Using real test: {test_file}")

    for tr in train_files:
        run_one_augmented(tr, test_file)

    log("\n[DONE] All augmented sets evaluated.")
