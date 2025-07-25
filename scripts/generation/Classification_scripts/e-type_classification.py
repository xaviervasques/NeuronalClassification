# Imports
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import PowerTransformer, MaxAbsScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, brier_score_loss
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from scipy.stats import t
import csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings('ignore')

# === Utility ===
def compute_confidence_interval(scores, confidence=0.95):
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    n = len(scores)
    h = std * t.ppf((1 + confidence) / 2., n - 1) / np.sqrt(n)
    return mean, std, mean - h, mean + h

# === CONFIGURATION ===
file_path = '../Generation_scripts_with_reducer_scaling_before_sampling/'
type_path = './data/e-type_synthetic/'
file_list = ['e-type_ddpm_maxabs_lda_35.csv', 'e-type_flow_maxabs_lda_100.csv', 'e-type_flow_maxabs_lda_1000.csv', 'e-type_flow_maxabs_lda_10000.csv', 'e-type_gan_maxabs_lda_100.csv', 'e-type_gan_maxabs_lda_1000.csv', 'e-type_gan_maxabs_lda_10000.csv', 'e-type_smote_maxabs_lda_1000.csv', 'e-type_smote_maxabs_lda_2000.csv', 'e-type_smote_maxabs_lda_5000.csv', 'e-type_smote_maxabs_lda_10000.csv', 'e-type_vae_maxabs_lda_35.csv']

X_test = pd.read_csv(file_path + 'X_test_e.csv')
y_test = pd.read_csv(file_path + 'y_test_e.csv').values.ravel()

maxabs = MaxAbsScaler()
X_test_scaled = maxabs.fit(X_test).transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=[f"feature_{i+1}" for i in range(X_test.shape[1])])
lda = LinearDiscriminantAnalysis(n_components=2)
X_test_scaled = lda.fit(X_test_scaled, y_test).transform(X_test_scaled)
X_test_scaled = pd.DataFrame(X_test_scaled)

classifiers = {

    # === Logistic Regression (standard & balanced, l2/l1, différents C) ===
    'logistic_l2_1': LogisticRegression(C=1.0, penalty='l2', solver='saga', max_iter=1000),
    'logistic_l2_2': LogisticRegression(C=0.1, penalty='l2', solver='saga', max_iter=1000),
    'logistic_l2_3': LogisticRegression(C=10, penalty='l2', solver='saga', max_iter=1000),
    'logistic_l1_1': LogisticRegression(C=1.0, penalty='l1', solver='saga', max_iter=1000),
    'logistic_l1_2': LogisticRegression(C=0.1, penalty='l1', solver='saga', max_iter=1000),
    'logistic_l1_3': LogisticRegression(C=10, penalty='l1', solver='saga', max_iter=1000),

    'logistic_l2_balanced_1': LogisticRegression(C=1.0, penalty='l2', solver='saga', class_weight='balanced', max_iter=1000),
    'logistic_l1_balanced_1': LogisticRegression(C=1.0, penalty='l1', solver='saga', class_weight='balanced', max_iter=1000),

    # === Random Forest (depth, n_estimators, balanced) ===
    'rf_100_10': RandomForestClassifier(n_estimators=100, max_depth=10),
    'rf_100_20': RandomForestClassifier(n_estimators=100, max_depth=20),
    'rf_200_10': RandomForestClassifier(n_estimators=200, max_depth=10),
    'rf_200_20': RandomForestClassifier(n_estimators=200, max_depth=20),

    'rf_bal_100_10': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced'),
    'rf_bal_200_20': RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced_subsample'),

    # === Decision Trees
    'decision_tree': DecisionTreeClassifier(),
    'decision_tree_bal': DecisionTreeClassifier(class_weight='balanced'),

    # === Extra Trees
    'extra_tree': ExtraTreesClassifier(),
    'extra_tree_bal': ExtraTreesClassifier(class_weight='balanced'),

    # === LDA
    'lda': LinearDiscriminantAnalysis(),

    # === Naive Bayes
    'gaussian_nb': GaussianNB(),

    # === SVM - RBF, SIGMOID, POLY, BALANCED ===
    'svm_rbf_1': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
    'svm_rbf_2': SVC(kernel='rbf', C=10, gamma='scale', probability=True),
    'svm_rbf_3': SVC(kernel='rbf', C=0.1, gamma='scale', probability=True),
    'svm_rbf_auto': SVC(kernel='rbf', C=1.0, gamma='auto', probability=True),

    'svm_rbf_bal': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True),
    #'svm_linear_bal': SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True),

    'svm_sigmoid': SVC(kernel='sigmoid', C=1.0, gamma='scale', probability=True),
    'svm_sigmoid_bal': SVC(kernel='sigmoid', C=1.0, gamma='scale', class_weight='balanced', probability=True),

    'svm_poly_3': SVC(kernel='poly', degree=3, C=1.0, gamma='scale', probability=True),
    'svm_poly_4': SVC(kernel='poly', degree=4, C=1.0, gamma='scale', probability=True),

    # === MLP Classifiers ===
    'mlp_50_lr0.001': MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.001, max_iter=300),
    'mlp_100_lr0.001': MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=300),
    'mlp_50x2_lr0.001': MLPClassifier(hidden_layer_sizes=(50, 50), learning_rate_init=0.001, max_iter=300),
    'mlp_128_64': MLPClassifier(hidden_layer_sizes=(128, 64), learning_rate_init=0.001, max_iter=300),
    'mlp_128_64_es': MLPClassifier(hidden_layer_sizes=(128, 64), learning_rate_init=0.001, max_iter=300, early_stopping=True),
    'mlp_128_64_alpha': MLPClassifier(hidden_layer_sizes=(128, 64), learning_rate_init=0.001, max_iter=300, alpha=0.0001),
}

temp_output_path = '../e-type_summary_temp.csv'

# Write header if the file does not exist
if not os.path.exists(temp_output_path):
    with open(temp_output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'file', 'classifier', 'mean_train_accuracy', 'mean_train_std',
            'mean_test_accuracy', 'test_std', 'test_ci_low', 'test_ci_high',
            'weighted_precision', 'prec_std', 'prec_ci_low', 'prec_ci_high',
            'brier_score'
        ])


for file in file_list:
    print(f"\n--- Processing: {file} ---")
    synthetic_data = pd.read_csv(file_path + type_path + file).dropna()
    X_merged = synthetic_data.drop(columns=['e-type'])
    y_merged = synthetic_data['e-type']

    le = LabelEncoder()
    y_merged_encoded = le.fit_transform(y_merged)

    train_labels = set(np.unique(y_merged_encoded))

    y_test_encoded = []
    valid_test_indices = []
    for i, label in enumerate(y_test):
        if label in le.classes_:
            encoded = le.transform([label])[0]
            if encoded in train_labels:
                y_test_encoded.append(encoded)
                valid_test_indices.append(i)

    X_test_filtered = X_test_scaled.iloc[valid_test_indices]
    y_test_filtered = np.array(y_test_encoded)


    for clf_name, clf in classifiers.items():
        print(f"Training: {clf_name}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc_scores = []
        for train_idx, val_idx in skf.split(X_merged, y_merged_encoded):
            X_train_cv, X_val_cv = X_merged.iloc[train_idx], X_merged.iloc[val_idx]
            y_train_cv, y_val_cv = y_merged_encoded[train_idx], y_merged_encoded[val_idx]
            model = clf.fit(X_train_cv, pd.Series(y_train_cv))
            y_val_pred = model.predict(X_val_cv)
            acc_scores.append(accuracy_score(y_val_cv, y_val_pred))

        mean_cv, std_cv = np.mean(acc_scores), np.std(acc_scores)

        # Repeated test evaluation
        test_accs, test_precs, test_briers = [], [], []
        for seed in range(5):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            train_idx, _ = next(iter(skf.split(X_merged, y_merged_encoded)))
            X_train_seed = X_merged.iloc[train_idx]
            y_train_seed = y_merged_encoded[train_idx]

            clf.fit(X_train_seed, pd.Series(y_train_seed))
            y_pred = clf.predict(X_test_filtered)
            test_accs.append(accuracy_score(y_test_filtered, y_pred))
            test_precs.append(precision_score(y_test_filtered, y_pred, average='weighted'))

            if hasattr(clf, "predict_proba"):
                try:
                    y_proba = clf.predict_proba(X_test_filtered)
                    brier = np.mean([
                        brier_score_loss((y_test_filtered == i).astype(int), y_proba[:, i])
                        for i in range(y_proba.shape[1])
                    ])
                    test_briers.append(brier)
                except:
                    test_briers.append(np.nan)
            else:
                test_briers.append(np.nan)

        acc_mean, acc_std, acc_ci_low, acc_ci_high = compute_confidence_interval(test_accs)
        prec_mean, prec_std, prec_ci_low, prec_ci_high = compute_confidence_interval(test_precs)
        brier_mean = np.nanmean(test_briers)

        # Save to temporary CSV immediately
        with open(temp_output_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                file, clf_name, mean_cv, std_cv,
                acc_mean, acc_std, acc_ci_low, acc_ci_high,
                prec_mean, prec_std, prec_ci_low, prec_ci_high,
                brier_mean
            ])

# Optional: compile final version
summary_df = pd.read_csv(temp_output_path)
summary_df.to_csv('../e-type_summary_results.csv', index=False)
print("\nFinal summary saved to summary_gpu_results.csv")
