# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, PowerTransformer, MaxAbsScaler, QuantileTransformer, LabelEncoder
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, TSNE
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

# Load the data
file_path = './data/mee-type.csv'
data = pd.read_csv(file_path, delimiter=';')

# Label encode the neuron_reconstruction_type column
label_encoder = LabelEncoder()
data['neuron_reconstruction_type'] = label_encoder.fit_transform(data['neuron_reconstruction_type'])
data['superseded'] = label_encoder.fit_transform(data['superseded'])

# Split the data into features and target
X = data.drop(columns=['e-type'])
y = data['e-type']

# Fill missing values with the mean of each column
X = X.fillna(X.mean())

# List of scalers and classifiers
scalers = {
    'minmax': MinMaxScaler(),
    'standard': StandardScaler(),
    'robust': RobustScaler(),
    'maxabs': MaxAbsScaler(),
    'normalizer': Normalizer(),
    'box-cox': PowerTransformer(method="box-cox"),
    'yeo-johnson': PowerTransformer(method="yeo-johnson"),
    'quantile-normal': QuantileTransformer(n_quantiles=1000, output_distribution="normal"),
    'quantile-uniform': QuantileTransformer(n_quantiles=1000, output_distribution="uniform")
}

classifiers = {
    'svm_linear': SVC(kernel='linear'),
    'svm_rbf': SVC(kernel='rbf'),
    'svm_sigmoid': SVC(kernel='sigmoid'),
    'svm_poly': SVC(kernel='poly'),
    'logistic_regression': LogisticRegression(solver='sag', multi_class='auto'),
    'random_forest': RandomForestClassifier(),
    'lda': LDA(),
    'gaussian_nb': GaussianNB(),
    'multinomial_nb': MultinomialNB(),
    'decision_tree': DecisionTreeClassifier(),
    'extra_trees': ExtraTreesClassifier(),
    'mlp': MLPClassifier(max_iter=1000)
}

param_grid = {
    'svm_linear': {'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 'auto']},
    'svm_rbf': {'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 'auto']},
    'svm_sigmoid': {'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 'auto']},
    'svm_poly': {'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 'auto']},
    'random_forest': {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [None, 10, 20], 'classifier__min_samples_split': [2, 10, 20], 'classifier__min_samples_leaf': [1, 5, 10]},
    'extra_trees': {'classifier__n_estimators': [100, 200, 300, 500], 'classifier__max_depth': [None, 10, 20], 'classifier__min_samples_split': [2, 10, 20],'classifier__min_samples_leaf': [1, 5, 10]},
    'lda': {},
    'logistic_regression': {
        'classifier__penalty': ['l1', 'l2'],
        'classifier__C': [0.1, 1, 10],
        'classifier__solver': ['liblinear']
    },
    'gaussian_nb': {},
    'multinomial_nb': {'classifier__alpha': [0.1, 1, 10]},
    'decision_tree': {
        'classifier__max_depth': [5, 10, 20],
        'classifier__min_samples_split': [2, 10, 20],
        'classifier__min_samples_leaf': [1, 5, 10]
    },
    'mlp': {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'classifier__alpha': [0.0001, 0.001, 0.01]
    },
    'lle': {'reducer__n_neighbors': [5, 10, 15]},
    'modified_lle': {'reducer__n_neighbors': [5, 10, 15]},
    'hessian_lle': {'reducer__n_neighbors': [5, 10, 15]},
    'ltsa_lle': {'reducer__n_neighbors': [5, 10, 15]},
    'isomap': {'reducer__n_neighbors': [5, 10, 15]}
}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the training and testing sets to CSV files
X_train.to_csv('X_train_mee.csv', index=False)
X_test.to_csv('X_test_mee.csv', index=False)
y_train.to_csv('y_train_mee.csv', index=False)
y_test.to_csv('y_test_mee.csv', index=False)

# Initialize CSV file for results
full_results_csv_path = './mee-type_full_classification_results.csv'
with open(full_results_csv_path, 'w') as f:
    # Write the header
    f.write('mean_fit_time,mean_score_time,mean_test_score,std_test_score,mean_train_score,std_train_score,params,rank_test_score,scaler,reducer,classifier\n')

"""

Without Reducer

"""

# Loop through scalers, reducers, and classifiers to create pipelines and perform grid search
for scaler_name, scaler in scalers.items():
    for classifier_name, classifier in classifiers.items():
            
        pipe = Pipeline([
            ('scaler', scaler),
            ('classifier', classifier)
        ])
        grid = GridSearchCV(pipe, param_grid[classifier_name], cv=10, scoring='accuracy', n_jobs=-1, return_train_score=True)
        grid.fit(X_train, y_train)
    
        # Get relevant results from GridSearchCV
        relevant_columns = ['mean_fit_time', 'mean_score_time', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score', 'params', 'rank_test_score']
        all_results = pd.DataFrame(grid.cv_results_)[relevant_columns]
    
        # Add additional columns for scaler, reducer, and classifier names
        all_results['scaler'] = scaler_name
        all_results['classifier'] = classifier_name
        
        # Save results to CSV incrementally
        all_results.to_csv(full_results_csv_path, mode='a', header=False, index=False)
        
        # Print results incrementally
        print(f"Results after {scaler_name}, {classifier_name}:")
        print(all_results.head())
