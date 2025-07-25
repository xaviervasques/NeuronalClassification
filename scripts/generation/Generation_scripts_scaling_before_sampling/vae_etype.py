import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, PowerTransformer, MaxAbsScaler, QuantileTransformer
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Libraries for VAE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

# Check GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Load the data
file_path = './data/e-type.csv'
data = pd.read_csv(file_path, delimiter=';')

# Split the data into features and target
X = data.drop(columns=['e-type'])
y = data['e-type']

# Fill missing values with the mean of each column
X = X.fillna(X.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define function for VAE data generation
def generate_vae_data(X, n_samples):
    n_features = X.shape[1]
    latent_dim = 2
    
    # Encoder
    inputs = Input(shape=(n_features,))
    h = Dense(64, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    # Decoder
    decoder_h = Dense(64, activation='relu')
    decoder_mean = Dense(n_features, activation='linear')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    
    # VAE model
    vae = Model(inputs, x_decoded_mean)
    
    # Compute VAE loss
    xent_loss = binary_crossentropy(inputs, x_decoded_mean)
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam(learning_rate=0.001))
    
    # Train VAE
    vae.fit(X, X, epochs=100, batch_size=32, verbose=0)
    
    # Generate new data
    z_samples = np.random.normal(size=(n_samples, latent_dim))
    synthetic_data = vae.predict(z_samples)
    
    return synthetic_data

# Data augmentation function
def augment_data(X, y, n_samples, method):
    X_resampled, y_resampled = [], []
    unique_classes = np.unique(y)
    for cls in unique_classes:
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        if len(X_cls) >= n_samples:
            X_resampled.append(X_cls[:n_samples])
            y_resampled.append(y_cls[:n_samples])
        else:
            n_generate = n_samples - len(X_cls)
            print(f"class {cls}")
            if method == 'vae':
                try:
                    X_syn = generate_vae_data(X_cls, n_generate)
                    y_syn = np.array([cls] * n_generate)
                except Exception as e:
                    print(f"Error with VAE for class {cls}: {e}")
                    continue
            else:
                raise ValueError("Unknown method: {}".format(method))

            # Combine original and synthetic data
            X_resampled.append(np.vstack([X_cls, X_syn]))
            y_resampled.append(np.hstack([y_cls, y_syn]))

    # Aggregate all resampled classes into one dataset
    if X_resampled:
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
    else:
        raise ValueError(f"All resampling attempts failed for method {method}.")

    return X_resampled, y_resampled

# Augment data to have a balanced number of samples per class using various methods
methods = ['vae']
sizes = [35]  # Example sizes to augment data to

"""
Without reducer
"""

for method in methods:
    for size in sizes:
        print(f"Augmenting data using method {method} with size {size}")
        scaler_1 = PowerTransformer(method="yeo-johnson")
        X_scaled_1 = scaler_1.fit_transform(X_train)
        X_resampled, y_resampled = augment_data(X_scaled_1, y_train, size, method=method)
        data_resampled = pd.DataFrame(X_resampled, columns=[f'feature_{i+1}' for i in range(X_resampled.shape[1])])
        data_resampled['e-type'] = y_resampled
        data_resampled.to_csv(f'./e-type_synthetic/e-type_{method}_yeo_{size}.csv', index=False)
        print(f"Saved dataset for {method} with {size} samples per class.")
        
