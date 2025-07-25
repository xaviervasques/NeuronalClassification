# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, PowerTransformer, MaxAbsScaler, QuantileTransformer
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Libraries for Flow, DDPM, and GANs
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input, Lambda, Layer
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

# Save the training and testing sets to CSV files
X_train.to_csv('X_train_e.csv', index=False)
X_test.to_csv('X_test_e.csv', index=False)
y_train.to_csv('y_train_e.csv', index=False)
y_test.to_csv('y_test_e.csv', index=False)

# Define function for GAN data generation
def generate_gan_data(X, n_samples):
    n_features = X.shape[1]
    adam = Adam(learning_rate=0.0002, beta_1=0.5)
    
    # Generator
    generator = Sequential()
    generator.add(Dense(32, input_dim=n_features))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Dense(n_features, activation='linear'))
    generator.compile(loss='binary_crossentropy', optimizer=adam)
    
    # Discriminator
    discriminator = Sequential()
    discriminator.add(Dense(32, input_dim=n_features))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    # GAN
    discriminator.trainable = False
    gan = Sequential([generator, discriminator])
    gan.compile(loss='binary_crossentropy', optimizer=adam)
    
    # Training the GAN
    for epoch in range(300):
        idx = np.random.randint(0, X.shape[0], n_samples)
        real_data = X[idx]
        noise = np.random.normal(0, 1, (n_samples, n_features))
        synthetic_data = generator.predict(noise)
        
        real_labels = np.ones((n_samples, 1))
        fake_labels = np.zeros((n_samples, 1))
        
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(synthetic_data, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        noise = np.random.normal(0, 1, (n_samples, n_features))
        valid_y = np.array([1] * n_samples)
        g_loss = gan.train_on_batch(noise, valid_y)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")
    
    noise = np.random.normal(0, 1, (n_samples, n_features))
    synthetic_data = generator.predict(noise)
    
    return synthetic_data
    
# Define function for Flow data generation
def generate_flow_data(X, n_samples):
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors

    n_features = X.shape[1]
    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([n_features]))
    flow = tfb.Chain([tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=tfb.AutoregressiveNetwork(params=2)) for _ in range(4)])
    transformed_dist = tfd.TransformedDistribution(distribution=base_dist, bijector=flow)

    synthetic_data = transformed_dist.sample(n_samples).numpy()
    
    return synthetic_data
    
# Define function for DDPM data generation
def generate_ddpm_data(X, n_samples):
    from diffusers import DDPMPipeline, DDPMScheduler
    import torch

    n_features = X.shape[1]
    
    # Placeholder model: In practice, you should train your own DDPM model
    model_id = "google/ddpm-cifar10-32"
    ddpm = DDPMPipeline.from_pretrained(model_id)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    synthetic_data = []
    for _ in range(n_samples):
        noise = torch.randn(1, n_features)
        generated_data = ddpm(noise, scheduler=scheduler)["sample"]
        synthetic_data.append(generated_data.numpy())
    
    synthetic_data = np.vstack(synthetic_data)
    
    return synthetic_data

# Data augmentation function
def augment_data(X, y, n_samples, method):
    if method == 'smote':
        try:
            if method == 'smote':
                print(f"Applying SMOTE to generate {n_samples} samples per class.")
                print(f"X shape: {X.shape}, y shape: {y.shape}")
                print(f"X type: {type(X)}, y type: {type(y)}")
                #smote = SMOTE(random_state=42, n_jobs=-1)
                smote = SMOTE(sampling_strategy={cls: n_samples for cls in np.unique(y)}, random_state=42, n_jobs=-1)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                print(f"SMOTE successful. Resampled X shape: {X_resampled.shape}, Resampled y shape: {y_resampled.shape}")
                return X_resampled, y_resampled  # Add this return statement
        except Exception as e:
            print(f"Error with {method}: {e}")
            raise

    else:
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
                if method == 'gan':
                    try:
                        X_syn = generate_gan_data(X_cls, n_generate)
                        y_syn = np.array([cls] * n_generate)
                    except Exception as e:
                        print(f"Error with GAN for class {cls}: {e}")
                        continue
                elif method == 'flow':
                    try:
                        X_syn = generate_flow_data(X_cls, n_generate)
                        y_syn = np.array([cls] * n_generate)
                    except Exception as e:
                        print(f"Error with Flow for class {cls}: {e}")
                        continue
                elif method == 'ddpm':
                    try:
                        X_syn = generate_ddpm_data(X_cls, n_generate)
                        y_syn = np.array([cls] * n_generate)
                    except Exception as e:
                        print(f"Error with DDPM for class {cls}: {e}")
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


"""
Without reducer
"""

methods = ['smote']
sizes = [1000, 2000, 5000, 10000]  # Example sizes to augment data to

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
        
methods = ['ddpm']
sizes = [35]  # Example sizes to augment data to

for method in methods:
    for size in sizes:
        print(f"Augmenting data using method {method} with size {size}")
        scaler_4 = PowerTransformer(method="yeo-johnson")
        X_scaled_4 = scaler_4.fit_transform(X_train)
        X_resampled, y_resampled = augment_data(X_scaled_4, y_train, size, method=method)
        data_resampled = pd.DataFrame(X_resampled, columns=[f'feature_{i+1}' for i in range(X_resampled.shape[1])])
        data_resampled['e-type'] = y_resampled
        data_resampled.to_csv(f'./e-type_synthetic/e-type_{method}_yeo_{size}.csv', index=False)
        print(f"Saved dataset for {method} with {size} samples per class.")
        
methods = ['flow']
sizes = [100, 1000, 10000]  # Example sizes to augment data to

for method in methods:
    for size in sizes:
        print(f"Augmenting data using method {method} with size {size}")
        scaler_7 = PowerTransformer(method="yeo-johnson")
        X_scaled_7 = scaler_7.fit_transform(X_train)
        X_resampled, y_resampled = augment_data(X_scaled_7, y_train, size, method=method)
        data_resampled = pd.DataFrame(X_resampled, columns=[f'feature_{i+1}' for i in range(X_resampled.shape[1])])
        data_resampled['e-type'] = y_resampled
        data_resampled.to_csv(f'./e-type_synthetic/e-type_{method}_yeo_{size}.csv', index=False)
        print(f"Saved dataset for {method} with {size} samples per class.")

methods = ['gan']
sizes = [100, 1000, 10000]  # Example sizes to augment data to

for method in methods:
    for size in sizes:
        print(f"Augmenting data using method {method} with size {size}")
        scaler_10 = PowerTransformer(method="yeo-johnson")
        X_scaled_10 = scaler_10.fit_transform(X_train)
        X_resampled, y_resampled = augment_data(X_scaled_10, y_train, size, method=method)
        data_resampled = pd.DataFrame(X_resampled, columns=[f'feature_{i+1}' for i in range(X_resampled.shape[1])])
        data_resampled['e-type'] = y_resampled
        data_resampled.to_csv(f'./e-type_synthetic/e-type_{method}_yeo_{size}.csv', index=False)
        print(f"Saved dataset for {method} with {size} samples per class.")

print("Augmented datasets without reducer have been successfully saved.")

