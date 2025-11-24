"""
Synthetic data generator for E→e-type and M+E→e-type.

Generates per-class synthetic samples using: SMOTE, VAE, GAN, Flow, DDPM.
- Drops technical variables
- Re-references *_t_* times to stimulus onset
- Uses ColumnTransformer-consistent scaling (manual inverse if ColumnTransformer lacks it)
- Stratified 80/20 split; synthetic added to TRAIN only
- Uniform per-class targets (grid)
- Mahalanobis outlier filtering vs real-train (per class)
- Saves augmented TRAIN (inverse-transformed), real TEST saved separately
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

def log(msg): print(msg, flush=True)

# ---------- Load ----------
def load_data(csv_path):
    log(f"[LOAD] {csv_path}")
    df = pd.read_csv(csv_path, delimiter=';')
    assert 'e-type' in df.columns, "Expected an 'e-type' label column."
    return df

# ---------- Curate ----------
TECH_DROP = [
    "electrode_0_pa","seal_gohm","vm_for_sag",
    "neuron_reconstruction_type","scale_factor_x","scale_factor_y","scale_factor_z",
    "superseded"
]
def drop_technical(df):
    drop_cols = [c for c in TECH_DROP if c in df.columns]
    if drop_cols:
        log(f"[CURATE] Dropping technical columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
    else:
        log("[CURATE] No technical columns to drop.")
    return df

STIM_ONSETS = {"short_square": 1.0, "long_square": 1.0, "ramp": 1.0}
def re_reference_time(df):
    df = df.copy()
    n = 0
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if c.endswith('_t_short_square'):
                df[c] = df[c] - STIM_ONSETS['short_square']; n += 1
            elif c.endswith('_t_long_square'):
                df[c] = df[c] - STIM_ONSETS['long_square']; n += 1
            elif c.endswith('_t_ramp'):
                df[c] = df[c] - STIM_ONSETS['ramp']; n += 1
    log(f"[TIME] Re-referenced {n} timing columns to stimulus onset.")
    return df

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
def cols_endwith(df, suffixes):
    return [c for c in df.columns if any(c.endswith(sfx) for sfx in suffixes)]
def build_preprocessor(X):
    time_cols = cols_endwith(X, ['_t_short_square','_t_long_square','_t_ramp'])
    volt_cols = [c for c in X.columns if c.endswith('_v') or c in ['vrest','rheobase','max_firing_rate']]
    # keep generic list; your dataset has 'adaptation','avg_isi','sag' etc.
    ratio_cols = [c for c in X.columns if c in ['adaptation','sag_ratio','upstroke_downstroke_ratio','ISI_CV','avg_isi','sag']]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in num_cols if c not in set(time_cols + volt_cols + ratio_cols)]
    log(f"[COLUMNS] TIME={len(time_cols)} | VOLT={len(volt_cols)} | RATIO={len(ratio_cols)} | OTHER={len(other_cols)}")
    pre = ColumnTransformer(
        transformers=[
            ('time', RobustScaler(), time_cols),
            ('volt', StandardScaler(), volt_cols),
            ('ratio','passthrough', ratio_cols),
            ('other', StandardScaler(), other_cols)
        ],
        remainder='drop'
    )
    return pre, time_cols, volt_cols, ratio_cols, other_cols

from sklearn.model_selection import StratifiedShuffleSplit
def stratified_split(X, y, test_size=0.2, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    i_tr, i_te = next(sss.split(X, y))
    Xtr, Xte = X.iloc[i_tr].reset_index(drop=True), X.iloc[i_te].reset_index(drop=True)
    ytr, yte = y.iloc[i_tr].reset_index(drop=True), y.iloc[i_te].reset_index(drop=True)
    assert set(yte.unique()) == set(y.unique()), "Some classes missing in test split."
    log(f"[SPLIT] Train: {Xtr.shape} | Test: {Xte.shape} | Classes: {sorted(y.unique())}")
    return Xtr, Xte, ytr, yte

from scipy.spatial.distance import mahalanobis
def mahalanobis_filter_per_class(X_real_class, X_syn_class, cutoff_quantile=0.995, eps=1e-6):
    if len(X_real_class) < 3:
        return X_syn_class, 0.0
    mu = X_real_class.mean(axis=0)
    cov = np.cov(X_real_class.T) + eps * np.eye(X_real_class.shape[1])
    cov_inv = np.linalg.inv(cov)
    dists = np.array([mahalanobis(x, mu, cov_inv) for x in X_syn_class])
    thr = np.quantile(dists, cutoff_quantile)
    keep = dists <= thr
    kept = X_syn_class[keep]
    removed_pct = 100.0 * (1.0 - keep.mean()) if len(keep) else 0.0
    return kept, removed_pct

def ensure_numeric_impute(X):
    # Coerce to numeric (non-numeric -> NaN), impute with mean, drop constants
    X = X.apply(pd.to_numeric, errors='coerce')
    num = X.select_dtypes(include=[np.number]).columns
    X[num] = X[num].fillna(X[num].mean())
    const_cols = [c for c in num if X[c].nunique(dropna=False) <= 1]
    if const_cols:
        log(f"[CURATE] Dropping constant columns: {const_cols}")
        X = X.drop(columns=const_cols)
    return X

# ---------- Manual inverse (for older sklearn without ColumnTransformer.inverse_transform) ----------
def manual_inverse_transform(Z, pre, time_cols, volt_cols, ratio_cols, other_cols):
    """
    Reconstruct original-scale DataFrame from transformed matrix Z using fitted scalers.
    Order in Z is [time, volt, ratio (passthrough), other], matching build_preprocessor().
    """
    n_time = len(time_cols); n_volt = len(volt_cols); n_ratio = len(ratio_cols); n_other = len(other_cols)
    p = 0
    Z_time = Z[:, p:p+n_time];  p += n_time
    Z_volt = Z[:, p:p+n_volt];  p += n_volt
    Z_ratio = Z[:, p:p+n_ratio]; p += n_ratio
    Z_other = Z[:, p:p+n_other]; p += n_other

    inv = {}
    if n_time:
        inv_time = pre.named_transformers_['time'].inverse_transform(Z_time)
        for i, col in enumerate(time_cols): inv[col] = inv_time[:, i]
    if n_volt:
        inv_volt = pre.named_transformers_['volt'].inverse_transform(Z_volt)
        for i, col in enumerate(volt_cols): inv[col] = inv_volt[:, i]
    if n_ratio:
        for i, col in enumerate(ratio_cols): inv[col] = Z_ratio[:, i]  # passthrough
    if n_other:
        inv_other = pre.named_transformers_['other'].inverse_transform(Z_other)
        for i, col in enumerate(other_cols): inv[col] = inv_other[:, i]

    cols_order = time_cols + volt_cols + ratio_cols + other_cols
    X_inv = pd.DataFrame({c: inv[c] for c in cols_order})
    return X_inv

# ---------- SMOTE ----------
from imblearn.over_sampling import SMOTE
def make_target_counts_per_class(y_train, add_per_class):
    cls_counts = y_train.value_counts().to_dict()
    return {cls: cls_counts[cls] + add_per_class for cls in cls_counts}

def synth_smote(Ztr, ytr, add_per_class, random_state=42):
    target = make_target_counts_per_class(ytr, add_per_class)
    smote = SMOTE(random_state=random_state, sampling_strategy=target)
    X_res, y_res = smote.fit_resample(Ztr, ytr)
    from collections import Counter
    before = Counter(ytr); after = Counter(y_res)
    X_syn_list, y_syn_list = [], []
    df_res = pd.DataFrame(X_res); df_res['label'] = y_res.values
    for cls in after:
        df_cls = df_res[df_res['label'] == cls]
        extra = after[cls] - before[cls]
        if extra > 0:
            X_syn_list.append(df_cls.tail(extra).drop(columns=['label']).values)
            y_syn_list.append(np.array([cls]*extra))
    if X_syn_list:
        X_syn = np.vstack(X_syn_list); y_syn = np.hstack(y_syn_list)
    else:
        X_syn = np.empty((0, Ztr.shape[1])); y_syn = np.array([])
    return X_syn, y_syn

# ---------- VAE (Keras 3–safe functional) ----------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Sampling(layers.Layer):
    """Reparameterization + KL via add_loss (Keras 3–safe)."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = keras.random.normal(shape=keras.ops.shape(z_mean))
        z = z_mean + keras.ops.exp(0.5 * z_log_var) * eps
        kl = -0.5 * keras.ops.sum(1 + z_log_var - keras.ops.square(z_mean) - keras.ops.exp(z_log_var), axis=-1)
        self.add_loss(keras.ops.mean(kl))
        return z

def build_vae(n_features, latent_dim=8, width=128):
    # Encoder
    inputs = keras.Input(shape=(n_features,), name="vae_inputs")
    h = layers.Dense(width, activation='relu')(inputs)
    h = layers.Dense(width, activation='relu')(h)
    z_mean = layers.Dense(latent_dim, name="z_mean")(h)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)
    z = Sampling(name="z")([z_mean, z_log_var])
    # Decoder (reuse layers for separate decoder model)
    dec_h1 = layers.Dense(width, activation='relu', name="dec_h1")
    dec_h2 = layers.Dense(width, activation='relu', name="dec_h2")
    dec_out = layers.Dense(n_features, activation='linear', name="dec_out")
    h_dec = dec_h1(z); h_dec = dec_h2(h_dec); outputs = dec_out(h_dec)
    vae = keras.Model(inputs, outputs, name="vae")
    vae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    # Standalone decoder
    z_in = keras.Input(shape=(latent_dim,), name="z_in")
    d = dec_h1(z_in); d = dec_h2(d); x_gen = dec_out(d)
    decoder = keras.Model(z_in, x_gen, name="vae_decoder")
    return vae, decoder

def synth_vae(X_class, n_samples, epochs=200, batch_size=32, latent_dim=8, width=128, verbose=0):
    n_features = X_class.shape[1]
    vae, dec = build_vae(n_features, latent_dim=latent_dim, width=width)
    vae.fit(X_class.astype('float32'), X_class.astype('float32'), epochs=epochs, batch_size=batch_size, verbose=verbose)
    z = np.random.normal(size=(n_samples, latent_dim)).astype('float32')
    X_syn = dec.predict(z, verbose=0)
    return X_syn

# ---------- GAN (no BatchNorm; Keras-3 safe) ----------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
def synth_gan(X_class, n_samples, epochs=400, batch=64, nz=16, width=128, verbose=0):
    n_features = X_class.shape[1]
    opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    G = Sequential([
        Dense(width, input_shape=(nz,)), LeakyReLU(0.2),
        Dense(width), LeakyReLU(0.2),
        Dense(n_features, activation='linear')
    ])
    D = Sequential([
        Dense(width, input_shape=(n_features,)), LeakyReLU(0.2),
        Dense(width), LeakyReLU(0.2),
        Dense(1, activation='sigmoid')
    ])
    D.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    D.trainable = False
    GAN = Sequential([G, D]); GAN.compile(loss='binary_crossentropy', optimizer=opt)

    Xf = X_class.astype('float32'); m = Xf.shape[0]
    for epoch in range(epochs):
        idx = np.random.randint(0, m, batch)
        real = Xf[idx]
        z = np.random.normal(0, 1, (batch, nz)).astype('float32')
        fake = G.predict(z, verbose=0)
        y_real = np.ones((batch, 1), dtype='float32'); y_fake = np.zeros((batch,1), dtype='float32')
        D.train_on_batch(real, y_real); D.train_on_batch(fake, y_fake)
        z = np.random.normal(0,1,(batch, nz)).astype('float32')
        GAN.train_on_batch(z, np.ones((batch,1), dtype='float32'))
    z = np.random.normal(0,1,(n_samples, nz)).astype('float32')
    return G.predict(z, verbose=0)

# ---------- Flow (MAF via TFP) ----------
def synth_flow(X_class, n_samples, layers=4, hidden=64, epochs=300, batch=64, verbose=0):
    try:
        import tensorflow_probability as tfp
    except Exception as e:
        raise ImportError("tensorflow_probability is required for Flow. Install `tensorflow_probability`.") from e
    tfd = tfp.distributions; tfb = tfp.bijectors
    n_features = X_class.shape[1]
    made = [tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=tfb.AutoregressiveNetwork(params=2, hidden_units=[hidden, hidden]))
            for _ in range(layers)]
    flow = tfb.Chain(made)
    base = tfd.MultivariateNormalDiag(loc=tf.zeros([n_features]))
    dist = tfd.TransformedDistribution(distribution=base, bijector=flow)
    x_in = keras.Input(shape=(n_features,), dtype=tf.float32)
    log_prob = dist.log_prob(x_in)
    model = keras.Model(x_in, log_prob)
    model.add_loss(-tf.reduce_mean(log_prob)); model.compile(optimizer=keras.optimizers.Adam(1e-3))
    model.fit(X_class.astype(np.float32), epochs=epochs, batch_size=batch, verbose=verbose)
    X_syn = dist.sample(n_samples).numpy()
    return X_syn

# ---------- DDPM for tabular (Keras-safe) ----------
from tensorflow.keras.layers import Layer as KLayer, Concatenate
from tensorflow.keras import Model as KModel

class SinusoidalTimeEmbedding(KLayer):
    def __init__(self, dim=128, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
    def call(self, t):  # t: int32 shape (batch,)
        half = self.dim // 2
        freqs = tf.exp(tf.range(half, dtype=tf.float32) * (-tf.math.log(10000.0) / tf.cast(tf.maximum(half-1, 1), tf.float32)))
        ang = tf.cast(tf.expand_dims(t, -1), tf.float32) * tf.expand_dims(freqs, 0)
        return tf.concat([tf.sin(ang), tf.cos(ang)], axis=-1)

def build_ddpm_denoiser(n_features, t_dim=128, width=256):
    x_in = layers.Input(shape=(n_features,), name="x_t")
    t_in = layers.Input(shape=(), dtype=tf.int32, name="t")
    t_emb = SinusoidalTimeEmbedding(t_dim)(t_in)
    h = Concatenate()([x_in, t_emb])
    h = layers.Dense(width, activation='swish')(h)
    h = layers.Dense(width, activation='swish')(h)
    out = layers.Dense(n_features, activation='linear')(h)
    model = KModel([x_in, t_in], out, name="ddpm_denoiser")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return model

def ddpm_schedules(T=1000, beta_start=1e-4, beta_end=2e-2):
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas, axis=0)
    return betas, alphas, alpha_bar

def synth_ddpm(X_class, n_samples, T=400, epochs=400, batch=128, width=256, tdim=128, verbose=0, seed=42):
    tf.random.set_seed(seed)
    n_features = X_class.shape[1]
    model = build_ddpm_denoiser(n_features, t_dim=tdim, width=width)
    betas, alphas, abar = ddpm_schedules(T=T)

    x = X_class.astype(np.float32)
    n = x.shape[0]
    steps_per_epoch = max(1, n // batch)
    for ep in range(epochs):
        for _ in range(steps_per_epoch):
            idx = np.random.randint(0, n, size=batch)
            x0 = x[idx]
            t = np.random.randint(0, T, size=(batch,))
            eps = np.random.normal(size=x0.shape).astype(np.float32)
            a_bar_t = abar[t][:, None]
            xt = np.sqrt(a_bar_t) * x0 + np.sqrt(1.0 - a_bar_t) * eps
            model.train_on_batch([xt, t], eps)  # Keras-3 safe
        if verbose and ((ep+1) % 100 == 0 or ep == 0):
            log(f"[DDPM] epoch {ep+1}/{epochs}")

    # Sampling
    x_t = np.random.normal(size=(n_samples, n_features)).astype(np.float32)
    for t in reversed(range(T)):
        at = alphas[t]; bt = betas[t]; abar_t = abar[t]
        abar_prev = abar[t-1] if t > 0 else 1.0
        t_arr = np.full((n_samples,), t, dtype=np.int32)
        eps_pred = model.predict([x_t, t_arr], verbose=0)
        x0_pred = (x_t - np.sqrt(1 - abar_t) * eps_pred) / np.sqrt(abar_t + 1e-8)
        coef1 = np.sqrt(abar_prev) * bt / (1 - abar_t + 1e-8)
        coef2 = np.sqrt(at) * (1 - abar_prev) / (1 - abar_t + 1e-8)
        mean = coef1 * x0_pred + coef2 * x_t
        if t > 0:
            z = np.random.normal(size=x_t.shape).astype(np.float32)
            var = bt * (1 - abar_prev) / (1 - abar_t + 1e-8)
            x_t = mean + np.sqrt(var) * z
        else:
            x_t = mean
    return x_t.astype(np.float32)

# ---------- Orchestration ----------
def per_class_generator(gen_name, Xz_cls, n_needed):
    if n_needed <= 0:
        return np.empty((0, Xz_cls.shape[1]))
    if gen_name == 'vae':
        return synth_vae(Xz_cls, n_needed)
    elif gen_name == 'gan':
        return synth_gan(Xz_cls, n_needed)
    elif gen_name == 'flow':
        return synth_flow(Xz_cls, n_needed)
    elif gen_name == 'ddpm':
        return synth_ddpm(Xz_cls, n_needed)
    else:
        raise ValueError(f"Unknown generator: {gen_name}")

def augment_once(method, add_per_class, X_train, y_train, preprocessor, col_groups, outlier_q=0.995, seed=42):
    time_cols, volt_cols, ratio_cols, other_cols = col_groups
    Xtr = X_train.copy(); ytr = y_train.copy()
    Ztr = preprocessor.fit_transform(Xtr)
    classes = sorted(ytr.unique())
    X_syn_all, y_syn_all, removed_report = [], [], {}

    if method == 'smote':
        log(f"[AUG] SMOTE target: +{add_per_class} per class")
        Z_syn_all, y_syn = synth_smote(Ztr, ytr, add_per_class, random_state=seed)
        for cls in classes:
            idx_real = (ytr.values == cls); idx_syn = (y_syn == cls)
            if idx_syn.sum() == 0: continue
            kept, removed_pct = mahalanobis_filter_per_class(Ztr[idx_real], Z_syn_all[idx_syn], cutoff_quantile=outlier_q)
            removed_report[cls] = removed_pct
            if len(kept):
                X_syn_all.append(kept); y_syn_all.append(np.array([cls]*len(kept)))
    else:
        log(f"[AUG] {method.upper()} target: +{add_per_class} per class")
        for cls in classes:
            Z_cls = Ztr[ytr.values == cls]
            n_add = add_per_class
            if n_add <= 0: continue
            Z_syn_cls = per_class_generator(method, Z_cls, n_add)
            Z_syn_cls, removed_pct = mahalanobis_filter_per_class(Z_cls, Z_syn_cls, cutoff_quantile=outlier_q)
            removed_report[cls] = removed_pct
            if len(Z_syn_cls):
                X_syn_all.append(Z_syn_cls); y_syn_all.append(np.array([cls]*len(Z_syn_cls)))

    # Inverse-transform to original units (manual if ColumnTransformer lacks inverse)
    if X_syn_all:
        Z_syn = np.vstack(X_syn_all); y_syn = np.hstack(y_syn_all)
        if hasattr(preprocessor, "inverse_transform"):
            X_syn = preprocessor.inverse_transform(Z_syn)
            X_syn = pd.DataFrame(X_syn, columns=(time_cols + volt_cols + ratio_cols + other_cols))
        else:
            X_syn = manual_inverse_transform(Z_syn, preprocessor, time_cols, volt_cols, ratio_cols, other_cols)
    else:
        X_syn = pd.DataFrame(columns=(time_cols + volt_cols + ratio_cols + other_cols)); y_syn = np.array([])

    if removed_report:
        mean_removed = np.mean(list(removed_report.values()))
        log(f"[FILTER] Mean % removed by Mahalanobis = {mean_removed:.2f}%")
    return X_syn, y_syn, removed_report

def save_augmented_run(prefix, method, add_per_class, X_train_aug, y_train_aug, X_test, y_test, removed_report, config):
    out_train = f"{prefix}_train_aug_{method}_{add_per_class}.csv"
    df_train = X_train_aug.copy(); df_train['e-type'] = y_train_aug; df_train.to_csv(out_train, index=False)
    out_test = f"{prefix}_test_real.csv"
    df_test = X_test.copy(); df_test['e-type'] = y_test; df_test.to_csv(out_test, index=False)
    out_rem = f"{prefix}_removed_{method}_{add_per_class}.csv"
    if removed_report:
        pd.DataFrame([{"class": k, "pct_removed": v} for k, v in removed_report.items()]).to_csv(out_rem, index=False)
    else:
        pd.DataFrame(columns=['class','pct_removed']).to_csv(out_rem, index=False)
    out_cfg = f"{prefix}_config_{method}_{add_per_class}.json"
    with open(out_cfg, "w") as f: json.dump(config, f, indent=2)
    log(f"[SAVE] Train+synthetic: {out_train}")
    log(f"[SAVE] Test (real):    {out_test}")
    log(f"[SAVE] Removed%:       {out_rem}")
    log(f"[SAVE] Config:         {out_cfg}")

# ---------- MAIN ----------
if __name__ == "__main__":
    # ======== CONFIG ========
    INPUT_CSV = "../data/e-type.csv"      # or "../data/mee-type.csv"
    METHODS = ['smote', 'vae', 'gan', 'flow', 'ddpm']
    SYN_PER_CLASS_GRID = [0, 100, 500, 1000, 5000, 10000]
    OUTLIER_QUANTILE = 0.995
    SEED = 42
    PREFIX = "aug_e" if "e-type" in os.path.basename(INPUT_CSV) else "aug_mee"
    # ========================

    # GPU memory growth (TF)
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        if gpus: log(f"[GPU] Enabled memory growth on {len(gpus)} GPU(s).")
    except Exception as e:
        log(f"[GPU] {e}")

    # 1) Load & curate
    df = load_data(INPUT_CSV)
    df = drop_technical(df)

    # 2) Features/labels, impute, timing
    X = df.drop(columns=['e-type']); y = df['e-type'].astype(str)
    X = ensure_numeric_impute(X); X = re_reference_time(X)

    # 3) Preprocessor
    pre, tcols, vcols, rcols, ocols = build_preprocessor(X)

    # 4) Stratified split (keep classes in test)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, seed=SEED)

    # Save raw splits for parity
    X_train.to_csv(f"{PREFIX}_X_train_real.csv", index=False)
    X_test.to_csv(f"{PREFIX}_X_test_real.csv", index=False)
    y_train.to_csv(f"{PREFIX}_y_train_real.csv", index=False)
    y_test.to_csv(f"{PREFIX}_y_test_real.csv", index=False)
    log(f"[SAVE] Real splits saved with prefix {PREFIX}")

    col_groups = (tcols, vcols, rcols, ocols)

    # 5) Generate per method & grid
    for method in METHODS:
        for add_per_class in SYN_PER_CLASS_GRID:
            log(f"\n[RUN] Method={method.upper()} | +{add_per_class} per class")
            t0 = time.time()
            try:
                X_syn, y_syn, removed = augment_once(
                    method=method,
                    add_per_class=add_per_class,
                    X_train=X_train, y_train=y_train,
                    preprocessor=pre,
                    col_groups=col_groups,
                    outlier_q=OUTLIER_QUANTILE,
                    seed=SEED
                )
                if len(X_syn) > 0:
                    # ensure same column order as train
                    X_syn = X_syn[X_train.columns]
                    X_train_aug = pd.concat([X_train.reset_index(drop=True), X_syn.reset_index(drop=True)], axis=0)
                    y_train_aug = pd.concat([y_train.reset_index(drop=True), pd.Series(y_syn, name='e-type')], axis=0)
                else:
                    X_train_aug = X_train.copy(); y_train_aug = y_train.copy()

                cfg = {
                    "input_csv": INPUT_CSV,
                    "method": method,
                    "add_per_class": add_per_class,
                    "outlier_quantile": OUTLIER_QUANTILE,
                    "seed": SEED,
                    "dropped_cols": [c for c in TECH_DROP if c in df.columns],
                    "time_cols": tcols, "volt_cols": vcols, "ratio_cols": rcols, "other_cols": ocols
                }
                save_augmented_run(PREFIX, method, add_per_class, X_train_aug, y_train_aug,
                                   X_test, y_test, removed, cfg)

                log(f"[OK] Done in {time.time()-t0:.1f}s")
            except Exception as e:
                log(f"[ERR] {method} +{add_per_class}/class: {e}")
