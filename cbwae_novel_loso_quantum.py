# cbwae_novel_loso_quantum.py
# CBWAE LOSO with advanced evaluation metrics, novelty mechanisms, and a quantum-inspired feature module
# Author: Mustafa
# Usage: Put all EDF files anywhere under DATA_ROOT (e.g. /content in Colab). Script will discover S###R04.edf & S###R06.edf pairs.
# Output: per-subject CSV and summary in OUTDIR.

import os, glob, math, random, warnings, csv, time
from collections import defaultdict

import numpy as np
import pandas as pd
import mne
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef,
    cohen_kappa_score, log_loss, brier_score_loss, confusion_matrix
)
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import joblib

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)
random.seed(RNG)
tf.random.set_seed(RNG)

# ---------------- USER SETTINGS ----------------
DATA_ROOT = "/content"          # root to search recursively for EDF files (change to your path)
OUTDIR = "./cbwae_novel_loso_out"
os.makedirs(OUTDIR, exist_ok=True)

WINDOW_SECONDS = 2.0
INCLUDE_WAVELET = True
WAVELET_LEVEL = 3
BAND = (8, 12)
SFREQ_EXPECTED = 160

N_SPLITS_OOF = 5
META_EPOCHS = 40
META_BATCH = 32
VERBOSE = 0

# Novelty/inference
CONFIDENCE_THRESHOLD = 0.75
MAX_UNCERTAINTY_LOOPS = 3
CALIBRATION_BUFFER_MAX = 200
PSEUDO_LABEL_CONF_THRESH = 0.95

# Quantum-inspired features
QF_DIM = 32
QF_PHASE_SCALE = 10.0
QF_RANDOM_SEED = 1234

# ------------------------------------------------

# ---------- utility functions ----------
def find_edf_pairs(root):
    """Search recursively and return dict subject_id -> {'R04':path, 'R06':path} for files matching S###R04/06.edf (case-insensitive)."""
    files = glob.glob(os.path.join(root, "**", "*.edf"), recursive=True)
    files += glob.glob(os.path.join(root, "**", "*.EDF"), recursive=True)
    pairs = defaultdict(dict)
    for f in files:
        basename = os.path.basename(f)
        # expect format SxxxRyy.edf (e.g., S001R04.edf)
        if len(basename) < 8:
            continue
        name = basename.split('.')[0]
        # crude parse
        if name.startswith('S') and 'R' in name:
            try:
                sid = name[1:4]
                run = name.split('R')[-1][:2]  # '04' or '06'
                key = int(sid)
                pairs[key][f"R{run}"] = f
            except Exception:
                continue
    # only return subjects with both R04 and R06
    good = {k: v for k, v in pairs.items() if 'R04' in v and 'R06' in v}
    missing = [k for k in pairs.keys() if k not in good]
    if missing:
        print(f"Warning: {len(missing)} subjects missing one of the pair files (they will be skipped).")
    return good

def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        df = model.decision_function(X)
        if df.ndim == 1:
            logits = np.vstack([-df, df]).T
        else:
            logits = df
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    preds = model.predict(X)
    classes = np.unique(preds)
    prob = np.zeros((len(preds), len(classes)))
    for i, p in enumerate(preds):
        prob[i, np.where(classes == p)[0][0]] = 1.0
    return prob

def sample_entropy(probs):
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)

def minmax_norm(v):
    v = np.array(v)
    if v.size == 0:
        return v
    if v.max() == v.min():
        return np.zeros_like(v)
    return (v - v.min()) / (v.max() - v.min())

# ---------- signal & features ----------
def bandpower(data, sf, band, nperseg=None):
    if nperseg is None:
        nperseg = min(256, len(data))
    freqs, psd = welch(data, fs=sf, nperseg=nperseg)
    if len(freqs) < 2:
        return 0.0
    freq_res = freqs[1] - freqs[0]
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    if np.sum(idx) == 0:
        return 0.0
    return np.trapz(psd[idx], dx=freq_res)

def wavelet_features(sig, level=WAVELET_LEVEL):
    try:
        import pywt
        coeffs = pywt.wavedec(sig, 'db4', level=level)
        feats = []
        for c in coeffs[1:]:
            feats.extend([np.mean(c), np.std(c), np.percentile(c, 75)])
        return feats
    except Exception:
        return [0.0] * (level * 3)

def extract_features_segment(segment, sfreq=SFREQ_EXPECTED, include_wavelet=INCLUDE_WAVELET):
    feats = []
    for ch in range(segment.shape[0]):
        sig = segment[ch, :]
        feats += [
            np.mean(sig),
            np.std(sig),
            skew(sig),
            kurtosis(sig),
            bandpower(sig, sfreq, BAND)
        ]
        if include_wavelet:
            feats += wavelet_features(sig, level=WAVELET_LEVEL)
    return np.array(feats, dtype=np.float32)

# ---------- quantum-inspired map ----------
def quantum_inspired_map(X_feat, qdim=QF_DIM, seed=QF_RANDOM_SEED, phase_scale=QF_PHASE_SCALE):
    if X_feat.ndim == 1:
        X_feat = X_feat.reshape(1, -1)
    n, d = X_feat.shape
    rng = np.random.RandomState(seed)
    W = rng.randn(d, qdim)
    phases = X_feat.dot(W)
    phases = phases / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    phases = phases * phase_scale
    complex_emb = np.exp(1j * phases)
    U = rng.randn(qdim, qdim)
    try:
        Q, _ = np.linalg.qr(U)
    except Exception:
        Q = U
    mixed = complex_emb.dot(Q)
    k = min(qdim, max(4, qdim // 4))
    prototypes = Q[:, :k]
    proto_complex = np.exp(1j * prototypes.T)
    proto_complex = proto_complex / (np.linalg.norm(proto_complex, axis=1, keepdims=True) + 1e-12)
    mixed_norm = mixed / (np.linalg.norm(mixed, axis=1, keepdims=True) + 1e-12)
    fidelities = np.abs(mixed_norm.dot(proto_complex.T))**2
    mag = np.abs(mixed)
    ang_mean = np.angle(mixed).mean(axis=1).reshape(-1, 1)
    mag_stats = np.hstack([mag.mean(axis=1).reshape(-1,1), mag.std(axis=1).reshape(-1,1)])
    qfeatures = np.hstack([fidelities[:, :k], mag_stats, ang_mean])
    if qfeatures.shape[1] < qdim:
        pad = np.zeros((n, qdim - qfeatures.shape[1]))
        qfeatures = np.hstack([qfeatures, pad])
    elif qfeatures.shape[1] > qdim:
        qfeatures = qfeatures[:, :qdim]
    return np.real(qfeatures)

# ---------- calibration & memory ----------
class CalibrationBuffer:
    def __init__(self, max_size=200):
        self.max_size = max_size
        self.X = []; self.y = []
    def add(self, X_batch, y_batch):
        for Xb, yb in zip(X_batch, y_batch):
            if len(self.X) < self.max_size:
                self.X.append(Xb); self.y.append(yb)
            else:
                idx = random.randint(0, self.max_size-1)
                self.X[idx], self.y[idx] = Xb, yb
    def get(self):
        if len(self.X) == 0: return None, None
        return np.vstack(self.X), np.array(self.y)
    def clear(self): self.X, self.y = [], []

# ---------- evaluation helpers ----------
def expected_calibration_error(probs, labels, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confidences > lo) & (confidences <= hi)
        if np.sum(mask) == 0:
            continue
        acc = np.mean(predictions[mask] == labels[mask])
        avg_conf = np.mean(confidences[mask])
        ece += (np.sum(mask) / probs.shape[0]) * abs(avg_conf - acc)
    return ece

def safe_error_correlation(err_mat):
    corr = np.corrcoef(err_mat)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr

def evaluate_predictions(y_true, y_pred, y_proba):
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['F1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['BalancedAcc'] = balanced_accuracy_score(y_true, y_pred)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    metrics['Kappa'] = cohen_kappa_score(y_true, y_pred)
    try:
        metrics['LogLoss'] = log_loss(y_true, y_proba)
    except Exception:
        metrics['LogLoss'] = np.nan
    if y_proba.shape[1] == 2:
        metrics['Brier'] = brier_score_loss(y_true, y_proba[:,1])
    else:
        onehot = np.zeros_like(y_proba)
        onehot[np.arange(len(y_true)), y_true] = 1
        metrics['Brier'] = np.mean(np.sum((onehot - y_proba)**2, axis=1))
    metrics['ECE'] = expected_calibration_error(y_proba, y_true, n_bins=10)
    confidences = np.max(y_proba, axis=1)
    correctness = (y_true == y_pred).astype(int)
    if np.sum(correctness==1) > 0 and np.sum(correctness==0) > 0:
        metrics['ConfidenceGap'] = float(np.mean(confidences[correctness==1]) - np.mean(confidences[correctness==0]))
    else:
        metrics['ConfidenceGap'] = 0.0
    uncertainties = -np.sum(y_proba * np.log(y_proba + 1e-12), axis=1)
    if np.std(uncertainties) == 0 or np.std(correctness) == 0:
        metrics['UncertaintyCorr'] = 0.0
    else:
        metrics['UncertaintyCorr'] = float(np.corrcoef(uncertainties, correctness)[0,1])
    metrics['ConfusionMatrix'] = confusion_matrix(y_true, y_pred).tolist()
    return metrics

# ---------- load & window EDF ----------
def load_and_window(edf_path, label, window_seconds=WINDOW_SECONDS):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, stim=False)
    if len(picks) == 0:
        raise RuntimeError(f"No EEG channels in {edf_path}")
    data = raw.get_data(picks=picks)
    sfreq = raw.info.get('sfreq', SFREQ_EXPECTED) or SFREQ_EXPECTED
    window_size = int(window_seconds * sfreq)
    n_windows = data.shape[1] // window_size
    feats, labels = [], []
    for w in range(n_windows):
        seg = data[:, w*window_size : (w+1)*window_size]
        feats.append(extract_features_segment(seg, sfreq=sfreq))
        labels.append(label)
    if len(feats) == 0:
        return np.empty((0,0)), np.empty((0,))
    return np.vstack(feats), np.array(labels, dtype=int)

# ---------- main LOSO pipeline ----------
def run_loso(data_root=DATA_ROOT, outdir=OUTDIR):
    print("Discovering EDF pairs under:", data_root)
    pairs = find_edf_pairs(data_root)
    subj_ids = sorted(pairs.keys())
    print(f"Found {len(subj_ids)} subjects with both runs.")
    if len(subj_ids) == 0:
        raise RuntimeError("No complete subject pairs found. Check DATA_ROOT and filenames.")

    # load features for all subjects (cache)
    X_subjects, y_subjects = [], []
    subject_map = []
    print("Loading and extracting features for each subject (this may take time)...")
    for sid in subj_ids:
        left = pairs[sid]['R04']
        right = pairs[sid]['R06']
        Xl, yl = load_and_window(left, 0)
        Xr, yr = load_and_window(right, 1)
        X_all = np.vstack([Xl, Xr])
        y_all = np.concatenate([yl, yr])
        # add quantum-inspired features and append
        qf = quantum_inspired_map(X_all, qdim=QF_DIM)
        X_aug = np.hstack([X_all, qf])
        X_subjects.append(X_aug)
        y_subjects.append(y_all)
        subject_map.append(sid)
        print(f" Subject S{sid:03d}: windows {X_aug.shape[0]}, features {X_aug.shape[1]}")

    # prepare base learners
    names = ['XGBoost','Logistic','RandomForest','kNN','SVM']
    base_classifiers = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RNG),
        'Logistic': LogisticRegression(max_iter=1000, random_state=RNG),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RNG),
        'kNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True, random_state=RNG)
    }
    n_classes = 2

    feature_scaler = StandardScaler()
    results_rows = []

    # LOSO loop
    for i, (X_test_subj, y_test_subj, sid) in enumerate(zip(X_subjects, y_subjects, subject_map), start=1):
        print(f"\n=== LOSO {i}/{len(subject_map)}: Subject S{sid:03d} ===")
        # build train set from all other subjects
        X_train = np.vstack([X_subjects[j] for j in range(len(X_subjects)) if j != (i-1)])
        y_train = np.concatenate([y_subjects[j] for j in range(len(y_subjects)) if j != (i-1)])
        # scale
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test_subj)

        # OOF for meta features
        oof_probs = {n: np.zeros((X_train_scaled.shape[0], n_classes), dtype=np.float32) for n in names}
        final_models = {}
        val_probs_test = {}

        skf = StratifiedKFold(n_splits=min(N_SPLITS_OOF, max(2, len(np.unique(y_train)))), shuffle=True, random_state=RNG)
        for name, clf in base_classifiers.items():
            oof = np.zeros((X_train_scaled.shape[0], n_classes))
            for tr_idx, val_idx in skf.split(X_train_scaled, y_train):
                model = clf.__class__(**clf.get_params())
                try:
                    model.set_params(random_state=RNG)
                except Exception:
                    pass
                model.fit(X_train_scaled[tr_idx], y_train[tr_idx])
                oof[val_idx] = safe_predict_proba(model, X_train_scaled[val_idx])
            oof_probs[name] = oof
            # final model trained on all train
            fm = clf.__class__(**clf.get_params())
            try:
                fm.set_params(random_state=RNG)
            except Exception:
                pass
            fm.fit(X_train_scaled, y_train)
            final_models[name] = fm
            val_probs_test[name] = safe_predict_proba(fm, X_test_scaled)

        # compute OOF preds & model quality
        oof_preds = {n: np.argmax(oof_probs[n], axis=1) for n in names}
        f1_scores = {n: f1_score(y_train, oof_preds[n], average='weighted') for n in names}
        err_vectors = {n: (oof_preds[n] != y_train).astype(int) for n in names}
        err_mat = np.vstack([err_vectors[n] for n in names])
        corr = safe_error_correlation(err_mat)

        avg_corr = np.mean(corr, axis=0)
        raw_w = np.array([f1_scores[n] * (1.0 - avg_corr[i]) for i,n in enumerate(names)], dtype=np.float32)
        raw_w = np.nan_to_num(raw_w, nan=0.0, posinf=0.0, neginf=0.0)
        model_weights = raw_w / raw_w.sum() if raw_w.sum() > 0 else np.ones_like(raw_w)/len(raw_w)
        model_weights_dict = dict(zip(names, model_weights))

        # meta features
        raw_probs_train = np.concatenate([oof_probs[n] for n in names], axis=1)
        per_model_ent_train = np.vstack([sample_entropy(oof_probs[n]) for n in names]).T
        oof_preds_matrix = np.vstack([oof_preds[n] for n in names]).T
        agreements_train = np.array([np.max(np.unique(row, return_counts=True)[1]) / len(row) for row in oof_preds_matrix]).reshape(-1,1)
        per_model_ent_norm = np.vstack([minmax_norm(per_model_ent_train[:,i]) for i in range(per_model_ent_train.shape[1])]).T
        per_model_conf_train = 1.0 - per_model_ent_norm
        weighted_probs_train = np.zeros((raw_probs_train.shape[0], n_classes))
        for i_n, n in enumerate(names):
            gw = model_weights_dict[n]
            weighted_probs_train += oof_probs[n] * (gw * per_model_conf_train[:, i_n].reshape(-1,1))

        meta_X_train = np.hstack([raw_probs_train, weighted_probs_train, per_model_ent_train, agreements_train])
        meta_y_train = y_train

        # meta test
        raw_probs_test = np.concatenate([val_probs_test[n] for n in names], axis=1)
        per_model_ent_test = np.vstack([sample_entropy(val_probs_test[n]) for n in names]).T
        per_model_ent_test_norm = np.vstack([minmax_norm(per_model_ent_test[:,i]) for i in range(per_model_ent_test.shape[1])]).T
        per_model_conf_test = 1.0 - per_model_ent_test_norm
        weighted_probs_test = np.zeros((X_test_scaled.shape[0], n_classes))
        for i_n, n in enumerate(names):
            gw = model_weights_dict[n]
            weighted_probs_test += val_probs_test[n] * (gw * per_model_conf_test[:, i_n].reshape(-1,1))
        oof_preds_test_matrix = np.vstack([np.argmax(val_probs_test[n], axis=1) for n in names]).T
        agreements_test = np.array([np.max(np.unique(row, return_counts=True)[1]) / len(row) for row in oof_preds_test_matrix]).reshape(-1,1)

        meta_X_test = np.hstack([raw_probs_test, weighted_probs_test, per_model_ent_test, agreements_test])
        meta_y_test = y_test_subj

        # scale meta, train meta DNN
        meta_scaler = StandardScaler()
        meta_X_train_scaled = meta_scaler.fit_transform(meta_X_train)
        meta_X_test_scaled = meta_scaler.transform(meta_X_test)

        tf.keras.backend.clear_session()
        inp = meta_X_train_scaled.shape[1]
        meta_model = Sequential([
            Input(shape=(inp,)), Dense(128, activation='relu'), Dropout(0.4),
            Dense(64, activation='relu'), Dropout(0.3),
            Dense(n_classes, activation='softmax')
        ])
        meta_model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        meta_model.fit(meta_X_train_scaled, meta_y_train, validation_split=0.15,
                       epochs=META_EPOCHS, batch_size=META_BATCH, verbose=VERBOSE)

        # predict & eval
        meta_proba = meta_model.predict(meta_X_test_scaled)
        meta_pred = np.argmax(meta_proba, axis=1)
        meta_metrics = evaluate_predictions(meta_y_test, meta_pred, meta_proba)

        # best single baseline (by OOF f1)
        best_name = max(f1_scores.items(), key=lambda kv: kv[1])[0]
        best_proba = safe_predict_proba(final_models[best_name], X_test_scaled)
        best_pred = np.argmax(best_proba, axis=1)
        best_metrics = evaluate_predictions(meta_y_test, best_pred, best_proba)

        weighted_proba = weighted_probs_test.copy()
        weighted_pred = np.argmax(weighted_proba, axis=1)
        weighted_metrics = evaluate_predictions(meta_y_test, weighted_pred, weighted_proba)

        row = {
            'subject': f"S{sid:03d}",
            'meta_acc': meta_metrics['Accuracy'],
            'meta_f1': meta_metrics['F1'],
            'meta_balanced_acc': meta_metrics['BalancedAcc'],
            'meta_mcc': meta_metrics['MCC'],
            'meta_kappa': meta_metrics['Kappa'],
            'meta_logloss': meta_metrics['LogLoss'],
            'meta_brier': meta_metrics['Brier'],
            'meta_ece': meta_metrics['ECE'],
            'meta_confidence_gap': meta_metrics['ConfidenceGap'],
            'meta_uncertainty_corr': meta_metrics['UncertaintyCorr'],
            'meta_agreement_rate': float(np.mean(agreements_test)),
            'best_acc': best_metrics['Accuracy'],
            'best_f1': best_metrics['F1'],
            'best_logloss': best_metrics['LogLoss'],
            'weighted_acc': weighted_metrics['Accuracy'],
            'weighted_f1': weighted_metrics['F1'],
            'weighted_logloss': weighted_metrics['LogLoss']
        }
        results_rows.append(row)
        print(f" S{sid:03d} - META acc {row['meta_acc']:.4f} f1 {row['meta_f1']:.4f} | BEST {row['best_acc']:.4f} f1 {row['best_f1']:.4f}")

    # save per-subject CSV
    df = pd.DataFrame(results_rows)
    csv_path = os.path.join(outdir, "cbwae_loso_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-subject results to {csv_path}")

    # summary
    def mean_std(col): return f"{df[col].mean():.4f} ± {df[col].std():.4f}"
    summary_report = {
        'meta_acc': mean_std('meta_acc'),
        'meta_f1': mean_std('meta_f1'),
        'meta_ece': mean_std('meta_ece'),
        'meta_uncertainty_corr': mean_std('meta_uncertainty_corr'),
        'meta_agreement_rate': mean_std('meta_agreement_rate'),
        'best_acc': mean_std('best_acc'),
        'weighted_acc': mean_std('weighted_acc')
    }
    summary_path = os.path.join(outdir, "cbwae_loso_summary.txt")
    with open(summary_path, "w") as f:
        f.write("LOSO Summary (mean ± std):\n")
        for k, v in summary_report.items():
            f.write(f"{k}: {v}\n")
    print("Summary saved to", summary_path)
    return df, summary_report

if __name__ == "__main__":
    t0 = time.time()
    df, summary = run_loso(DATA_ROOT, OUTDIR)
    print("\nDone in %.1f seconds." % (time.time() - t0))
