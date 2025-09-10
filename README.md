# CBWAE-Quantum: Correlation-Based Weighted Average Ensemble with Quantum-Inspired Features for EEG Motor Imagery Classification

CBWAE-Quantum: Correlation-Based Weighted Average Ensemble with Quantum-Inspired Features for EEG Motor Imagery (LOSO)
Overview

CBWAE-Quantum is a robust ensemble framework for EEG Motor Imagery (MI) classification under Leave-One-Subject-Out (LOSO) evaluation.
It combines classical and quantum-inspired methods to address noise, non-stationarity, and high inter-subject variability in BCIs.

Key Features

Diverse Base Classifiers: RF, SVM, kNN, XGBoost, Logistic Regression

Correlation-Aware Weighting: promotes accurate + diverse models

Quantum-Inspired Feature Mapping: nonlinear embeddings in complex space

Uncertainty-Aware Correction: retries low-confidence predictions

Deep Meta-Classifier: learns nonlinear dependencies across models

Self-Calibration: adapts dynamically per subject/session

Novel Contributions

Quantum embeddings: capture complex EEG channel relations

Uncertainty correction loop with calibration buffer & pseudo-labeling

Dynamic weighting: per-batch weights = F1 × (1 − avg_correlation)

Entropy-weighted fusion + confidence gating for expert override

Advanced evaluation metrics: accuracy, F1, Cohen’s κ, MCC, log-loss, Brier score, ECE, calibration error

Dataset

We demonstrate on the PhysioNet EEG Motor Movement/Imagery Dataset
:

Subjects: 109

Channels: 64 EEG @ 160 Hz

Tasks: Motor execution & imagery (LH, RH, Feet, Tongue)

Example (tutorial setting):

SUBJECT = "S001"
EDF_FILES = {"left": "S001R04.edf", "right": "S001R06.edf"}

Preprocessing Pipeline

Windowing: 2s non-overlapping (~320 samples)

Features per window × channel:

Statistical: mean, std, skew, kurtosis

Spectral: α bandpower (8–12 Hz)

Wavelet: db4 coefficients

Entropy: Shannon entropy

Normalization: StandardScaler

Augmentation: Quantum-inspired embeddings (QF_DIM=32)

Split: LOSO → train on all but one subject, test on left-out

Base Models

Logistic Regression

Random Forest

kNN

SVM

XGBoost
All produce probabilities → fusion + meta-classifier.

Weighted Fusion & Meta-Classifier

Weights: wi = F1i × (1 − avg_corr_i)

Fusion: entropy-adjusted probabilities, normalized per sample

Meta-Classifier Input: raw probs, weighted probs, entropy, agreement

Architecture:

Dense(128) → ReLU → Dropout(0.4)

Dense(64) → ReLU → Dropout(0.3)

Dense(#classes) → Softmax

Installation
pip install mne pyedflib pywavelets xgboost tensorflow scikit-learn matplotlib

Usage

Upload EEG EDF files (e.g., S001R04.edf, S001R06.edf)

Define subject paths in config:

SUBJECT = "S001"
EDF_FILES = {"left": "S001R04.edf", "right": "S001R06.edf"}


Run LOSO + CBWAE-Quantum pipeline:

df, summary = run_loso(DATA_ROOT, OUTDIR)


Outputs:

Per-subject CSV

Summary metrics

Uncertainty plots

Model weight distributions

Results (Expected)

Outperforms single models

Reduces uncertainty by ~30% with weighted fusion + meta-classifier

Maintains temporal consistency

Strong LOSO generalization

Future Work

Multi-class MI (LH, RH, Feet, Tongue)

CNN/RNN base models for raw EEG

Domain adaptation across subjects

Real-time BCI deployment on FPGA/edge

Citation

If you use CBWAE-Quantum in your research, please cite:

@article{helloamad,
  title={CBWAE-Quantum: Correlation-Based Weighted Average Ensemble with Quantum-Inspired Features for EEG Motor Imagery Classification},
  author=Mustafa Arif,
  year={2025},
  journal={GitHub Repository}
}
