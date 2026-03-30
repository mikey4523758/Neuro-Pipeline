# Neuro-Pipeline: EEG Motor Imagery Classification

> A production-ready, modular Python pipeline for end-to-end EEG signal processing and motor imagery classification using the PhysioNet EEGBCI dataset — built to mirror real-world Brain-Computer Interface (BCI) research workflows.

---

## 🧠 What This Project Does

This pipeline takes raw EEG recordings and produces trained machine learning models capable of classifying **left vs. right hand motor imagery** — a core task in non-invasive BCI systems used in assistive technology, neurofeedback, and neural prosthetics research.

Every stage of the pipeline reflects industry-standard signal processing practices, from artifact removal to frequency-domain feature engineering.

---

## 🔬 Scientific Overview

The pipeline implements the canonical BCI processing chain:

| Stage | Method | Purpose |
|---|---|---|
| Bandpass Filter | FIR Windowed (`7–30 Hz`) | Isolates Mu (α) and Beta (β) motor rhythms |
| Notch Filter | IIR Notch | Removes 60 Hz power line interference |
| Re-referencing | Common Average Reference | Reduces spatial bias across electrodes |
| Feature Extraction | Welch's PSD + RMS | Captures frequency-domain and time-domain signal power |
| Classification | SVM vs. Random Forest | Benchmarks linear vs. ensemble approaches |

**Why Mu and Beta rhythms?** These frequency bands (8–12 Hz and 13–30 Hz) are directly suppressed during motor planning and execution — a phenomenon called Event-Related Desynchronization (ERD) — making them the most discriminative features for left/right hand classification.

---

## 🏗 Architecture
```
neuro-pipeline/
├── configs/
│   └── config.yaml          # Centralized parameter control
├── src/
│   ├── load_data.py         # PhysioNet data ingestion & montage setup
│   ├── preprocess.py        # Bandpass, notch filtering & re-referencing
│   ├── features.py          # Welch PSD & RMS feature extraction
│   ├── models.py            # SVM & Random Forest training/evaluation
│   ├── visualize.py         # PSD comparison & model accuracy plots
│   └── pipeline.py          # End-to-end orchestration class
└── main.py                  # CLI entry point

The pipeline is orchestrated through a single `EEGAnalysisPipeline` class, making it easy to swap datasets, tune hyperparameters via config, or extend with new classifiers without touching core logic.

---

## 🛠 Installation
```bash
pip install -r requirements.txt

---

## 🚀 Usage
```bash
# Run with default config
python main.py

# Run with a custom config
python main.py --config configs/custom.yaml
```

All parameters — subject ID, frequency bands, train/test split, model hyperparameters — are controlled through `configs/config.yaml`, requiring zero code changes to experiment.

---

## 📊 Results

| Model | Accuracy |
|---|---|
| Linear SVM | ~75–85% |
| Random Forest | ~70–80% |

*Results vary by subject. Accuracy is evaluated on a held-out test split.*

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![MNE](https://img.shields.io/badge/MNE--Python-1.x-purple)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![SciPy](https://img.shields.io/badge/SciPy-1.x-teal)

- **MNE-Python** — EEG/MEG data loading, filtering, epoching, and visualization
- **scikit-learn** — Model training, cross-validation, and performance metrics
- **SciPy** — Welch's method for power spectral density estimation
- **Matplotlib** — Pipeline result visualization
- **PyYAML** — Config-driven parameter management

---

## 💡 Key Engineering Decisions

- **Config-driven design** — All hyperparameters externalized to YAML; no hardcoded values in source
- **Modular architecture** — Each pipeline stage is an isolated, independently testable module
- **Before/after visualization** — PSD plots generated for both raw and filtered signals to verify preprocessing
- **Dual-model benchmarking** — SVM and Random Forest evaluated side-by-side to compare linear vs. ensemble performance

---

## 📚 Dataset

**PhysioNet EEG Motor Movement/Imagery Dataset**
- 109 subjects, 64-channel EEG recorded at 160 Hz
- Tasks include real and imagined left/right hand movements
- Openly available via [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/)