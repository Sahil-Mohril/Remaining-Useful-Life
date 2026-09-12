# Bearing RUL Prediction with Fine-Tuned PatchTST

End-to-end Remaining Useful Life (RUL) prediction for rolling-element bearings
using the IEEE PHM 2012 Prognostic Challenge dataset, with a fine-tuned
PatchTST time-series model from HuggingFace `transformers`.

## Overview

This project implements a complete ML pipeline that predicts bearing RUL from
raw vibration sensor data using **transfer learning** — fine-tuning the
PatchTST architecture (a Transformer-based time-series model) rather than
training from scratch.

### Architecture

```
Raw Vibration Data (25,600 Hz, 2,560 samples/10s)
    ↓
Feature Extraction (time-domain, frequency-domain, envelope analysis)
    ↓
Health Index Construction (monotonicity/trendability-based feature selection)
    ↓
PatchTST Encoder + Regression Head (sliding window → scalar RUL)
    ↓
RUL Prediction (seconds)
```

### Dataset

IEEE PHM 2012 Prognostic Challenge (PRONOSTIA platform, FEMTO-ST Institute):
- **6 training bearings** (run-to-failure) under 3 operating conditions
- **11 test bearings** (truncated mid-life) with known actual RULs
- Vibration: horizontal + vertical accelerometers at 25,600 Hz
- Recordings: 2,560 samples every 10 seconds

### Model

- **PatchTSTForRegression** from HuggingFace `transformers`
- Encoder-only architecture, patch-based tokenization of time series
- Lightweight regression head for scalar RUL output
- Three training strategies: frozen backbone, LoRA fine-tuning, full fine-tuning

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd "Predictive Maintainance"

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- transformers ≥ 4.36
- peft, accelerate, tensorboard
- scikit-learn, scipy, pandas, numpy
- matplotlib, seaborn, PyWavelets

## Quick Start

### Run the entire pipeline end-to-end:

```bash
python run_pipeline.py
```

This executes all phases:
1. Data loading & validation
2. Feature extraction (time/frequency/envelope domains)
3. Health Index construction
4. Model training (frozen, LoRA, full fine-tuning + classical baselines)
5. Inference on 11 test bearings
6. PHM 2012 challenge scoring & metrics
7. Report generation

### Skip data preparation (use cached features):

```bash
python run_pipeline.py --skip-data-prep
```

### Run only inference with saved models:

```bash
python run_pipeline.py --skip-data-prep --skip-training
```

### Run unit tests:

```bash
python -m pytest tests/ -v
```

## Project Structure

```
Predictive_Maintainance/
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── src/
│   ├── data_loading.py          # Raw CSV loader with validation
│   ├── feature_extraction.py    # Time/frequency/envelope features
│   ├── health_index.py          # HI construction from top-K features
│   ├── dataset.py               # PyTorch Dataset/DataLoader
│   ├── model.py                 # PatchTST wrapper + classical baselines
│   ├── train.py                 # Fine-tuning loop with early stopping
│   ├── evaluate.py              # PHM scoring + regression metrics + plots
│   └── inference.py             # Test-set inference + RUL trajectories
├── tests/
│   ├── test_data_loading.py     # Data loading & 2560-sample validation
│   ├── test_feature_extraction.py  # Feature correctness tests
│   └── test_scoring.py          # PHM scoring function tests
├── models/                      # Saved model checkpoints
├── data/processed/              # Extracted features (cached)
├── reports/
│   ├── figures/                 # All plots
│   ├── results.md               # Final report
│   └── metrics.json             # All numeric metrics
├── Learning_set/                # Raw training data (6 bearings)
├── Test_set/                    # Raw test data (11 bearings)
├── run_pipeline.py              # Single-command pipeline runner
├── requirements.txt
└── README.md
```

## Metrics

The pipeline computes:

- **Regression metrics**: RMSE, MAE, MAPE, R² (seconds)
- **IEEE PHM 2012 Challenge Score**: Asymmetric scoring penalizing late
  (dangerous) predictions more than early (safe) ones
- **Model comparison table** across all training strategies

### PHM Scoring Function

```
%Er_i = 100 × (ActualRUL - PredRUL) / ActualRUL
A_i = exp(-ln(0.5) × Er/5)   if Er ≤ 0  (late → steep penalty)
A_i = exp(+ln(0.5) × Er/20)  if Er > 0  (early → gentle penalty)
Score = mean(A_i) across 11 test bearings
```

## Citation

Patrick Nectoux, Rafael Gouriveau, Kamal Medjaher, Emmanuel Ramasso,
Brigitte Morello, Nourredine Zerhouni, Christophe Varnier.
*PRONOSTIA: An Experimental Platform for Bearings Accelerated Life Test.*
IEEE International Conference on Prognostics and Health Management, Denver, CO, USA, 2012.

## License

Dataset was publicly available for the IEEE PHM 2012 challenge.
Code is provided as-is for research and educational purposes.
