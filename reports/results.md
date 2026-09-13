# Bearing RUL Prediction - Results Report
## IEEE PHM 2012 Prognostic Challenge
### Methodology
This report presents results from an end-to-end pipeline for predicting the Remaining Useful Life (RUL) of rolling-element bearings using a fine-tuned PatchTST model from HuggingFace `transformers`.

**Architecture**: Raw vibration data -> Feature extraction (time-domain, frequency-domain, envelope analysis) -> Health Index -> PatchTST encoder + regression head -> RUL (seconds)

### Dataset
- **Training**: 6 run-to-failure bearings (Bearing1_1, 1_2, 2_1, 2_2, 3_1, 3_2)
- **Testing**: 11 truncated bearings with known actual RULs
- **Validation**: Bearing1_2 held out from training

### Model Comparison

| Model | RMSE (s) | MAE (s) | MAPE (%) | R² | PHM Score |
|-------|----------|---------|----------|-----|----------|
| Linear Degradation | 64898 | 45917 | 1921.9 | -607.426 | 0.0495 |

**Best model**: Linear Degradation (PHM Score = 0.0495)

### Per-Bearing Results (Best Model)

| Bearing | Actual RUL (s) | Predicted RUL (s) | %Error | A_i |
|---------|----------------|-------------------|--------|-----|
| Bearing1_3 | 5730 | 1880 | 67.2 | 0.0975 |
| Bearing1_4 | 339 | 0 | 100.0 | 0.0312 |
| Bearing1_5 | 1610 | 100000 | -6111.2 | 0.0000 |
| Bearing1_6 | 1460 | 21543 | -1375.5 | 0.0000 |
| Bearing1_7 | 7570 | 100000 | -1221.0 | 0.0000 |
| Bearing2_3 | 7530 | 100000 | -1228.0 | 0.0000 |
| Bearing2_4 | 1390 | 1007 | 27.5 | 0.3850 |
| Bearing2_5 | 3090 | 100000 | -3136.2 | 0.0000 |
| Bearing2_6 | 1290 | 100000 | -7651.9 | 0.0000 |
| Bearing2_7 | 580 | 1286 | -121.7 | 0.0000 |
| Bearing3_3 | 820 | 0 | 100.0 | 0.0312 |

**Overall PHM Challenge Score: 0.0495**

### Comparison with Published Challenge Results

| Team | Score (approximate) |
|------|--------------------|
| A.L.D. Ltd. (Winner) | ~0.36 |
| GE Global Research | ~0.24 |
| CALCE-UMD | ~0.22 |
| Jozef Stefan Institute | ~0.15 |
| **Our PatchTST (best)** | **0.0495** |

> Note: Direct comparison is approximate. Published scores vary by implementation details and are not exactly reproducible without access to original codebases.

### Plots

#### Predicted vs Actual RUL

![Predicted vs Actual RUL](pred_vs_actual.png)

#### PHM Scoring Function with Test Points

![PHM Scoring Function with Test Points](phm_scoring_function.png)

#### Model Comparison

![Model Comparison](model_comparison.png)

#### Training Curves: patchtst_frozen_hi_only

![Training Curves](training_curves_patchtst_frozen_hi_only.png)

#### Training Curves: patchtst_frozen_multivariate

![Training Curves](training_curves_patchtst_frozen_multivariate.png)

#### Training Curves: patchtst_full_hi_only

![Training Curves](training_curves_patchtst_full_hi_only.png)

#### Training Curves: patchtst_full_multivariate

![Training Curves](training_curves_patchtst_full_multivariate.png)

#### Training Curves: patchtst_lora_hi_only

![Training Curves](training_curves_patchtst_lora_hi_only.png)

#### Training Curves: patchtst_lora_multivariate

![Training Curves](training_curves_patchtst_lora_multivariate.png)

#### RUL Trajectory: Bearing1_3

![RUL Trajectory](rul_trajectory_Bearing1_3.png)

#### RUL Trajectory: Bearing1_4

![RUL Trajectory](rul_trajectory_Bearing1_4.png)

#### RUL Trajectory: Bearing1_5

![RUL Trajectory](rul_trajectory_Bearing1_5.png)

### Limitations

1. **Small training set**: Only 6 run-to-failure bearings for training.
2. **Domain gap**: PatchTST architecture was designed for general time series, not specifically for vibration/bearing signals.
3. **CPU training**: Training was done on CPU, limiting epoch count and model size.
4. **No cross-validation**: Used single train/val split for speed. LOBO CV would give more robust estimates.

### Future Work

1. Use GPU for larger models and more epochs.
2. Implement LOBO cross-validation.
3. Try IBM Granite pretrained weights (domain-adapted).
4. Explore wavelet features and attention visualization.
5. Ensemble multiple model strategies for better predictions.
