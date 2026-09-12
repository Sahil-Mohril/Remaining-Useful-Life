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
| PatchTST Frozen (HI) | 2747 | 2343 | 183.9 | -0.090 | 0.1017 |
| PatchTST Frozen (Multi) | 2859 | 2588 | 239.7 | -0.181 | 0.1174 |
| PatchTST Full FT (HI) | 4034 | 3508 | 397.4 | -1.350 | 0.2012 |
| PatchTST Full FT (Multi) | 3275 | 2461 | 159.2 | -0.549 | 0.2055 |
| PatchTST LoRA (HI) | 3883 | 2855 | 100.0 | -1.178 | 0.0312 |
| PatchTST LoRA (Multi) | 3882 | 2855 | 100.0 | -1.177 | 0.0313 |
| Random Forest | 5158 | 3915 | 361.7 | -2.844 | 0.0452 |
| SVR | 6640 | 5310 | 485.1 | -5.369 | 0.0354 |

**Best model**: PatchTST Full FT (Multi) (PHM Score = 0.2055)

### Per-Bearing Results (Best Model)

| Bearing | Actual RUL (s) | Predicted RUL (s) | %Error | A_i |
|---------|----------------|-------------------|--------|-----|
| Bearing1_3 | 5730 | 4620 | 19.4 | 0.5109 |
| Bearing1_4 | 339 | 360 | -6.2 | 0.4206 |
| Bearing1_5 | 1610 | 5177 | -221.5 | 0.0000 |
| Bearing1_6 | 1460 | 8558 | -486.2 | 0.0000 |
| Bearing1_7 | 7570 | 3598 | 52.5 | 0.1623 |
| Bearing2_3 | 7530 | 4747 | 37.0 | 0.2778 |
| Bearing2_4 | 1390 | 1495 | -7.5 | 0.3517 |
| Bearing2_5 | 3090 | 2529 | 18.1 | 0.5330 |
| Bearing2_6 | 1290 | 5870 | -355.0 | 0.0000 |
| Bearing2_7 | 580 | 3526 | -507.9 | 0.0000 |
| Bearing3_3 | 820 | 1150 | -40.2 | 0.0038 |

**Overall PHM Challenge Score: 0.2055**

### Comparison with Published Challenge Results

| Team | Score (approximate) |
|------|--------------------|
| A.L.D. Ltd. (Winner) | ~0.36 |
| GE Global Research | ~0.24 |
| CALCE-UMD | ~0.22 |
| Jozef Stefan Institute | ~0.15 |
| **Our PatchTST (best)** | **0.2055** |

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
