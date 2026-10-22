# NHAB Single-Mutant Feature Selection

PLSR-based feature selection and inference pipeline for single-mutant fitness prediction using **AAindex physicochemical properties** and **ESM-derived zero-shot fitness scores**.

## Overview

This repository provides feature-selection and inference pipelines for predicting the fitness of NHAB single-mutant variants under limited-data conditions.

The workflow integrates two complementary types of sequence representations:

* **AAindex physicochemical features** — sequence descriptors derived from AAindex properties and encoded using Fast Fourier Transform (FFT).
* **ESM-derived zero-shot fitness scores** — evolutionary information extracted from pretrained protein language models (PLMs).
* **Combined AAindex + ESM representations** — integration of physicochemical and evolutionary features through greedy feature selection.
* **PLSR-based prediction** — partial least squares regression (PLSR) models trained using the selected features for single-mutant fitness prediction.

Fitness values are modeled in `log1p` space during model training and prediction. Final predictions are transformed back to the original fitness scale using the inverse `expm1` transformation.

---

## Repository Structure

```text
ML_Single_mutant/
│
├── FFT.py
├── aaindex_plsr_selection.py
├── esm_plsr_selection.py
├── aaindex_ESM_pls.py
├── prediction.py
│
├── Data_for_input/
│   ├── Train_set.csv
│   └── Test_set.csv
│
├──Data_for_output/
│   └──  aaindex_round_cache.pkl
│   └── esm_greedy_results/
│   └──  esm_selection_results/
│
└── prediction/
    ├── best_feats_round1.txt
    ├── best_model_round1.pkl
    ├── single_data.csv
    └── result.csv
    └──Single_mutant_interface_restricted.csv
```

### Scripts

| File                        | Description                                                                                          |
| --------------------------- | ---------------------------------------------------------------------------------------------------- |
| `FFT.py`                    | Encodes protein sequences using AAindex physicochemical properties and Fast Fourier Transform (FFT). |
| `aaindex_plsr_selection.py` | Performs greedy selection of AAindex-derived features using PLS regression.                          |
| `esm_plsr_selection.py`     | Performs greedy selection of ESM-derived zero-shot fitness features.                                 |
| `aaindex_ESM_pls.py`        | Performs combined AAindex + ESM feature selection and evaluates the contribution of ESM features.    |
| `prediction.py`             | Performs inference on novel mutant sequences using saved feature lists and trained PLS models.       |

---

## Input Data

Training and test datasets should be placed in:

```text
Data_for_input/
├── Train_set.csv
└── Test_set.csv
```

Each CSV file should contain the following information:

* `mutated_sequence` — amino acid sequence of the mutant.
* `fitness` — experimentally measured fitness value.
* **16 ESM-derived zero-shot fitness score columns**.

The ESM-derived features include scores obtained from:

* ESM-1v
* ESM-1
* ESM-1b
* ESM-2

The feature column names in the input datasets should remain consistent across the feature-selection and prediction pipelines.

---

## Workflow

The overall workflow is summarized below:

```text
                         Protein sequences
                                │
                ┌───────────────┴───────────────┐
                │                               │
        AAindex physicochemical             ESM zero-shot
              features                         scores
                │                               │
        FFT-based encoding              Greedy feature selection
                │                               │
       Greedy feature selection                 │
                │                               │
                └───────────────┬───────────────┘
                                │
                    AAindex + ESM integration
                                │
                     Greedy feature selection
                                │
                              PLSR
                                │
                       Trained prediction
                                │
                           Inference
                                │
                     Predicted fitness values
```

Feature selection is performed using cross-validated mean squared error (MSE), while the final predictive performance is evaluated on an independent test set using metrics including **R², Spearman's ρ, and RMSE**.

---

## Usage

### 1. AAindex Feature Selection

Run:

```bash
python aaindex_plsr_selection.py
```

This script performs greedy selection of AAindex-derived physicochemical features and trains PLS regression models using the selected feature subsets.

The workflow generates selected features, trained PLS models, predictions, and an intermediate cache file:

```text
Data_for_output/aaindex_round_cache.pkl
```

The cache file is required for the subsequent combined AAindex + ESM feature-selection step.

---

### 2. ESM Feature Selection

Run:

```bash
python esm_plsr_selection.py
```

This script performs greedy forward selection of the 16 ESM-derived zero-shot fitness features.

The selected feature orders and corresponding prediction results are saved under:

```text
Data_for_output/esm_greedy_results/
```

---

### 3. Combined AAindex + ESM Feature Selection

The combined feature-selection step should be performed after completing the AAindex feature-selection step.

Run:

```bash
python aaindex_ESM_pls.py
```

This script integrates the selected AAindex features with ESM-derived zero-shot fitness scores and evaluates their combined predictive performance.

Results are saved under:

```text
Data_for_output/esm_selection_results/
```

The script uses the AAindex feature-selection cache generated by:

```text
aaindex_plsr_selection.py
```

---

### 4. Fitness Prediction / Inference

After model training, `prediction.py` can be used to predict fitness values for novel mutant sequences.

Run:

```bash
python prediction.py
```

The inference pipeline:

1. Loads pre-computed sequence features from:

   ```text
   prediction/single_data.csv
   ```

2. Extracts the required AAindex-FFT features and ESM-derived features according to:

   ```text
   prediction/best_feats_round1.txt
   ```

3. Loads the trained PLSR model:

   ```text
   prediction/best_model_round1.pkl
   ```

4. Generates predictions in the `log1p`-transformed space.

5. Applies the inverse `expm1` transformation to recover predictions on the original fitness scale.

6. Saves the final predictions to:

   ```text
   prediction/result.csv
   ```

---

## Feature Selection Strategy

### AAindex Features

AAindex descriptors are used to represent physicochemical properties of amino acid sequences.

The AAindex feature-selection pipeline:

```text
AAindex descriptors
        │
        ▼
Sequence encoding
        │
        ▼
FFT transformation
        │
        ▼
Greedy feature selection
        │
        ▼
PLSR
```

A greedy forward-selection strategy is used to identify a compact subset of informative AAindex features.

---

### ESM Zero-Shot Features

The ESM-based pipeline evaluates 16 zero-shot fitness scores derived from pretrained protein language models.

The features are sequentially incorporated using greedy forward selection:

```text
16 ESM-derived zero-shot scores
              │
              ▼
      Greedy forward selection
              │
              ▼
             PLSR
              │
              ▼
    Cross-validated evaluation
```

The selected feature subset is determined based on cross-validated predictive performance.

---

### Combined AAindex + ESM Features

The combined pipeline evaluates whether ESM-derived evolutionary information provides additional predictive information beyond the selected AAindex physicochemical features.

The workflow is:

```text
Selected AAindex features
           +
ESM zero-shot fitness scores
           │
           ▼
   Greedy feature selection
           │
           ▼
          PLSR
```

This allows the contribution of evolutionary information to be evaluated while retaining the physicochemical information captured by AAindex descriptors.

---

## Model Training and Evaluation

Fitness values are transformed using:

```text
y_transformed = log1p(y)
```

PLSR models are trained and optimized in the transformed space.

Predictions are converted back to the original scale using:

```text
y_pred = expm1(y_pred_transformed)
```

Feature selection is performed using cross-validation within the training set. The independent test set is reserved for final model evaluation.

Model performance can be assessed using:

* **R²** — coefficient of determination.
* **Spearman's ρ** — rank correlation between predicted and experimental fitness.
* **RMSE** — root mean squared error.

---

## Requirements

The pipeline requires **Python 3** and the following packages:

```text
numpy
pandas
scikit-learn
scipy
matplotlib
tqdm
aaindex
```

Install the required dependencies with:

```bash
pip install numpy pandas scikit-learn scipy matplotlib tqdm aaindex
```

---

## Recommended Execution Order

For a complete feature-selection and prediction workflow, run the scripts in the following order:

```text
1. aaindex_plsr_selection.py
             │
             ▼
2. esm_plsr_selection.py
             │
             ▼
3. aaindex_ESM_pls.py
             │
             ▼
4. prediction.py
```

More specifically:

1. **AAindex feature selection**
   Generate the selected physicochemical features and the AAindex selection cache.

2. **ESM feature selection**
   Evaluate the predictive contribution of ESM-derived zero-shot fitness scores.

3. **AAindex + ESM integration**
   Combine selected AAindex features with ESM-derived features and perform integrated feature selection.

4. **Inference**
   Apply the trained model to novel mutant sequences using the selected feature set.

---

## Output Files

Typical output files and directories include:

```text
Data_for_output/
│
├── aaindex_round_cache.pkl
├── esm_greedy_results/
└── esm_selection_results/
```

The inference pipeline uses:

```text
prediction/
├── best_feats_round1.txt
├── best_model_round1.pkl
├── single_data.csv
└── result.csv
```

### Key inference files

| File                    | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| `best_feats_round1.txt` | Selected feature names used by the trained model.       |
| `best_model_round1.pkl` | Serialized trained PLSR model.                          |
| `single_data.csv`       | Pre-computed features for sequences to be predicted.    |
| `result.csv`            | Predicted fitness values on the original fitness scale. |

---

## Notes

* The three feature-selection scripts are designed to form an integrated workflow but can also be run independently where applicable.
* `aaindex_ESM_pls.py` requires the AAindex selection cache generated by `aaindex_plsr_selection.py`.
* Input feature names must remain consistent between the training data and inference data.
* When using `prediction.py`, ensure that all features required by `best_feats_round1.txt` are present in `prediction/single_data.csv`.
* The trained model and selected feature list should correspond to the same feature-selection procedure and dataset version.
* The final fitness predictions are reported on the original, non-log-transformed scale.

---

## Citation

If you use this code or the associated methodology in your research, please cite the corresponding publication:

```text
[Add publication citation here]
```

---

## License

Please add the appropriate license for this repository.
