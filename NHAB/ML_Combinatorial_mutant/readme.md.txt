# ZSaML for NHAB Combinatorial Mutant Prediction

## Overview

This repository contains machine learning workflows for predicting the fitness of NHAB combinatorial mutants. It includes AAindex feature selection, ESM zero-shot fitness score integration, model evaluation, and large-scale sequence prediction.

## Directory Structure

```text
ML_Combinatorial_mutant/
├── Data_for_input/
│   ├── Train_set.csv
│   └── Test_set.csv
├── Data_for_output/
│   ├── aaindex_modelwise_selection/
│   └── esm_greedy_selection/
├── prediction/
│   ├── Combinatorial_mutant_squence.csv
│   ├── Combinatorial_Variants_Based_on_H10Q_T156S_V349N_Scaffold.csv
│   └── prediction_results.csv
├── FFT.py
├── AAindex_model_screening.py
├── AAindex_esm_screening.py
└── prediction.py
```

## Workflow

1. **AAindex feature selection** (`AAindex_model_screening.py`): Selects informative AAindex features for different regression models using five-fold cross-validation.
2. **ESM feature selection** (`AAindex_esm_screening.py`): Greedily integrates ESM-derived zero-shot fitness scores with selected AAindex features.
3. **Combinatorial mutant prediction** (`prediction.py`): Uses a trained model to predict fitness for large sets of mutant sequences in batches.

`FFT.py` provides the AAindex-based sequence encoding functions used by the workflow.

## Model and Evaluation

- **Sequence features:** AAindex descriptors and ESM-derived zero-shot fitness scores.
- **Regression models:** Linear regression, Ridge, Lasso, ElasticNet, PLS, Bayesian Ridge, Huber, KNN, Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, and SVR.
- **Evaluation metrics:** Spearman correlation, R2, RMSE, and cross-validation MSE.
- **Fitness transformation:** Fitness values are transformed using `log1p` during model training and converted back to the original scale for prediction.

## Usage

Run the scripts from the project root directory.

```bash
python AAindex_model_screening.py
python AAindex_esm_screening.py
python prediction.py
```

Run AAindex feature selection before ESM feature selection. The ESM selection script expects the AAindex step-cache files to be available. Prediction requires a trained model and the corresponding selected AAindex features.

## Requirements

Python 3 with NumPy, pandas, SciPy, Matplotlib, scikit-learn, tqdm, and `aaindex`.

## Notes

- Input CSV files should contain the columns required by the scripts, including `fitness` and `mut_seq` for training and testing.
- For new-sequence prediction, the input CSV should contain a `sequence` or `mut_seq` column.
- Update file paths and selected features in the scripts when using different datasets or trained models.