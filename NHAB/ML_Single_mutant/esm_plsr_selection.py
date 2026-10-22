"""
Standalone ESM Feature Selection via PLS Regression.

This script performs Sequential Forward Selection (Greedy Search) on ESM 
scores to optimize protein fitness prediction. It iteratively 
adds the ESM feature that minimizes Cross-Validated Mean Squared Error (CV MSE).
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # Headless backend for server/script execution
import matplotlib.pyplot as plt

# --------------------
# Configuration
# --------------------
DEFAULT_ESM_FEATS = [
    'esm1v_t33_650M_UR90S_1', 'esm1v_t33_650M_UR90S_2', 'esm1v_t33_650M_UR90S_3',
    'esm1v_t33_650M_UR90S_4', 'esm1v_t33_650M_UR90S_5', 'esm2_t33_650M_UR50D',
    'esm2_t36_3B_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t6_8M_UR50D',
    'esm1_t6_43M_UR50S', 'esm1_t34_670M_UR50D', 'esm1_t34_670M_UR50S', 'esm1_t34_670M_UR100',
    'esm1_t12_85M_UR50S', 'esm1b_t33_650M_UR50S'
]


def load_and_preprocess(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/test datasets and preprocess target fitness values."""
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing data files. Check paths:\nTrain: {train_path}\nTest: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    def convert_fitness(df):
        if df['fitness'].dtype == object:
            df['fitness'] = df['fitness'].str.rstrip('%').astype(float)

    convert_fitness(df_train)
    convert_fitness(df_test)

    # Preserve original fitness values for final R² metric evaluation
    y_train_orig = df_train['fitness'].values
    y_test_orig = df_test['fitness'].values

    # Perform model fitting and validation in log1p space
    y_train = np.log1p(y_train_orig)
    y_test = np.log1p(y_test_orig)

    return df_train, df_test, y_train, y_test, y_train_orig, y_test_orig


def compute_cv_mse(df_train: pd.DataFrame, y_train: np.ndarray, esm_list: List[str], cv_splits: int) -> Tuple[float, int]:
    """
    Construct training matrix with selected ESM features and perform grid search
    to find the optimal number of PLS components.
    
    Returns:
        Tuple containing (best_cv_mse, best_n_components)
    """
    X = df_train[esm_list].values
    
    # Cap maximum components to feature dimension or 10 to prevent overfitting
    max_comp = min(X.shape[1], 10)
    param_grid = {'n_components': range(1, max_comp + 1)}
    
    grid = GridSearchCV(
        PLSRegression(),
        param_grid,
        cv=cv_splits,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid.fit(X, y_train)
    
    return -grid.best_score_, grid.best_params_['n_components']


def plot_r2_progress(selection_history: List[dict], output_dir: Path, current_step: int):
    """Generate publication-ready plot of R² progression during feature selection."""
    plt.figure(figsize=(6, 4.5), dpi=300)
    
    steps = [h['step'] for h in selection_history]
    test_r2s = [h['test_r2'] for h in selection_history]
    
    plt.plot(steps, test_r2s, marker='o', color='#1f77b4', linewidth=1.8)
    plt.xlabel('ESM Selection Step')
    plt.ylabel('Test $R^2$')
    plt.title('Greedy ESM Feature Selection Progress')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save both PNG and vector PDF
    plt.savefig(output_dir / f'r2_progress_step{current_step}.png', dpi=300)
    plt.savefig(output_dir / f'r2_progress_step{current_step}.pdf')
    plt.close()


def run_greedy_selection(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.train_csv} ...")
    df_train, df_test, y_train, y_test, y_train_orig, y_test_orig = load_and_preprocess(
        Path(args.train_csv), Path(args.test_csv)
    )

    remaining = DEFAULT_ESM_FEATS.copy()
    selected = []
    selection_history = []

    print("\n=== Start Greedy ESM Selection ===")
    for step in range(1, len(DEFAULT_ESM_FEATS) + 1):
        best_mse, best_feat, best_n = np.inf, None, None

        for feat in tqdm(remaining, desc=f"Step {step}"):
            trial = selected + [feat]
            mse, n_comp = compute_cv_mse(df_train, y_train, trial, args.cv_splits)
            
            if mse < best_mse:
                best_mse, best_feat, best_n = mse, feat, n_comp

        selected.append(best_feat)
        remaining.remove(best_feat)
        
        # Evaluate on Test Set
        X_train = df_train[selected].values
        X_test  = df_test[selected].values

        pls = PLSRegression(n_components=best_n).fit(X_train, y_train)
        pred_log = pls.predict(X_test)
        pred = np.expm1(pred_log)
        r2 = r2_score(y_test_orig, pred)

        selection_history.append({
            'step': step,
            'selected_feat': best_feat,
            'cv_mse': best_mse,
            'n_components': best_n,
            'test_r2': r2
        })
        
        print(f" -> Selected: {best_feat} | CV MSE: {best_mse:.4f} | PLS Comps: {best_n} | Test R²: {r2:.4f}")

        # Save iterative outputs
        df_test[f'step{step}_pred'] = pred.flatten()
        
        pd.DataFrame(selection_history).to_csv(output_dir / 'esm_selection_history.csv', index=False)
        df_test.to_csv(output_dir / f'predictions_step{step}.csv', index=False)
        
        with open(output_dir / f'pls_step{step}.pkl', 'wb') as f:
            pickle.dump(pls, f)

        plot_r2_progress(selection_history, output_dir, step)

    print("\n=== Greedy ESM Selection Complete ===")
    print(f"Results saved to: {output_dir.resolve()}")
    print("Final Selected ESM Sequence:")
    for h in selection_history:
        print(f"  Step {h['step']}: {h['selected_feat']} (R² = {h['test_r2']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Greedy Forward Selection for ESM Features")
    
    parser.add_argument("--train_csv", type=str, default="./Data_for_input/Train_set.csv", help="Path to training CSV file")
    parser.add_argument("--test_csv", type=str, default="./Data_for_input/Test_set.csv", help="Path to test CSV file")
    parser.add_argument("--output_dir", type=str, default="./Data_for_output/esm_greedy_results", help="Directory to save models, predictions, and plots")
    parser.add_argument("--cv_splits", type=int, default=5, help="Number of folds for cross-validation")
    
    args = parser.parse_args()
    run_greedy_selection(args)