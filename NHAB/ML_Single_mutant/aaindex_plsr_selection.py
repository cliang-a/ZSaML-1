"""
Iterative AAindex Feature Selection using PLS Regression.

This script performs forward selection of AAindex physical-chemical properties 
to optimize fitness prediction via Partial Least Squares (PLS) regression.
"""

import argparse
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aaindex import aaindex1
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from FFT import get_aai_encoding


def load_and_preprocess(train_path: Path, test_path: Path):
    """Load the data and perform percentage cleaning and log1p transformation on Fitness."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    def convert_fitness(df):
        if df['fitness'].dtype == object:
            df['fitness'] = df['fitness'].str.rstrip('%').astype(float)

    convert_fitness(df_train)
    convert_fitness(df_test)

    # Log1p transformation
    y_train_orig = df_train['fitness'].values
    y_test_orig = df_test['fitness'].values
    y_train = np.log1p(y_train_orig)
    y_test = np.log1p(y_test_orig)

    train_seqs = df_train['mutated_sequence'].to_dict()
    test_seqs = df_test['mutated_sequence'].to_dict()

    return train_seqs, test_seqs, y_train, y_test, y_train_orig, y_test_orig


def compute_cv_mse(indices, seq_dict, y, cv_splits=5):
    """Calculate cross-validated MSE and the optimal number of principal components."""
    X = np.array(get_aai_encoding(seq_dict, indices=indices))
    max_comp = min(X.shape[1], 11)
    comp_range = list(range(2, max_comp)) if max_comp > 2 else [1]

    grid = GridSearchCV(
        PLSRegression(),
        {'n_components': comp_range},
        cv=cv_splits,  
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid.fit(X, y)
    return -grid.best_score_, grid.best_params_['n_components']


def run_selection(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    train_seqs, test_seqs, y_train, y_test, y_train_orig, y_test_orig = load_and_preprocess(
        Path(args.train_csv), Path(args.test_csv)
    )

    # 2. Read or initialize cache
    cache_path = output_dir / args.cache_file
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            selected_indices, history, r2_history = pickle.load(f)
        start_round = len(history) + 1
        print(f"Resuming from round {start_round}")
    else:
        selected_indices = []
        history = []
        r2_history = []
        start_round = 1

    all_indices = aaindex1.record_codes()
    remaining = [idx for idx in all_indices if idx not in selected_indices]

    # 3. Greedy forward selection in each round
    for round_id in range(start_round, args.n_rounds + 1):
        best_mse, best_idx, best_n = float('inf'), None, None

        for idx in tqdm(remaining, desc=f"Round {round_id} search"):
            trial = selected_indices + [idx]
            mse, n_comp = compute_cv_mse(trial, train_seqs, y_train, cv_splits=args.cv_splits)
            if mse < best_mse:
                best_mse, best_idx, best_n = mse, idx, n_comp

        selected_indices.append(best_idx)
        remaining.remove(best_idx)
        history.append({
            'round': round_id, 
            'cv_mse': best_mse, 
            'n_components': best_n, 
            'selected': list(selected_indices)
        })
        print(f"Round {round_id}: selected {best_idx}, CV_MSE={best_mse:.4f}, n_comp={best_n}")

        # Train and evaluate the interim model for the current round
        X_train = np.array(get_aai_encoding(train_seqs, indices=selected_indices))
        X_test = np.array(get_aai_encoding(test_seqs, indices=selected_indices))
        
        pls = PLSRegression(n_components=best_n)
        pls.fit(X_train, y_train)
        preds_log = pls.predict(X_test)
        preds = np.expm1(preds_log)

        r2 = r2_score(y_test_orig, preds)
        r2_history.append(r2)
        print(f"Round {round_id}: Test R2={r2:.4f}")

        # Save the prediction results of this round for the test set
        results_df = pd.DataFrame({
            'mutant_sequence': list(test_seqs.values()),
            'actual_fitness': y_test_orig,
            'predicted_fitness': preds.flatten()
        })
        results_df.to_csv(output_dir / f'test_results_round_{round_id}.csv', index=False)

        # Update cache and save intermediate models
        with open(cache_path, 'wb') as f:
            pickle.dump((selected_indices, history, r2_history), f)
        with open(output_dir / f'pls_round_{round_id}_n{best_n}.pkl', 'wb') as f:
            pickle.dump(pls, f)

    # 4. Plot publication-quality figures
    plot_metrics(history, r2_history, output_dir)


def plot_metrics(history, r2_history, output_dir):
    """Plot and save high-resolution results (including PNG and vector PDF)."""
    rounds = [h['round'] for h in history]
    mse_vals = [h['cv_mse'] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    # Left plot: CV MSE
    axes[0].plot(rounds, mse_vals, marker='o', color='#1f77b4', linewidth=1.8, markersize=5)
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('CV MSE')
    axes[0].set_title('CV MSE per Round')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Right plot: Test R²
    axes[1].plot(rounds, r2_history, marker='o', color='#2ca02c', linewidth=1.8, markersize=5)
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Test $R^2$')
    axes[1].set_title('Test $R^2$ per Round')
    axes[1].set_ylim(min(0, min(r2_history)), 1.0)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'aaindex_selection_performance.png', dpi=300)
    plt.savefig(output_dir / 'aaindex_selection_performance.pdf')  # Save vector graphic for publication formatting
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AAindex feature selection pipeline for protein fitness.")
    
    # Defaults have been updated to match your local project structure
    parser.add_argument("--train_csv", type=str, default="./Data_for_input/Train_set.csv", help="Path to training CSV file")
    parser.add_argument("--test_csv", type=str, default="./Data_for_input/Test_set.csv", help="Path to test CSV file")
    parser.add_argument("--output_dir", type=str, default="./Data_for_output", help="Directory to store models and results")
    
    parser.add_argument("--cache_file", type=str, default="aaindex_round_cache.pkl", help="Cache file name")
    parser.add_argument("--n_rounds", type=int, default=20, help="Number of selection rounds")
    parser.add_argument("--cv_splits", type=int, default=5, help="Number of cross-validation folds")

    args = parser.parse_args()
    run_selection(args)