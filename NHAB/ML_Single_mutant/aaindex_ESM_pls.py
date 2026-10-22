"""
Combined Feature Selection: AAindex Properties + ESM Embeddings via PLS Regression.

This script performs greedy forward selection of ESM scores
on top of the AAindex features selected in previous rounds to optimize protein 
fitness prediction using Partial Least Squares (PLS) regression.
"""

import argparse
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Headless backend for server/script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from FFT import get_aai_encoding

# Default candidate ESM embedding feature names
DEFAULT_ESM_FEATS = [
    'esm1v_t33_650M_UR90S_1', 'esm1v_t33_650M_UR90S_2', 'esm1v_t33_650M_UR90S_3',
    'esm1v_t33_650M_UR90S_4', 'esm1v_t33_650M_UR90S_5', 'esm2_t33_650M_UR50D',
    'esm2_t36_3B_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t6_8M_UR50D',
    'esm1_t6_43M_UR50S', 'esm1_t34_670M_UR50D', 'esm1_t34_670M_UR50S', 'esm1_t34_670M_UR100',
    'esm1_t12_85M_UR50S', 'esm1b_t33_650M_UR50S'
]


def load_and_preprocess(train_path: Path, test_path: Path):
    """Load train/test datasets and preprocess target fitness values."""
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

    train_seqs = df_train['mutated_sequence'].to_dict()
    test_seqs = df_test['mutated_sequence'].to_dict()

    return df_train, df_test, train_seqs, test_seqs, y_train, y_test, y_train_orig, y_test_orig


def compute_cv_mse(X_aa, df_feats, esm_list, y_train, cv_splits=5):
    """Compute cross-validated MSE and optimal PLS components in log1p space."""
    X_esms = np.hstack([df_feats[[f]].values for f in esm_list]) if esm_list else np.empty((len(df_feats), 0))
    X = np.hstack([X_aa, X_esms])

    max_comp = min(X.shape[1], 11)
    comp_range = list(range(2, max_comp)) if max_comp > 2 else [1]

    grid = GridSearchCV(
        PLSRegression(),
        {'n_components': comp_range},
        cv=cv_splits,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid.fit(X, y_train)
    return -grid.best_score_, grid.best_params_['n_components']


def plot_round_performance(r2_history, round_id, output_dir):
    """Plot publication-quality figures for Greedy ESM performance."""
    plt.figure(figsize=(6, 4.5), dpi=300)
    steps = range(1, len(r2_history) + 1)

    plt.plot(steps, r2_history, marker='o', color='#1f77b4', linewidth=1.8, label='Greedy ESM + AAindex $R^2$')

    plt.xlabel('ESM Feature Addition Step')
    plt.ylabel('Test $R^2$')
    plt.title(f'Feature Selection Performance (Round {round_id})')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save both PNG and vector PDF for publication formatting
    plt.savefig(output_dir / f'fixedAA_round{round_id}_r2.png', dpi=300)
    plt.savefig(output_dir / f'fixedAA_round{round_id}_r2.pdf')
    plt.close()


def run_esm_selection(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df_train, df_test_base, train_seqs, test_seqs, y_train, y_test, y_train_orig, y_test_orig = load_and_preprocess(
        Path(args.train_csv), Path(args.test_csv)
    )

    # 2. Load cached AAindex selections from prior stage
    cache_path = Path(args.cache_file)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found at {cache_path}. Please run stage 1 AAindex selection first.")

    with open(cache_path, 'rb') as f:
        aa_idxs_all, history, _ = pickle.load(f)

    # Parse round range
    rounds_to_run = list(range(args.start_round, args.end_round + 1))

    # 3. Iterative greedy selection of ESM features for each fixed AAindex round
    for rr in rounds_to_run:
        df_test = df_test_base.copy()

        # Retrieve selected AAindices for current round
        aa_idxs = history[rr - 1]['selected']
        X_train_aa = np.array(get_aai_encoding(train_seqs, indices=aa_idxs))
        X_test_aa = np.array(get_aai_encoding(test_seqs, indices=aa_idxs))

        remaining = DEFAULT_ESM_FEATS.copy()
        selected = []
        r2_history = []

        print(f"\n=== Round {rr} Greedy ESM Selection ===")
        for step in range(1, len(DEFAULT_ESM_FEATS) + 1):
            best_mse, best_feat, best_n = np.inf, None, None

            for feat in tqdm(remaining, desc=f"Round {rr} Step {step}"):
                trial = selected + [feat]
                mse, n_comp = compute_cv_mse(X_train_aa, df_train, trial, y_train, cv_splits=args.cv_splits)
                if mse < best_mse:
                    best_mse, best_feat, best_n = mse, feat, n_comp

            print(f" Step {step}: Selected {best_feat} | CV MSE = {best_mse:.4f} | n_components = {best_n}")
            selected.append(best_feat)
            remaining.remove(best_feat)

            # Concatenate AAindex and selected ESM features
            X_tr = np.hstack([X_train_aa] + [df_train[[f]].values for f in selected])
            X_te = np.hstack([X_test_aa] + [df_test[[f]].values for f in selected])

            # Train PLS regression model in log1p space
            pls = PLSRegression(n_components=best_n).fit(X_tr, y_train)
            pred_log = pls.predict(X_te)

            # Transform back to original space and compute test R²
            pred = np.expm1(pred_log)
            r2_history.append(r2_score(y_test_orig, pred))

            # Store predictions for current step
            df_test[f'r{rr}_esm_step{step}'] = pred.flatten()

        print(f" => Round {rr} Selected ESM order: {selected}")

        # Save prediction results and performance plots
        df_test.to_csv(output_dir / f'fixedAA_round{rr}_predictions.csv', index=False)
        plot_round_performance(r2_history, rr, output_dir)

    print(f"\nAnalysis complete. All predictions and figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greedy ESM feature selection on top of fixed AAindex sets.")

    parser.add_argument("--train_csv", type=str, default="./Data_for_input/Train_set.csv", help="Path to training CSV file")
    parser.add_argument("--test_csv", type=str, default="./Data_for_input/Test_set.csv", help="Path to test CSV file")
    parser.add_argument("--cache_file", type=str, default="./Data_for_output/aaindex_round_cache.pkl", help="Path to stage 1 cache file")
    parser.add_argument("--output_dir", type=str, default="./Data_for_output/esm_selection_results", help="Directory to save combined results")

    parser.add_argument("--start_round", type=int, default=1, help="Start round for fixed AAindex analysis")
    parser.add_argument("--end_round", type=int, default=20, help="End round for fixed AAindex analysis")
    parser.add_argument("--cv_splits", type=int, default=5, help="Number of cross-validation folds")

    args = parser.parse_args()
    run_esm_selection(args)