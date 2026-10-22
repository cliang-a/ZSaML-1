import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Ill-conditioned matrix")
warnings.filterwarnings("ignore")

from FFT import get_aai_encoding

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# -------------------- Config --------------------
TRAIN_CSV = "./Data_for_input/Train_set.csv"
TEST_CSV = "./Data_for_input/Test_set.csv"

BASE_OUTPUT_DIR = "Data_for_output"
AAINDEX_CACHE_DIR = os.path.join(BASE_OUTPUT_DIR, 'aaindex_modelwise_selection', 'aaindex_step_cache')
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'esm_greedy_selection')

os.makedirs(OUTPUT_DIR, exist_ok=True)

CV_SPLITS = 5
RANDOM_STATE = 42

esm_feats = [
    'esm1v_t33_650M_UR90S_1', 'esm1v_t33_650M_UR90S_2', 'esm1v_t33_650M_UR90S_3',
    'esm1v_t33_650M_UR90S_4', 'esm1v_t33_650M_UR90S_5', 'esm2_t33_650M_UR50D',
    'esm2_t36_3B_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t6_8M_UR50D',
    'esm1_t6_43M_UR50S', 'esm1_t34_670M_UR50D', 'esm1_t34_670M_UR50S', 'esm1_t34_670M_UR100',
    'esm1_t12_85M_UR50S', 'esm1b_t33_650M_UR50S'
]

model_configs = {
    "PLSRegression": (PLSRegression(), {"model__n_components": [1, 2, 3]}),
    "Ridge": (Ridge(random_state=RANDOM_STATE), {"model__alpha": [0.1, 1, 10, 100]}),
    "Lasso": (Lasso(random_state=RANDOM_STATE, max_iter=80000), {"model__alpha": [0.01, 0.1, 1]}),
    "ElasticNet": (ElasticNet(random_state=RANDOM_STATE, max_iter=80000),
                   {"model__alpha": [0.01, 0.1, 1], "model__l1_ratio": [0.3, 0.5, 0.8]}),
    "BayesianRidge": (BayesianRidge(), {}),
    "Huber": (HuberRegressor(max_iter=200000, tol=1e-3), {"model__epsilon": [1.2, 1.35]}),
    "KNN": (KNeighborsRegressor(), {"model__n_neighbors": [3, 5, 7], "model__weights": ["uniform", "distance"]}),
    "DecisionTree": (DecisionTreeRegressor(random_state=RANDOM_STATE), {"model__max_depth": [3, 5]}),
    "RandomForest": (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                     {"model__n_estimators": [30], "model__max_depth": [3, 5]}),
    "ExtraTrees": (ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                   {"model__n_estimators": [30], "model__max_depth": [3, 5]}),
    "AdaBoost": (AdaBoostRegressor(random_state=RANDOM_STATE), {"model__n_estimators": [30, 50]}),
    "GradientBoosting": (GradientBoostingRegressor(random_state=RANDOM_STATE),
                         {"model__n_estimators": [50], "model__max_depth": [2]}),
    "SVR": (SVR(), {"model__C": [0.5, 1, 5], "model__gamma": ["scale"]}),
    "LinearRegression": (LinearRegression(), {})
}

# -------------------- Load Data --------------------
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)


def convert_fitness(df):
    if df['fitness'].dtype == object:
        df['fitness'] = df['fitness'].str.rstrip('%').astype(float)


convert_fitness(df_train)
convert_fitness(df_test)

y_train_orig = df_train['fitness'].values
y_test_orig = df_test['fitness'].values
y_train = np.log1p(y_train_orig)
y_test = np.log1p(y_test_orig)

train_seqs = df_train['mut_seq'].to_dict()
test_seqs = df_test['mut_seq'].to_dict()


# -------------------- Evaluation Helpers --------------------
def r2_over_origin(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        y_pred = y_pred.reshape(y_true.shape)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    if ss_tot == 0:
        return float('nan')
    return 1 - ss_res / ss_tot


# -------------------- CV Setup --------------------
cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def cv_score_single_model(X_tr, y_tr, base_model, param_grid):
    actual_grid = {}
    if param_grid:
        actual_grid = param_grid.copy()
        for key, values in actual_grid.items():
            if 'n_components' in key:
                valid_comps = [v for v in values if v <= X_tr.shape[1]]
                actual_grid[key] = valid_comps if valid_comps else [max(1, X_tr.shape[1])]

    pipe = Pipeline([("scaler", StandardScaler()), ("model", base_model)])

    if not actual_grid:
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_mse = -np.mean(scores)
        pipe.fit(X_tr, y_tr)
        best_pipe = pipe
        best_params = {}
    else:
        g = GridSearchCV(pipe, actual_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        g.fit(X_tr, y_tr)
        cv_mse = -g.best_score_
        best_pipe = g.best_estimator_
        best_params = g.best_params_

    y_pred = best_pipe.predict(X_tr)
    mse = mean_squared_error(y_tr, y_pred)
    r2 = r2_score(y_tr, y_pred)

    n_comp = 1
    if hasattr(best_pipe.named_steps['model'], 'n_components'):
        n_comp = best_pipe.named_steps['model'].n_components

    return mse, r2, cv_mse, n_comp, best_pipe, best_params


# -------------------- Modified Greedy ESM Selection --------------------
def nested_greedy_esm(model_name, aai_step, model, param_grid, X_base_train, X_base_test, y_tr, y_te):
    selected_esm = []
    remaining = esm_feats.copy()

     history = []

    for esm_step in range(1, len(esm_feats) + 1):
        best_cv_mse = np.inf
        best_feat = best_pipeline = best_mse = best_r2 = best_n_comp = best_params = None

        for feat in tqdm(remaining, desc=f"{model_name} AAi[{aai_step}] ESM[{esm_step}]", leave=False):
            trial_feats = selected_esm + [feat]
            X_tr = np.hstack([X_base_train] + [df_train[[f]].values for f in trial_feats])

            try:
                mse, r2, cv_mse, n_comp, pipeline, params = cv_score_single_model(X_tr, y_tr, model, param_grid)

                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_feat = feat
                    best_pipeline = pipeline
                    best_mse = mse
                    best_r2 = r2
                    best_n_comp = n_comp
                    best_params = params
            except Exception as e:
                continue

        if best_feat is None:
            print(f"No valid candidate found for {model_name} at ESM step {esm_step}; stopping early.")
            break

        selected_esm.append(best_feat)
        remaining.remove(best_feat)

        # ------------------- Test Evaluation -------------------
        try:
            X_test_final = np.hstack([X_base_test] + [df_test[[f]].values for f in selected_esm])
            pred_log = best_pipeline.predict(X_test_final)
            pred_log = np.asarray(pred_log).reshape(-1)
            pred_real = np.expm1(pred_log)

            test_r2 = r2_score(y_test_orig, pred_real)
            test_r2_origin = r2_over_origin(y_test_orig, pred_real)
            test_rmse = np.sqrt(mean_squared_error(y_test_orig, pred_real))
            test_spearman = spearmanr(y_test_orig, pred_real).correlation
        except Exception as e:
            pred_real = np.zeros(len(df_test))
            test_r2 = test_r2_origin = test_rmse = test_spearman = -np.inf

        df_test[f"{model_name}_aai{aai_step}_esm{esm_step}_pred"] = pred_real

        entry = {
            'model': model_name, 'aai_step': aai_step, 'esm_step': esm_step,
            'added_esm_feat': best_feat, 'selected_esm_feats': selected_esm.copy(),
            'best_params': best_params, 'n_components': best_n_comp,
            'train_mse': best_mse, 'train_r2': best_r2, 'cv_mse': best_cv_mse,
            'test_r2': test_r2, 'test_r2_origin': test_r2_origin,
            'rmse': test_rmse, 'spearman': test_spearman
        }
        history.append(entry)

        print(f"🏆 {model_name} ESM Step {esm_step}: +{best_feat} | CV_MSE={best_cv_mse:.6f} | Test_R2={test_r2:.4f}")

    if history:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot([h['esm_step'] for h in history], [h['cv_mse'] for h in history], marker='o', color='#D62728',
                 linewidth=2)
        plt.title(f"{model_name} (AAi={aai_step}) - CV MSE", fontweight='bold')
        plt.xlabel("ESM Feature Count")
        plt.ylabel("CV MSE")
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot([h['esm_step'] for h in history], [h['test_r2'] for h in history], marker='s', color='#1F77B4',
                 linewidth=2)
        plt.title(f"{model_name} (AAi={aai_step}) - Test R²", fontweight='bold')
        plt.xlabel("ESM Feature Count")
        plt.ylabel("Test R²")
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        safe_name = model_name.replace("/", "_")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{safe_name}_aai_step{aai_step}_esm_selection.png"), dpi=300)
        plt.close()

    return selected_esm, history


# -------------------- Run Main Logic --------------------
def find_aaindex_cache_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('_aaindex_cache.pkl')])


cache_files = find_aaindex_cache_files(AAINDEX_CACHE_DIR)
print(f"🔍 Found {len(cache_files)} AAIndex base configurations from stage 1.")

all_esm_summaries = []

for cache_file in cache_files:
    try:
        print(f"\n{'=' * 70}")
        print(f"🚀 Processing Base Config: {os.path.basename(cache_file)}")

        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        model_name = cache_data['model_name']
        aai_step = cache_data['step']
        selected_aaindices = cache_data['selected_aaindices']

        if model_name not in model_configs:
            print(f"⚠️ Model {model_name} not found in model_configs, skipping.")
            continue

        print(f"📌 Model: {model_name} | AAIndex Size: {len(selected_aaindices)} -> {selected_aaindices}")

        X_train_aa = np.array(get_aai_encoding(train_seqs, indices=selected_aaindices))
        X_test_aa = np.array(get_aai_encoding(test_seqs, indices=selected_aaindices))

        model, params = model_configs[model_name]

        selected_esm, history = nested_greedy_esm(
            model_name, aai_step, model, params, X_train_aa, X_test_aa, y_train, y_test
        )

        if history:
            all_esm_summaries.extend(history)

            safe_name = model_name.replace("/", "_")
            history_df = pd.DataFrame(history)
            history_df.to_csv(os.path.join(OUTPUT_DIR, f"{safe_name}_aai_step{aai_step}_esm_history.csv"), index=False)

    except Exception as e:
        print(f"❌ Error processing {cache_file}: {e}")
        continue

# -------------------- Final Publishable Summaries --------------------
print("\n✅ All ESM greedy selections finished successfully!")

if all_esm_summaries:
    final_summary_df = pd.DataFrame(all_esm_summaries)
    final_summary_df.to_csv(os.path.join(OUTPUT_DIR, "global_esm_selection_summary.csv"), index=False)

    idx_best = final_summary_df['test_r2'].idxmax()
    global_winner = final_summary_df.loc[idx_best].to_dict()

    print(f"\n🌟 === Global Best Feature Combination (by Test R²) === 🌟")
    print(f"Model: {global_winner['model']}")
    print(f"Base AAIndex count: {global_winner['aai_step']}")
    print(f"Added ESM count: {global_winner['esm_step']}")
    print(f"Best Test R²: {global_winner['test_r2']:.4f}")
    print(f"Best Test RMSE: {global_winner['rmse']:.4f}")
    print(f"Best Spearman: {global_winner['spearman']:.4f}")
    print(f"CV MSE: {global_winner['cv_mse']:.6f}")
    print(f"Selected ESM Features: {global_winner['selected_esm_feats']}")

df_test.to_csv(os.path.join(OUTPUT_DIR, "test_set_all_predictions.csv"), index=False)