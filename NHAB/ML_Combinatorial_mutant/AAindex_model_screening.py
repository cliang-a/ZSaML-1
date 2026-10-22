import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Ill-conditioned matrix")

from aaindex import aaindex1
from FFT import get_aai_encoding

# MODELS
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# --------------------
# CONFIG
# --------------------
TRAIN_CSV = "./Data_for_input/Train_set.csv"
TEST_CSV = "./Data_for_input/Test_set.csv"

BASE_OUTPUT_DIR = "Data_for_output"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'aaindex_modelwise_selection')

N_SELECTION_STEPS = 8
CV_SPLITS = 5
RANDOM_STATE = 42

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

AAINDEX_STEP_CACHE_DIR = os.path.join(OUTPUT_DIR, 'aaindex_step_cache')
os.makedirs(AAINDEX_STEP_CACHE_DIR, exist_ok=True)

# --------------------
# Model pool (fixed)
# --------------------
model_configs = {
    "PLSRegression": (
        PLSRegression(),
        {"model__n_components": [1, 2, 3]}
    ),
    "Ridge": (
        Ridge(random_state=RANDOM_STATE),
        {"model__alpha": [0.1, 1, 10, 100]}
    ),
    "Lasso": (
        Lasso(random_state=RANDOM_STATE, max_iter=80000),
        {"model__alpha": [0.01, 0.1, 1]}
    ),
    "ElasticNet": (
        ElasticNet(random_state=RANDOM_STATE, max_iter=80000),
        {"model__alpha": [0.01, 0.1, 1], "model__l1_ratio": [0.3, 0.5, 0.8]}
    ),
    "BayesianRidge": (BayesianRidge(), {}),
    "Huber": (
        HuberRegressor(max_iter=200000, tol=1e-3),
        {"model__epsilon": [1.2, 1.35]}
    ),
    "KNN": (
        KNeighborsRegressor(),
        {"model__n_neighbors": [3, 5, 7], "model__weights": ["uniform", "distance"]}
    ),
    "DecisionTree": (
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {"model__max_depth": [3, 5]}
    ),
    "RandomForest": (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [30], "model__max_depth": [3, 5]}
    ),
    "ExtraTrees": (
        ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [30], "model__max_depth": [3, 5]}
    ),
    "AdaBoost": (
        AdaBoostRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": [30, 50]}
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": [50], "model__max_depth": [2]}
    ),
    "SVR": (
        SVR(),
        {"model__C": [0.5, 1, 5], "model__gamma": ["scale"]}
    ),
    "LinearRegression": (LinearRegression(), {})
}

# --------------------
# LOAD DATA
# --------------------
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


# --------------------
# Helper: r2_over_origin
# --------------------
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


# --------------------
# CV function for single model
# --------------------
cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def cv_score_single_model(indices, base_model, param_grid):
    X = np.array(get_aai_encoding(train_seqs, indices=indices))

    actual_grid = {}
    if param_grid:
        actual_grid = param_grid.copy()
        for key, values in actual_grid.items():
            if 'n_components' in key:
                valid_comps = [v for v in values if v <= X.shape[1]]
                actual_grid[key] = valid_comps if valid_comps else [max(1, X.shape[1])]

    pipe = Pipeline([("scaler", StandardScaler()), ("model", base_model)])

    if not actual_grid:
        scores = cross_val_score(pipe, X, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_mse = -np.mean(scores)
        pipe.fit(X, y_train)
        best_pipe = pipe
        best_params = {}
    else:
        g = GridSearchCV(pipe, actual_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        g.fit(X, y_train)
        cv_mse = -g.best_score_
        best_pipe = g.best_estimator_
        best_params = g.best_params_

    y_pred = best_pipe.predict(X)
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    n_comp = 1
    if hasattr(best_pipe.named_steps['model'], 'n_components'):
        n_comp = best_pipe.named_steps['model'].n_components

    return mse, r2, cv_mse, n_comp, best_pipe, best_params


# --------------------
# Cache Generation
# --------------------
def create_step_cache_file(model_name, step, selected_indices, X_train_aa, X_test_aa,
                           best_pipeline, best_params, performance_metrics):
    safe_name = model_name.replace("/", "_")
    cache_filename = f"{safe_name}_step{step}_aaindex_cache.pkl"
    cache_filepath = os.path.join(AAINDEX_STEP_CACHE_DIR, cache_filename)

    cache_data = {
        'model_name': model_name,
        'step': step,
        'selected_aaindices': selected_indices.copy(),
        'X_train_aa_shape': X_train_aa.shape,
        'X_test_aa_shape': X_test_aa.shape,
        'best_params': best_params,
        'performance_metrics': performance_metrics,
        'timestamp': pd.Timestamp.now()
    }

    with open(cache_filepath, 'wb') as f:
        pickle.dump(cache_data, f)

    return cache_filepath


# --------------------
# Main Feature Selection Loop
# --------------------
all_models_summary = []
per_model_histories = {}
all_aaindex_codes = aaindex1.record_codes()
all_step_cache_files = []

for model_name, (base_model, param_grid) in model_configs.items():
    print(f"\n=== 🔁 Start AAindex feature selection for model: {model_name} ===")

    model_cache_file = os.path.join(OUTPUT_DIR, f'{model_name}_aaindex_cache.pkl')

    if os.path.exists(model_cache_file):
        selected_indices, model_history = pickle.load(open(model_cache_file, 'rb'))
        start_step = len(selected_indices) + 1
        remaining = [i for i in all_aaindex_codes if i not in selected_indices]
        print(f"🔁 Resume {model_name} from step {start_step}, already selected: {selected_indices}")

        best_for_model_r2 = max([h['test_r2'] for h in model_history]) if model_history else -np.inf
    else:
        selected_indices = []
        model_history = []
        
        start_step = 2
        remaining = all_aaindex_codes.copy()

        print(f"🆕 Starting fresh for {model_name}, finding best single AAindex...")
        best_cv_mse = np.inf
        best_single_index = None
        best_pipeline = None
        best_params = None
        best_mse = None
        best_r2 = None
        best_n_comp = None

        for aaindex in tqdm(remaining, desc=f"{model_name} single index"):
            try:
                mse, r2, cv_mse, n_comp, pipeline, params = cv_score_single_model([aaindex], base_model, param_grid)
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_single_index = aaindex
                    best_pipeline = pipeline
                    best_params = params
                    best_mse = mse
                    best_r2 = r2
                    best_n_comp = n_comp
            except Exception as e:
                pass

        if best_single_index:
            selected_indices = [best_single_index]
            remaining.remove(best_single_index)
            print(f"✅ First AAindex for {model_name} = {best_single_index}, CV_MSE = {best_cv_mse:.6f}")

            try:
                X_train_final = np.array(get_aai_encoding(train_seqs, indices=selected_indices))
                best_pipeline.fit(X_train_final, y_train)
            except Exception as e:
                print(f"Warning: fit failed for {model_name} at initial step: {e}")

            try:
                X_test_final = np.array(get_aai_encoding(test_seqs, indices=selected_indices))
                pred_log = best_pipeline.predict(X_test_final)
            except Exception as e:
                print(f"Error: prediction failed for {model_name} at initial step: {e}")
                pred_log = np.zeros(len(df_test))

          
            pred_log = np.asarray(pred_log).reshape(-1)
            pred = np.expm1(pred_log)
            # ---------------------------

            r2_test = r2_score(y_test_orig, pred)
            r2_origin_test = r2_over_origin(y_test_orig, pred)
            rmse_test = np.sqrt(mean_squared_error(y_test_orig, pred))
            spearman_test = spearmanr(y_test_orig, pred).correlation

            hist_entry = {
                'model': model_name, 'step': 1, 'added_feat': best_single_index,
                'selected_features': selected_indices.copy(), 'best_params': best_params,
                'train_mse': best_mse, 'train_r2': best_r2, 'cv_mse': best_cv_mse,
                'n_components': best_n_comp, 'test_r2': r2_test,
                'test_r2_origin': r2_origin_test, 'rmse': rmse_test, 'spearman': spearman_test
            }
            model_history.append(hist_entry)

            print(
                f"Model {model_name} Step 1: +{best_single_index}, Train_MSE={best_mse:.6f}, Train_R2={best_r2:.4f}, CV_MSE={best_cv_mse:.6f}, Test_R2={r2_test:.4f}")

            try:
                X_train_aa = np.array(get_aai_encoding(train_seqs, indices=selected_indices))
                X_test_aa = np.array(get_aai_encoding(test_seqs, indices=selected_indices))
                cache_file = create_step_cache_file(model_name, 1, selected_indices, X_train_aa, X_test_aa,
                                                    best_pipeline, best_params, hist_entry)
                all_step_cache_files.append(cache_file)
            except Exception as e:
                print(f"Warning: Failed to create cache for step 1: {e}")

            df_pred = df_test.copy()
            df_pred['predicted_fitness'] = pred
            safe_name = model_name.replace("/", "_")
            df_pred.to_csv(os.path.join(OUTPUT_DIR, f'{safe_name}_step1_pred.csv'), index=False)

            try:
                pickle.dump(best_pipeline, open(os.path.join(OUTPUT_DIR, f'{safe_name}_step1_model.pkl'), 'wb'))
            except Exception as e:
                pass

           
            pickle.dump((selected_indices, model_history), open(model_cache_file, 'wb'))

            best_for_model = {**hist_entry, 'best_step': 1, 'model_obj': best_pipeline}
            best_for_model_r2 = r2_test

        else:
            print(f"❌ No valid single AAindex found for {model_name}, skipping this model")
            continue

    for step in range(start_step, N_SELECTION_STEPS + 1):
        best_feat = None
        best_cv_mse = np.inf
        best_pipeline = best_params = best_mse = best_r2 = best_n_comp = None

        for aaindex in tqdm(remaining, desc=f"{model_name} Step {step} candidates"):
            trial_indices = selected_indices + [aaindex]
            try:
                mse, r2, cv_mse, n_comp, pipeline, params = cv_score_single_model(trial_indices, base_model, param_grid)
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_feat = aaindex
                    best_pipeline = pipeline
                    best_params = params
                    best_mse = mse
                    best_r2 = r2
                    best_n_comp = n_comp
            except Exception as e:
                pass

        if best_feat is None:
            print(f"No valid candidate found for model {model_name} at step {step}; stopping early.")
            break

        selected_indices.append(best_feat)
        remaining.remove(best_feat)

        try:
            X_train_final = np.array(get_aai_encoding(train_seqs, indices=selected_indices))
            best_pipeline.fit(X_train_final, y_train)
        except Exception as e:
            print(f"Warning: fit failed for {model_name} at step {step}: {e}")
            continue

        try:
            X_test_final = np.array(get_aai_encoding(test_seqs, indices=selected_indices))
            pred_log = best_pipeline.predict(X_test_final)
        except Exception as e:
            pred_log = np.zeros(len(df_test))

        pred_log = np.asarray(pred_log).reshape(-1)
        pred = np.expm1(pred_log)

        r2_test = r2_score(y_test_orig, pred)
        r2_origin_test = r2_over_origin(y_test_orig, pred)
        rmse_test = np.sqrt(mean_squared_error(y_test_orig, pred))
        spearman_test = spearmanr(y_test_orig, pred).correlation

        hist_entry = {
            'model': model_name, 'step': step, 'added_feat': best_feat,
            'selected_features': selected_indices.copy(), 'best_params': best_params,
            'train_mse': best_mse, 'train_r2': best_r2, 'cv_mse': best_cv_mse,
            'n_components': best_n_comp, 'test_r2': r2_test,
            'test_r2_origin': r2_origin_test, 'rmse': rmse_test, 'spearman': spearman_test
        }
        model_history.append(hist_entry)

        print(
            f"Model {model_name} Step {step}: +{best_feat}, Train_MSE={best_mse:.6f}, Train_R2={best_r2:.4f}, CV_MSE={best_cv_mse:.6f}, Test_R2={r2_test:.4f}")

        try:
            X_train_aa = np.array(get_aai_encoding(train_seqs, indices=selected_indices))
            X_test_aa = np.array(get_aai_encoding(test_seqs, indices=selected_indices))
            cache_file = create_step_cache_file(model_name, step, selected_indices, X_train_aa, X_test_aa,
                                                best_pipeline, best_params, hist_entry)
            all_step_cache_files.append(cache_file)
        except Exception as e:
            pass

        df_pred = df_test.copy()
        df_pred['predicted_fitness'] = pred
        safe_name = model_name.replace("/", "_")
        df_pred.to_csv(os.path.join(OUTPUT_DIR, f'{safe_name}_step{step}_pred.csv'), index=False)

        try:
            pickle.dump(best_pipeline, open(os.path.join(OUTPUT_DIR, f'{safe_name}_step{step}_model.pkl'), 'wb'))
        except:
            pass

        if r2_test > best_for_model_r2:
            best_for_model_r2 = r2_test
            best_for_model = {**hist_entry, 'best_step': step, 'model_obj': best_pipeline}

        pd.DataFrame(model_history).to_csv(os.path.join(OUTPUT_DIR, f'{safe_name}_selection_history.csv'), index=False)
        pickle.dump((selected_indices, model_history), open(model_cache_file, 'wb'))

    per_model_histories[model_name] = model_history
    if 'best_for_model' in locals() and best_for_model is not None:
        safe_name = model_name.replace("/", "_")
        try:
            pickle.dump(best_for_model['model_obj'],
                        open(os.path.join(OUTPUT_DIR, f'{safe_name}_best_model.pkl'), 'wb'))
        except:
            pass
        meta = {k: v for k, v in best_for_model.items() if k != 'model_obj'}
        all_models_summary.append(meta)

# --------------------
# Save Cache Summary
# --------------------
cache_summary = []
for cache_file in all_step_cache_files:
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        cache_summary.append({
            'cache_file': os.path.basename(cache_file),
            'model_name': cache_data['model_name'],
            'step': cache_data['step'],
            'num_features': len(cache_data['selected_aaindices']),
            'selected_features': cache_data['selected_aaindices'],
            'test_r2': cache_data['performance_metrics'].get('test_r2', 'N/A'),
            'cv_mse': cache_data['performance_metrics'].get('cv_mse', 'N/A')
        })
    except Exception as e:
        print(f"Warning: Could not read cache file {cache_file}: {e}")

if cache_summary:
    cache_summary_df = pd.DataFrame(cache_summary)
    cache_summary_df.to_csv(os.path.join(AAINDEX_STEP_CACHE_DIR, 'all_step_cache_summary.csv'), index=False)

# --------------------
# Final Summary & Plots
# --------------------
summary_df = pd.DataFrame(all_models_summary)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'all_models_best_summary.csv'), index=False)

if not summary_df.empty:
    idx_best = summary_df['test_r2'].idxmax()
    winner = summary_df.loc[idx_best].to_dict()
    print("\n=== ✅ Final Winner (by test R²) ===")
    for k, v in winner.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for model_name, history in per_model_histories.items():
    if not history: continue
    plt.plot([h['step'] for h in history], [h['test_r2'] for h in history], marker='o', label=model_name)
plt.xlabel('Selection Step')
plt.ylabel('Test R²')
plt.title('Test R² by Model and Step')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.subplot(1, 3, 2)
for model_name, history in per_model_histories.items():
    if not history: continue
    plt.plot([h['step'] for h in history], [h['cv_mse'] for h in history], marker='s', label=model_name)
plt.xlabel('Selection Step')
plt.ylabel('CV MSE')
plt.title('CV MSE by Model and Step')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.subplot(1, 3, 3)
if not summary_df.empty:
    plt.bar(summary_df['model'], summary_df['test_r2'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Best Test R²')
    plt.title('Best Model Performance Comparison')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "performance_comparison.png"), dpi=300, bbox_inches='tight')

print(f"\n🔮 Ready for ESM feature selection on {len(all_step_cache_files)} AAIndex configurations!")