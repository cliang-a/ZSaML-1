import pandas as pd
import numpy as np
import pickle

# **********************
# 1. Update the following file paths according to your local environment
# **********************
# CSV file containing AAindex-FFT columns and all ESM features
FEATURE_CSV    = "./prediction/single_data.csv"
# Trained optimal PLS model checkpoint
MODEL_PKL      = "./prediction/best_model_round1.pkl"
# Text file containing the list of best ESM features for this round
BEST_FEATS_TXT = "./prediction/best_feats_round1.txt"
# Output file path for the prediction results
OUTPUT_PRED_CSV = "./prediction/result.csv"

# **********************
# 2. Load feature data
# **********************
df = pd.read_csv(FEATURE_CSV)

# **********************
# 3. Automatically identify AAindex-FFT columns
# **********************
fft_cols = [c for c in df.columns if c.startswith("AA_MONM990101_fft_")]

# **********************
# 4. Read the best ESM features list
# **********************
with open(BEST_FEATS_TXT, 'r') as f:
    best_esms = [line.strip() for line in f if line.strip()]

# **********************
# 5. Load the trained PLS model
# **********************
with open(MODEL_PKL, 'rb') as f:
    pls_model = pickle.load(f)

# **********************
# 6. Construct the input matrix for prediction
# **********************
X_fft = df[fft_cols].values
X_esm = df[best_esms].values
X_new = np.hstack([X_fft, X_esm])

# **********************
# 7. Predict and apply inverse transformation
# **********************
# Note: The model was originally trained on log1p(y), so expm1 is required
pred_log = pls_model.predict(X_new)
pred     = np.expm1(pred_log).flatten()

# **********************
# 8. Save prediction results
# **********************
df['predicted_fitness'] = pred
df.to_csv(OUTPUT_PRED_CSV, index=False)
print("✅ Prediction complete. Results saved to:", OUTPUT_PRED_CSV)