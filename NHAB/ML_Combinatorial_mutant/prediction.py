import pickle
import numpy as np
import pandas as pd
from FFT import get_aai_encoding
import os
import matplotlib.pyplot as plt


def predict_new_sequences_batch(model_path, selected_features, new_sequences_csv, output_csv=None, batch_size=1000):
    """
    Batch prediction of new sequence fitness using a trained model (Memory-optimized version)
    """
    # 1. Load the model (Pipeline containing StandardScaler and estimator)
    print(f"🔍 Loading model: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

    # 2. Prepare new sequence data
    print(f"📊 Reading sequence data: {new_sequences_csv}")
    try:
        df_new = pd.read_csv(new_sequences_csv)

        # Check for required columns
        if 'sequence' not in df_new.columns and 'mut_seq' not in df_new.columns:
            raise ValueError("CSV file must contain either 'sequence' or 'mut_seq' column.")

        # Use 'sequence' if available, otherwise use 'mut_seq'
        seq_col = 'sequence' if 'sequence' in df_new.columns else 'mut_seq'

        total_sequences = len(df_new)
        print(f"✅ Successfully loaded {total_sequences} sequences.")

    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

    # Process in batches to prevent memory overflow
    all_results = []

    for batch_start in range(0, total_sequences, batch_size):
        batch_end = min(batch_start + batch_size, total_sequences)
        batch_df = df_new.iloc[batch_start:batch_end]

        current_batch = batch_start // batch_size + 1
        total_batches = (total_sequences - 1) // batch_size + 1
        print(f"🔢 Processing batch {current_batch}/{total_batches}: sequences {batch_start}-{batch_end}")

        # Convert sequences to dictionary format, resetting index for get_aai_encoding compatibility
        batch_seqs = dict(enumerate(batch_df[seq_col]))

        # 3. Feature encoding
        try:
            X_batch = np.array(get_aai_encoding(batch_seqs, indices=selected_features))
        except Exception as e:
            print(f"❌ Feature encoding failed for batch {current_batch}: {e}")
            continue

        # 4. Prediction
        try:
            # Predict (Note: target values were scaled using np.log1p during training)
            y_pred_log = model.predict(X_batch)
            y_pred_log = np.asarray(y_pred_log).reshape(-1)

            # Transform back to the original scale (inverse of np.log1p)
            y_pred_original = np.expm1(y_pred_log)

        except Exception as e:
            print(f"❌ Prediction failed for batch {current_batch}: {e}")
            continue

        # 5. Organize batch results with safe column checks
        batch_results = pd.DataFrame({
            'sequence_id': batch_df['mutant_id'].values if 'mutant_id' in batch_df.columns else list(
                range(batch_start, batch_end)),
            'mutant_name': batch_df['mutant_name'].values if 'mutant_name' in batch_df.columns else [f'Seq_{i}' for i in
                                                                                                    range(batch_start,
                                                                                                          batch_end)],
            'mutations': batch_df['mutations'].values if 'mutations' in batch_df.columns else [''] * len(batch_df),
            'sequence': batch_df[seq_col].values,
            'predicted_fitness_log': y_pred_log,
            'predicted_fitness': y_pred_original,
            'mutation_count': batch_df['mutation_count'].values if 'mutation_count' in batch_df.columns else [0] * len(
                batch_df)
        })

        all_results.append(batch_results)

    # Concatenate all batch results
    if all_results:
        results = pd.concat(all_results, ignore_index=True)

        # 6. Save results
        if output_csv:
            results.to_csv(output_csv, index=False)
            print(f"💾 Prediction results successfully saved to: {output_csv}")

        return results
    else:
        print("❌ Processing failed for all batches.")
        return None


def analyze_predictions_detailed(predictions, output_dir="prediction_analysis"):
    """
    Detailed statistical analysis and visualization of prediction results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Basic statistics
    print("\n📊 Prediction Results Summary:")
    print(f"    - Total predicted sequences: {len(predictions)}")
    print(f"    - Mean predicted fitness: {predictions['predicted_fitness'].mean():.4f}")
    print(f"    - Predicted fitness range: [{predictions['predicted_fitness'].min():.4f}, "
          f"{predictions['predicted_fitness'].max():.4f}]")
    print(f"    - Standard deviation of predicted fitness: {predictions['predicted_fitness'].std():.4f}")

    # Statistics by mutation count
    if 'mutation_count' in predictions.columns and predictions['mutation_count'].any():
        print(f"\n🧬 Statistics by Mutation Count:")
        for count in sorted(predictions['mutation_count'].unique()):
            count_data = predictions[predictions['mutation_count'] == count]
            print(f"    - {count}-point mutation: {len(count_data)} sequences, "
                  f"mean fitness: {count_data['predicted_fitness'].mean():.4f}")

    # Plot prediction distributions and top performers
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(predictions['predicted_fitness'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Predicted Fitness')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Fitness')

    plt.subplot(1, 3, 2)
    plt.hist(predictions['predicted_fitness_log'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Predicted Fitness (Log Scale)')
    plt.ylabel('Count')
    plt.title('Distribution of Log Fitness')

    # Top 20 sequences with highest fitness
    top_20 = predictions.nlargest(20, 'predicted_fitness')
    plt.subplot(1, 3, 3)
    plt.bar(range(len(top_20)), top_20['predicted_fitness'], color='salmon', edgecolor='black')
    plt.xlabel('Rank')
    plt.ylabel('Predicted Fitness')
    plt.title('Top 20 Highest Fitness Variants')
    plt.xticks(range(len(top_20)), [f"#{i + 1}" for i in range(len(top_20))], rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Prevent blocking on headless servers

    # Save ranking results
    top_100 = predictions.nlargest(100, 'predicted_fitness')
    top_100.to_csv(os.path.join(output_dir, 'top_100_predictions.csv'), index=False)

    top_20.to_csv(os.path.join(output_dir, 'top_20_predictions.csv'), index=False)

    # Save complete prediction results
    predictions.to_csv(os.path.join(output_dir, 'all_predictions.csv'), index=False)

    return top_20


def main_optimized():
    # Configuration parameters
    MODEL_PATH = "./Data_for_output/aaindex_modelwise_selection/GradientBoosting_best_model.pkl"
    SELECTED_FEATURES = ['TANS770107', 'GEIM800111', 'PALJ810105']
    NEW_SEQUENCES_CSV = "./prediction/Combinatorial_mutant_squence.csv"
    OUTPUT_CSV = "./prediction/prediction_results.csv"

    print("=== 🚀 Starting Batch Prediction for New Sequences (Optimized Version) ===")

    # Execute batch predictions
    results = predict_new_sequences_batch(
        model_path=MODEL_PATH,
        selected_features=SELECTED_FEATURES,
        new_sequences_csv=NEW_SEQUENCES_CSV,
        output_csv=OUTPUT_CSV,
        batch_size=500  # Adjust batch size based on memory availability
    )

    if results is not None:
        # Preview results
        print("\n📋 Prediction Results Preview:")
        print(results.head(10))

        # Perform detailed analysis
        print("\n📈 Analyzing prediction results in detail...")
        top_mutants = analyze_predictions_detailed(results)

        print(f"\n🎉 Prediction completed successfully!")
        print(f"🔬 Identified {len(top_mutants)} high-fitness mutants.")

        # Safe fallback for mutant name
        best_mutant_name = top_mutants.iloc[0]['mutant_name'] if 'mutant_name' in top_mutants.columns else \
        top_mutants.iloc[0]['sequence_id']
        print(f"🏆 Top Fitness Variant: {best_mutant_name}")
        print(f"📊 Maximum Predicted Fitness: {top_mutants.iloc[0]['predicted_fitness']:.4f}")
        print(f"🔢 Mutation Count: {top_mutants.iloc[0]['mutation_count']}")

    else:
        print("❌ Prediction failed. Please check error logs.")


if __name__ == "__main__":
    main_optimized()