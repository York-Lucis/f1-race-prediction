import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import pickle

# Bulletproof paths
THIS_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = THIS_DIR.parent / "datasets" / "processed"
MODELS_DIR = THIS_DIR.parent / "models"


def train_model():
    print("--- Starting Phase 4: Model Training ---")

    # 1. Load Data
    data_path = PROCESSED_DIR / "f1_ml_ready.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {data_path}")
        return

    # 2. Define Features and Target
    # These are the specific columns the AI will look at to make its predictions.
    features = [
        'Starting Grid', 'Rating', 'Experience', 'Racecraft',
        'Awareness', 'Pace', 'Driver_Season_Avg_Points',
        'Team_Season_Avg_Points', 'Career_Races_Logged'
    ]
    target = 'Points'

    # 3. Chronological Train-Test Split
    # Train on 2013-2023, Test on 2024
    train_df = df[df['Season_Year'] <= 2023]
    test_df = df[df['Season_Year'] == 2024]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    print(f"Training on {len(X_train)} rows (Seasons up to 2023)")
    print(f"Testing on {len(X_test)} rows (Season 2024)")

    # 4. Initialize and Train XGBoost
    print("\nInitializing XGBoost Regressor (Attempting to use RTX 3080 via CUDA)...")
    try:
        # Modern XGBoost uses device='cuda' to target the GPU
        model = xgb.XGBRegressor(
            n_estimators=500,  # Number of decision trees
            learning_rate=0.05,  # How aggressively the model learns
            max_depth=6,  # Maximum depth of each tree
            random_state=42,
            tree_method='hist',
            device='cuda'  # <-- Tapping into your RTX 3080!
        )
        model.fit(X_train, y_train)
        print("--> Successfully trained using CUDA (GPU)!")
    except Exception as e:
        # Fallback just in case your Python environment doesn't have the CUDA toolkit linked perfectly
        print(f"CUDA unavailable. Falling back to CPU. Error: {e}")
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train)
        print("--> Successfully trained using CPU!")

    # 5. Evaluate the Model
    predictions = model.predict(X_test)

    # F1 doesn't have negative points, so we cap predictions at 0
    predictions = np.maximum(0, predictions)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n--- Model Evaluation (2024 Season Accuracy) ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} points per race")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} points per race")
    print("*(Note: An MAE between 2-4 points is considered very strong for F1 prediction!)*")

    # 6. Save the Model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "f1_xgboost_model.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n--> Success! Model saved to {model_path}")
    print("--- Model Training Complete ---")


if __name__ == "__main__":
    train_model()