import pandas as pd
import pickle
import numpy as np
from pathlib import Path

# Bulletproof paths
THIS_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = THIS_DIR.parent / "datasets" / "processed"
MODELS_DIR = THIS_DIR.parent / "models"
OUTPUT_DIR = THIS_DIR.parent / "outputs" / "predictions"


def predict_2026():
    print("--- Starting Phase 5: 2026 Season Prediction ---")

    # 1. Load the trained XGBoost model
    model_path = MODELS_DIR / "f1_xgboost_model.pkl"
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}. Run model_training.py first.")
        return

    # 2. Load the ML-ready dataset
    data_path = PROCESSED_DIR / "f1_ml_ready.csv"
    df = pd.read_csv(data_path)

    # 3. Filter for the 2026 Season
    # If 2026 got filtered out during preprocessing (e.g., if real race points were blank),
    # we will dynamically grab the latest available year to simulate the "next" season.
    target_year = 2026
    df_future = df[df['Season_Year'] == target_year].copy()

    if df_future.empty:
        target_year = df['Season_Year'].max()
        print(
            f"Note: 2026 data wasn't found in the processed set. Predicting based on {target_year} data grid instead...")
        df_future = df[df['Season_Year'] == target_year].copy()

    # 4. Define features (Must match exactly what we trained the AI on)
    features = [
        'Starting Grid', 'Rating', 'Experience', 'Racecraft',
        'Awareness', 'Pace', 'Driver_Season_Avg_Points',
        'Team_Season_Avg_Points', 'Career_Races_Logged'
    ]

    print(f"Predicting outcomes for the {target_year} season...")

    # 5. Make Predictions
    X_future = df_future[features]
    predictions = model.predict(X_future)

    # F1 doesn't have negative points, so we floor the predictions at 0
    df_future['Predicted_Points'] = np.maximum(0, predictions)

    # 6. Aggregate to get Championship Standings
    # World Drivers' Championship (WDC)
    wdc_standings = df_future.groupby('Driver')['Predicted_Points'].sum().reset_index()
    wdc_standings = wdc_standings.sort_values(by='Predicted_Points', ascending=False).reset_index(drop=True)
    wdc_standings.index += 1  # Make index 1-based for position

    # Constructors' Championship (WCC)
    wcc_standings = df_future.groupby('Team')['Predicted_Points'].sum().reset_index()
    wcc_standings = wcc_standings.sort_values(by='Predicted_Points', ascending=False).reset_index(drop=True)
    wcc_standings.index += 1

    # 7. Save outputs for your GitHub
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wdc_standings.to_csv(OUTPUT_DIR / f"{target_year}_Drivers_Championship_Prediction.csv")
    wcc_standings.to_csv(OUTPUT_DIR / f"{target_year}_Constructors_Championship_Prediction.csv")

    # 8. Print Results to Terminal
    print("\n🏆 *** PREDICTED WORLD DRIVERS' CHAMPIONSHIP TOP 10 *** 🏆")
    for i, row in wdc_standings.head(10).iterrows():
        print(f"{i}. {row['Driver']} - {row['Predicted_Points']:.1f} pts")

    print("\n🏎️ *** PREDICTED CONSTRUCTORS' CHAMPIONSHIP TOP 5 *** 🏎️")
    for i, row in wcc_standings.head(5).iterrows():
        print(f"{i}. {row['Team']} - {row['Predicted_Points']:.1f} pts")

    print(f"\n--> Full prediction CSVs saved to outputs/predictions/")
    print("--- Phase 5 Complete! ---")


if __name__ == "__main__":
    predict_2026()