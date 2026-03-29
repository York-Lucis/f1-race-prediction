import pandas as pd
from pathlib import Path
import os

THIS_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = THIS_DIR.parent / "datasets" / "processed"


def engineer_features():
    print("--- Starting Feature Engineering ---")

    file_path = PROCESSED_DIR / "f1_analytics_master.csv"
    print(f"Looking for dataset at: {file_path}")

    try:
        df = pd.read_csv(file_path)
        print(f"Loaded master dataset: {len(df)} rows")
    except FileNotFoundError:
        print("Error: Could not find f1_analytics_master.csv. Run preprocess.py first.")
        return

    # Ensure Points and Starting Grid are numeric
    df['Points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)
    df['Starting Grid'] = pd.to_numeric(df['Starting Grid'], errors='coerce').fillna(20)

    print("Calculating Driver and Team Momentum...")

    # Feature 1: Driver's Average Points per Race in a given season
    driver_season_stats = df.groupby(['Season_Year', 'Driver'])['Points'].mean().reset_index()
    driver_season_stats.rename(columns={'Points': 'Driver_Season_Avg_Points'}, inplace=True)
    df = pd.merge(df, driver_season_stats, on=['Season_Year', 'Driver'], how='left')

    # Feature 2: Cumulative Experience (How many races they've logged in our dataset so far)
    # We sort chronologically to ensure the cumulative count makes sense
    df = df.sort_values(by=['Season_Year', 'Driver'])
    df['Career_Races_Logged'] = df.groupby('Driver').cumcount() + 1

    # Feature 3: Team Strength (Average points scored by the constructor in that season)
    team_season_stats = df.groupby(['Season_Year', 'Team'])['Points'].mean().reset_index()
    team_season_stats.rename(columns={'Points': 'Team_Season_Avg_Points'}, inplace=True)
    df = pd.merge(df, team_season_stats, on=['Season_Year', 'Team'], how='left')

    # Drop any remaining rows that might break the ML model (like weird text in numeric columns)
    df.dropna(subset=['Points', 'Rating', 'Pace'], inplace=True)

    # Save the final ML-ready dataset
    output_path = PROCESSED_DIR / "f1_ml_ready.csv"
    df.to_csv(output_path, index=False)
    print(f"--> Success! Saved ML-ready dataset to {output_path} ({len(df)} total rows)\n")
    print("--- Feature Engineering Complete ---")


if __name__ == "__main__":
    engineer_features()