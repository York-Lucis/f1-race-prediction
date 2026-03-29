import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path

PROCESSED_DIR = Path("datasets/processed")


def normalize_text(text):
    """
    Removes accents, special characters, and converts to uppercase
    to ensure perfect string matching across datasets.
    Example: 'Sergio Pérez' -> 'SERGIO PEREZ'
    """
    if pd.isna(text):
        return text

    text = str(text).strip().upper()
    # Normalize unicode characters to remove accents
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    return text


def preprocess_and_merge():
    print("--- Starting Data Preprocessing ---")

    # 1. Load the master files
    try:
        races_df = pd.read_csv(PROCESSED_DIR / "master_race_results.csv")
        ratings_df = pd.read_csv(PROCESSED_DIR / "master_driver_ratings.csv")
        print(f"Loaded Race Results: {len(races_df)} rows")
        print(f"Loaded Driver Ratings: {len(ratings_df)} rows")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure you ran data_ingestion.py first.")
        return

    # 2. Normalize Driver Names
    print("Normalizing driver names for exact matching...")
    races_df['Driver_Match'] = races_df['Driver'].apply(normalize_text)
    ratings_df['Driver_Match'] = ratings_df['Driver'].apply(normalize_text)

    # Note: Video game ratings often have multiple updates per year (e.g., initial, oct, jan).
    # For our baseline merge, let's take the *average* rating a driver had across that game's lifespan.
    numeric_rating_cols = ['Rating', 'Experience', 'Racecraft', 'Awareness', 'Pace']

    # Group by driver and year to get their average stats for that specific season
    avg_ratings = ratings_df.groupby(['Driver_Match', 'Game_Year'])[numeric_rating_cols].mean().reset_index()

    # 3. Merge Datasets
    # We do a LEFT JOIN so we keep all real-world race results, attaching game ratings where they exist.
    print("Merging race results with video game ratings...")
    merged_df = pd.merge(
        races_df,
        avg_ratings,
        left_on=['Driver_Match', 'Season_Year'],
        right_on=['Driver_Match', 'Game_Year'],
        how='left'
    )

    # 4. Handle Missing Values
    # Substitute drivers (like Liam Lawson or Nyck de Vries) might not have game ratings for that year.
    # We will fill their missing game ratings with the mean rating of the grid for that specific year.
    print("Imputing missing video game ratings for substitute/unrated drivers...")
    for col in numeric_rating_cols:
        merged_df[col] = merged_df.groupby('Season_Year')[col].transform(lambda x: x.fillna(x.mean()))

        # If an entire year is missing ratings (e.g., 2013-2019 games might not be in the dataset),
        # fill with the global mean to avoid NaNs crashing our future ML models.
        merged_df[col] = merged_df[col].fillna(merged_df[col].mean())

        # Round the stats back to integers (as they appear in the game)
        merged_df[col] = merged_df[col].round(0)

    # Clean up redundant columns
    if 'Game_Year' in merged_df.columns:
        merged_df.drop(columns=['Game_Year', 'Driver_Match'], inplace=True)

    # 5. Save the final Analytics Master table
    output_path = PROCESSED_DIR / "f1_analytics_master.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"--> Success! Saved ultimate analytics dataset to {output_path} ({len(merged_df)} total rows)\n")
    print("--- Preprocessing Complete ---")


if __name__ == "__main__":
    preprocess_and_merge()