import os
import re
import pandas as pd
from pathlib import Path

# Set up paths relative to the root project directory
RAW_DIR = Path("datasets/raw")
PROCESSED_DIR = Path("datasets/processed")


def extract_year_from_filename(filename):
    """Extracts a 4-digit year from the filename using regex."""
    match = re.search(r'(20\d{2})', filename)
    if match:
        return int(match.group(1))

    # Handle edge case for "F1_22" or "F1_23"
    match_short = re.search(r'f1_(\d{2})', filename.lower())
    if match_short:
        return int("20" + match_short.group(1))

    return None


def ingest_race_results():
    """Finds all race result CSVs, standardizes them, and combines them."""
    print("Ingesting Race Results...")

    # Find all files ending in 'raceResults.csv' (case insensitive) recursively
    # rglob allows us to check subfolders like 'PreviousSeasons'
    files = list(RAW_DIR.rglob("*[rR]ace[rR]esults.csv"))

    all_results = []

    for file in files:
        year = extract_year_from_filename(file.name)
        df = pd.read_csv(file)

        # Standardize column names
        df.rename(columns={
            'Total Time/Gap/Retirement': 'Time/Retired',
            'No': 'Car_Number'
        }, inplace=True)

        # Add the season year as a feature
        df['Season_Year'] = year
        all_results.append(df)
        print(f"  Loaded {file.name} (Year: {year}) - {len(df)} rows")

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        output_path = PROCESSED_DIR / "master_race_results.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"--> Saved master race results to {output_path} ({len(combined_df)} total rows)\n")
    else:
        print("--> No race result files found. Check your datasets/raw/ folder.\n")


def ingest_driver_ratings():
    """Finds all video game driver rating CSVs, standardizes them, and combines them."""
    print("Ingesting Video Game Driver Ratings...")

    files = list(RAW_DIR.rglob("*videogame_driver_ratings*.csv"))
    all_ratings = []

    # Dictionary to map varying column names to a single standard
    column_mapping = {
        'Race Craft': 'Racecraft',
        'RAC': 'Racecraft',
        'RTG': 'Rating',
        'EXP': 'Experience',
        'AWA': 'Awareness',
        'PAC': 'Pace',
        'Car Number': 'Car_Number'
    }

    for file in files:
        year = extract_year_from_filename(file.name)
        df = pd.read_csv(file)

        df.rename(columns=column_mapping, inplace=True)

        # Some datasets specify the exact patch/update month, but we'll log the base game year
        df['Game_Year'] = year

        # Keep track of the specific file origin to handle multiple updates in the same year later
        df['Source_File'] = file.name

        all_ratings.append(df)
        print(f"  Loaded {file.name} (Year: {year}) - {len(df)} rows")

    if all_ratings:
        combined_df = pd.concat(all_ratings, ignore_index=True)
        output_path = PROCESSED_DIR / "master_driver_ratings.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"--> Saved master driver ratings to {output_path} ({len(combined_df)} total rows)\n")
    else:
        print("--> No driver rating files found. Check your datasets/raw/ folder.\n")


if __name__ == "__main__":
    # Ensure processed directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("--- Starting Data Ingestion ---")
    ingest_race_results()
    ingest_driver_ratings()
    print("--- Data Ingestion Complete ---")