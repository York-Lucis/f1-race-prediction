import os
from pathlib import Path

def create_project_structure():
    # Define the core directories
    directories = [
        "datasets/raw",
        "datasets/processed",
        "src",
        "notebooks",
        "models",
        "outputs/visualizations",
        "outputs/predictions"
    ]

    # Define the foundational blank files to create
    files = [
        "src/__init__.py",
        "src/data_ingestion.py",
        "src/preprocess.py",
        "src/feature_engineering.py",
        "src/model_training.py",
        "src/utils.py",
        "notebooks/01_Exploratory_Data_Analysis.ipynb",
        "requirements.txt",
        "README.md",
        ".gitignore"
    ]

    print("Building F1 2026 Predictor Project Structure...")

    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

    # Create empty files
    for file_path in files:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                # Add a basic template for README and .gitignore
                if file_path == "README.md":
                    f.write("# F1 2026 Season Predictor\n\nPredicting the 2026 Formula 1 season using historical race data and video game driver ratings.")
                elif file_path == ".gitignore":
                    f.write("__pycache__/\n*.pyc\n.env\nvenv/\ndatasets/raw/\ndatasets/processed/\nmodels/*.pkl\n")
                else:
                    pass
            print(f"Created file: {file_path}")
        else:
            print(f"File already exists: {file_path}")

    print("\nProject setup complete! Don't forget to move your CSV files into 'datasets/raw/'.")

if __name__ == "__main__":
    create_project_structure()