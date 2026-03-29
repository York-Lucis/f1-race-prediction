import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Bulletproof paths
THIS_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = THIS_DIR.parent / "outputs" / "predictions"
VISUALIZATIONS_DIR = THIS_DIR.parent / "outputs" / "visualizations"


def create_standings_chart():
    print("--- Generating Visualizations ---")

    # 1. Find the generated prediction file
    # We use glob to find any Drivers' Championship prediction file dynamically
    prediction_files = list(PREDICTIONS_DIR.glob("*_Drivers_Championship_Prediction.csv"))

    if not prediction_files:
        print("Error: No prediction files found in outputs/predictions/. Run predict_2026.py first.")
        return

    # Grab the most recent file (e.g., 2026)
    latest_file = sorted(prediction_files)[-1]
    df = pd.read_csv(latest_file)

    # 2. Filter for the Top 10 drivers
    top_10 = df.head(10)

    # 3. Set up the visual styling
    # A darkgrid looks sleek and modern for motorsports data
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 7))

    # 4. Create the horizontal bar chart
    ax = sns.barplot(
        x='Predicted_Points',
        y='Driver',
        data=top_10,
        palette="magma"  # A cool gradient from dark purple to bright yellow
    )

    # 5. Add titles and labels
    year = latest_file.name.split('_')[0]
    plt.title(f"Predicted {year} F1 World Drivers' Championship (Top 10)", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Predicted Total Points", fontsize=14, fontweight='bold')
    plt.ylabel("Driver", fontsize=14, fontweight='bold')

    # 6. Add the exact numbers to the end of each bar for readability
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=5, fontsize=12)

    # 7. Save the image
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = VISUALIZATIONS_DIR / "standings.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"--> Success! High-resolution bar chart saved to {output_path}")
    print("--- Visualization Complete! ---")


if __name__ == "__main__":
    create_standings_chart()