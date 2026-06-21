from pathlib import Path
import re

import matplotlib.pyplot as plt


# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "classification_report.txt"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# Read Classification Report
# ==================================================

with open(
    REPORT_FILE,
    "r",
    encoding="utf-8"
) as f:

    report_text = f.read()


# ==================================================
# Extract Entity Counts
# ==================================================

chemical_match = re.search(
    r"Chemical\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+(\d+)",
    report_text
)

disease_match = re.search(
    r"Disease\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+(\d+)",
    report_text
)

chemical_count = (
    int(chemical_match.group(1))
    if chemical_match
    else 0
)

disease_count = (
    int(disease_match.group(1))
    if disease_match
    else 0
)


# ==================================================
# Extract Precision Recall F1
# ==================================================

micro_match = re.search(
    r"micro avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)",
    report_text
)

if micro_match:

    precision = float(
        micro_match.group(1)
    )

    recall = float(
        micro_match.group(2)
    )

    f1 = float(
        micro_match.group(3)
    )

else:

    precision = 0
    recall = 0
    f1 = 0


# ==================================================
# Entity Distribution Plot
# ==================================================

entity_names = [
    "Chemical",
    "Disease"
]

entity_counts = [
    chemical_count,
    disease_count
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    entity_names,
    entity_counts
)

plt.title(
    "BC5CDR Entity Distribution"
)

plt.ylabel(
    "Number of Entities"
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 50,
        f"{int(height)}",
        ha="center",
        fontsize=10
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "entity_distribution.png",
    dpi=300
)

plt.close()


# ==================================================
# Performance Plot
# ==================================================

metrics = [
    "Precision",
    "Recall",
    "F1 Score"
]

scores = [
    precision,
    recall,
    f1
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    metrics,
    scores
)

plt.ylim(
    0,
    1.0
)

plt.title(
    "Model Performance"
)

plt.ylabel(
    "Score"
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,
        f"{height:.4f}",
        ha="center",
        fontsize=10
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "performance_metrics.png",
    dpi=300
)

plt.close()


print("\nFigures generated successfully.")
print("Saved to:")
print(FIGURES_DIR)