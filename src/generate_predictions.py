from pathlib import Path
import json

from src.inference import BiomedicalNER


# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "sample_predictions.json"
)


# ==================================================
# Load Model
# ==================================================

ner = BiomedicalNER()


# ==================================================
# Sample Biomedical Texts
# ==================================================

texts = [

    "Tamoxifen is commonly used in breast cancer treatment.",

    "The patient was diagnosed with typhoid and prescribed paracetamol.",

    "Aspirin can reduce the risk of cardiovascular disease.",

    "Metformin is frequently prescribed for diabetes mellitus."

]


# ==================================================
# Run Predictions
# ==================================================

results = []

for text in texts:

    entities = ner.predict(text)

    results.append({
        "input_text": text,
        "entities": entities
    })


# ==================================================
# Save
# ==================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print(f"\nSaved predictions to:\n{OUTPUT_FILE}")