from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root():
    return PROJECT_ROOT


def get_data_dir():
    return PROJECT_ROOT / "data" / "bc5cdr"


def get_model_path():
    return PROJECT_ROOT / "model" / "best_model"


def load_labels():

    label_file = get_data_dir() / "label.json"

    with open(label_file, "r") as f:
        return json.load(f)


def create_output_dirs():

    directories = [
        PROJECT_ROOT / "outputs" / "reports",
        PROJECT_ROOT / "outputs" / "predictions",
        PROJECT_ROOT / "outputs" / "figures"
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )