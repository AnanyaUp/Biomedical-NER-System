from pathlib import Path
import json
import torch


from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

from torch.utils.data import DataLoader
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)

from tqdm.auto import tqdm


# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "bc5cdr"
MODEL_DIR = PROJECT_ROOT / "model" / "best_model"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# Load Labels
# ==================================================

with open(DATA_DIR / "label.json", "r") as f:
    label2id = json.load(f)

id2label = {
    int(v): k
    for k, v in label2id.items()
}


# ==================================================
# Load Dataset
# ==================================================

dataset = load_dataset(
    "json",
    data_files={
        "test": str(DATA_DIR / "test.json")
    }
)


# ==================================================
# Tokenizer
# ==================================================

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR)
)


def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

    all_labels = []

    for i, labels in enumerate(examples["tags"]):

        word_ids = tokenized_inputs.word_ids(batch_index=i)

        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])

            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs


tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True
)


# ==================================================
# DataLoader
# ==================================================

def collate_fn(batch):

    keys = [
        "input_ids",
        "attention_mask",
        "labels"
    ]

    output = {
        k: [torch.tensor(ex[k]) for ex in batch]
        for k in keys
    }

    output = {
        k: torch.nn.utils.rnn.pad_sequence(
            v,
            batch_first=True,
            padding_value=0
        )
        for k, v in output.items()
    }

    return output


test_loader = DataLoader(
    tokenized_dataset["test"],
    batch_size=8,
    collate_fn=collate_fn
)


# ==================================================
# Load Model
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = AutoModelForTokenClassification.from_pretrained(
    str(MODEL_DIR)
)

model.to(device)
model.eval()


# ==================================================
# Evaluation
# ==================================================

all_preds = []
all_labels = []

with torch.no_grad():

    for batch in tqdm(test_loader):

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        outputs = model(**batch)

        predictions = torch.argmax(
            outputs.logits,
            dim=-1
        )

        preds = predictions.cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        masks = batch["attention_mask"].cpu().numpy()

        for pred, label, mask in zip(
            preds,
            labels,
            masks
        ):

            pred_tags = []
            true_tags = []

            for p, l, m in zip(
                pred,
                label,
                mask
            ):

                if m == 1 and l != -100:

                    pred_tags.append(
                        id2label[p]
                    )

                    true_tags.append(
                        id2label[l]
                    )

            all_preds.append(pred_tags)
            all_labels.append(true_tags)


precision = precision_score(
    all_labels,
    all_preds
)

recall = recall_score(
    all_labels,
    all_preds
)

f1 = f1_score(
    all_labels,
    all_preds
)

report = classification_report(
    all_labels,
    all_preds,
    digits=4
)

print("\n")
print("=" * 60)
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("=" * 60)

print("\nClassification Report:\n")
print(report)


with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

print("\nReport saved successfully.")