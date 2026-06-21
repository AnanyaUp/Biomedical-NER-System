import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "bc5cdr"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "model" / "best_model"


# ==========================================================
# Load Labels
# ==========================================================

with open(DATA_DIR / "label.json", "r") as f:
    label2id = json.load(f)

id2label = {
    int(v): k
    for k, v in label2id.items()
}

num_labels = len(label2id)


# ==========================================================
# Load Dataset
# ==========================================================

dataset = load_dataset(
    "json",
    data_files={
        "train": str(DATA_DIR / "train.json"),
        "validation": str(DATA_DIR / "valid.json"),
        "test": str(DATA_DIR / "test.json")
    }
)

print(dataset)


# ==========================================================
# Tokenizer
# ==========================================================

MODEL_CHECKPOINT = (
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CHECKPOINT
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

        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )

        label_ids = []

        previous_word_idx = None

        for word_idx in word_ids:

            if word_idx is None:

                label_ids.append(-100)

            elif word_idx != previous_word_idx:

                label_ids.append(
                    labels[word_idx]
                )

            else:

                label_ids.append(-100)

            previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs


tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True
)


# ==========================================================
# DataLoader
# ==========================================================

def collate_fn(batch):

    return {
        "input_ids": torch.tensor(
            [item["input_ids"] for item in batch]
        ),
        "attention_mask": torch.tensor(
            [item["attention_mask"] for item in batch]
        ),
        "labels": torch.tensor(
            [item["labels"] for item in batch]
        )
    }


train_dataloader = DataLoader(
    tokenized_datasets["train"],
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

val_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=8,
    collate_fn=collate_fn
)


# ==========================================================
# Model
# ==========================================================

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

optimizer = AdamW(
    model.parameters(),
    lr=3e-5
)


# ==========================================================
# Training Loop
# ==========================================================

NUM_EPOCHS = 3

print(f"\nUsing Device: {device}")

for epoch in range(NUM_EPOCHS):

    model.train()

    total_loss = 0

    progress_bar = tqdm(
        train_dataloader,
        desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"
    )

    for batch in progress_bar:

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = total_loss / len(train_dataloader)

    print(
        f"Epoch {epoch+1} completed | "
        f"Average Loss = {avg_loss:.4f}"
    )


# ==========================================================
# Save Model
# ==========================================================

MODEL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

model.save_pretrained(
    MODEL_OUTPUT_DIR
)

tokenizer.save_pretrained(
    MODEL_OUTPUT_DIR
)

print("\nModel saved successfully.")
print(MODEL_OUTPUT_DIR)