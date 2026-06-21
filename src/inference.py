from pathlib import Path
from src.utils import get_model_path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)


class BiomedicalNER:

    def __init__(self):

        model_path = get_model_path()

        print("Loading model from:")
        print(model_path)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path)
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            str(model_path)
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

    def predict(self, text):

        print("STEP 1 - Tokenizing")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        print("STEP 2 - Running model")

        with torch.no_grad():
            outputs = self.model(**inputs)

        print("STEP 3 - Model completed")

        predictions = torch.argmax(
            outputs.logits,
            dim=-1
        )

        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0]
        )

        entities = []

        current_entity = None
        current_tokens = []

        for token, pred in zip(tokens, predictions[0]):

            label = self.id2label[int(pred)]

            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            if label.startswith("B-"):

                if current_entity:

                    entity_text = self.tokenizer.convert_tokens_to_string(
                        current_tokens
                    )

                    entities.append({
                        "entity": current_entity,
                        "text": entity_text
                    })

                current_entity = label.replace("B-", "")
                current_tokens = [token]

            elif label.startswith("I-") and current_entity:

                current_tokens.append(token)

            else:

                if current_entity:

                    entity_text = self.tokenizer.convert_tokens_to_string(
                        current_tokens
                    )

                    entities.append({
                        "entity": current_entity,
                        "text": entity_text
                    })

                current_entity = None
                current_tokens = []

        if current_entity:

            entity_text = self.tokenizer.convert_tokens_to_string(
                current_tokens
            )

            entities.append({
                "entity": current_entity,
                "text": entity_text
            })

        print("STEP 4 - Returning entities")
        
        return entities