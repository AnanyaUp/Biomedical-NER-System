from fastapi import FastAPI
from pydantic import BaseModel

from src.inference import BiomedicalNER

app = FastAPI(
    title="Biomedical NER API"
)

ner = BiomedicalNER()


class TextRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(request: TextRequest):

    entities = ner.predict(
        request.text
    )

    return {
        "entities": entities
    }