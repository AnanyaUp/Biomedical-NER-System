from src.inference import BiomedicalNER

ner = BiomedicalNER()

text = """
I have typhoid and should take paracetamol
"""

entities = ner.predict(text)

for entity in entities:
    print(entity)