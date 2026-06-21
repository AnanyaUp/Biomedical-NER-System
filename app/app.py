import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import streamlit as st
import pandas as pd

from src.inference import BiomedicalNER

st.set_page_config(
    page_title="Biomedical NER System",
    layout="wide"
)

st.title(" Biomedical Named Entity Recognition")

st.markdown(
    """
    Detect biomedical entities such as
    **Diseases** and **Chemicals**
    using a fine-tuned BioMedBERT model.
    """
)

text = st.text_area(
    "Enter Biomedical Text",
    height=200
)

if st.button("Extract Entities"):

    if text.strip():

        with st.spinner("Running inference..."):

            ner = BiomedicalNER()

            entities = ner.predict(text)

        if entities:

            st.success(
                f"Found {len(entities)} entities"
            )

            df = pd.DataFrame(entities)

            st.dataframe(
                df,
                use_container_width=True
            )

        else:

            st.warning(
                "No biomedical entities detected."
            )