import sys

import streamlit as st
import joblib
import pandas as pd
import os

import os
from pipeline import *

# Percorso assoluto della cartella 'pipeline'
pipeline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "pipeline"))

# Aggiungi la cartella 'pipeline' a sys.path
if pipeline_path not in sys.path:
    sys.path.append(pipeline_path)


# Ottieni la directory dello script attuale
script_dir = os.path.dirname(os.path.abspath(__file__))

# Percorso della cartella pipeline
pipeline_dir = os.path.join(script_dir, "pipeline")

# Verifica che la cartella esista
if not os.path.exists(pipeline_dir):
    raise FileNotFoundError(f"La cartella pipeline non esiste: {pipeline_dir}")

# Ora puoi caricare i file senza cambiare directory
model_path = os.path.join(pipeline_dir, "random_forest_regressor_model.pkl")
transformers_path = os.path.join(pipeline_dir, "pipeline_regressor_transformers.pkl")



@st.cache_resource
def load_resources():
    model = joblib.load(model_path)
    transformers = joblib.load(transformers_path)
    known_makes = transformers['preparator'].encoder.known_makes
    return model, transformers, known_makes


model, transformers, known_makes = load_resources()

# Titolo dell'applicazione
st.title("Stima del prezzo di un'auto")

# Creazione dei campi di input
marca = st.selectbox("Inserisci la marca", known_makes.tolist())
modello = st.text_input("Inserisci il modello:")
anno_produzione = st.number_input("Inserisci l'anno di produzione:", min_value=1900, max_value=2015, value=2010, step=1)
allestimento = st.selectbox("Inserisci l'allestimento:", ['base', 'sport', 'luxury', 'special edition', 'touring',  'other'])
carrozzeria = st.selectbox("Inserisci la carrozzeria:", ['sedan', 'suv', 'hatchback', 'coupé', 'cabriolet', 'station wagon', 'pickup', 'other'])

# ⭐ Sostituzione dello slider con stelle interattive
st.write("Inserisci la condizione dell'auto:")

# Impostiamo la condizione di default su 3 stelle, se non è presente nello stato
if "condizione" not in st.session_state:
    st.session_state.condizione = 3  # Valore di default

# Creazione delle colonne per le stelle
cols = st.columns(5)

# Iteriamo sulle 5 stelle
for i in range(5):
    with cols[i]:
        # Decidiamo quale simbolo mostrare in base alla condizione
        star_label = "⭐" if i < st.session_state.condizione else "☆"
        if st.button(star_label, key=f"star_{i}"):
            st.session_state.condizione = i + 1  # Aggiorna la condizione immediatamente

chilometraggio = st.number_input("Inserisci il chilometraggio:", min_value=0, value=0, step=500)
colore = st.text_input("Inserisci il colore esterno:")
interni = st.text_input("Inserisci il colore degli interni:")

if st.button("Stima il prezzo"):
    # Creazione del dizionario con i dati dell'input
    input_data = {
        'marca': [marca],
        'modello': [modello],
        'trasmissione': 'automatic',
        'anno produzione': [int(anno_produzione)],
        'allestimento': [allestimento],
        'carrozzeria': [carrozzeria],
        'condizione': [int(st.session_state.condizione)],  # Convertito in numero
        'chilometraggio': [int(chilometraggio)],
        'colorazione': [colore],
        'colore interni': [interni]
    }

    df_input = pd.DataFrame(input_data)
    df_input = transformers['preparator'].transform_test(df_input)

    print(df_input)
    # Effettua la predizione
    prezzo_predetto = model.predict(df_input)[0]

    st.success(f"Il prezzo stimato è: {prezzo_predetto:.2f}€")
