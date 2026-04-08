import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Prédiction Immobilière", page_icon="🏠")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# -----------------------
# LOAD MODEL
# -----------------------
try:
    model = joblib.load(MODEL_PATH)
except:
    st.error("Modèle non trouvé. Lancez train.py d'abord.")
    st.stop()

st.title("🏠 Estimation du Prix Immobilier")

st.write("Entrez les caractéristiques du logement :")

surface = st.number_input("Surface (m²)", 0.0)
rooms = st.number_input("Nombre de pièces", 0)
lstat = st.number_input("LSTAT", 0.0)

if st.button("Prédire le prix"):
    try:
        data = pd.DataFrame([{
            "RM": rooms,
            "LSTAT": lstat,
            "surface": surface
        }])

        prediction = model.predict(data)[0]

        st.success(f"💰 Prix estimé : {prediction:,.0f} $")

    except Exception as e:
        st.error(f"Erreur : {e}")