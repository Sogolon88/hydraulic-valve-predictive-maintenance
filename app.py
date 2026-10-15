"""Interface de la pipeline de maintenance prédictive.
Lancer depuis la racine du projet : streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Maintenance prédictive de la vanne", page_icon=":material/valve:",
                   layout="wide")

page = st.navigation([
    st.Page("app_pages/donnees.py", title="Données", icon=":material/monitoring:", default=True),
    st.Page("app_pages/entrainement.py", title="Entraînement", icon=":material/model_training:"),
    st.Page("app_pages/prediction.py", title="Prédiction", icon=":material/troubleshoot:"),
])

with st.sidebar:
    st.caption("Banc hydraulique : prédire si la vanne est optimale (100 %) "
               "à partir de la pression PS2 et du débit FS1 d'un cycle de 60 s.")

page.run()
