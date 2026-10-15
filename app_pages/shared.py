"""Chargements mis en cache, partagés par les pages de l'application."""
import numpy as np
import pandas as pd
import streamlit as st

from src import pipeline
from src.config import N_TRAIN, PIPELINE_SET
from src.data import load_model, load_processed
from src.evaluation import test_scores

FEATURE_LABELS = {"stats": "Statistiques", "spectre": "Spectre (FFT)",
                  "stats_spectre": "Stats + spectre", "histogramme": "Histogramme"}
VALVE_LEVELS = [100, 90, 80, 73]
STATE_LABELS = {1: "Optimale", 0: "Dégradée"}


@st.cache_data(show_spinner="Chargement des capteurs…")
def raw_data():
    """(PS2, FS1, profile) bruts."""
    return pipeline.load_raw()


@st.cache_data(show_spinner=False)
def target() -> pd.Series:
    return pipeline.make_target(raw_data()[2])


@st.cache_data(show_spinner=False)
def mean_pressure_start(seconds: float = 12) -> pd.DataFrame:
    """Pression PS2 moyenne en début de cycle, par état de la vanne (cycles du train)."""
    ps2, _, profile = raw_data()
    temps = np.arange(ps2.shape[1]) / 100          # 100 mesures par seconde
    debut = temps <= seconds
    return pd.DataFrame(
        {f"vanne à {niveau} %": ps2[:N_TRAIN][profile["valve"][:N_TRAIN] == niveau].mean().values[debut]
         for niveau in VALVE_LEVELS},
        index=pd.Index(temps[debut], name="temps (s)"))


@st.cache_data(max_entries=16, show_spinner=False)
def dataset(feature_set: str, split: str, test_size: float):
    """Features, découpage et sélection : (x_train, x_test, y_train, y_test, colonnes supprimées)."""
    ps2, fs1, _ = raw_data()
    return pipeline.prepare_dataset(feature_set, ps2, fs1, target(), split=split, test_size=test_size)


@st.cache_resource(max_entries=16)
def saved_model(name: str):
    """Modèle enregistré dans models/PIPELINE_SET/."""
    return load_model(name, PIPELINE_SET)


@st.cache_data(max_entries=16, show_spinner="Évaluation sur le jeu de test…")
def saved_model_test_report(name: str):
    """Scores du modèle enregistré sur le test de l'énoncé, et tableau état réel / prédiction."""
    _, x_test, _, y_test = load_processed(PIPELINE_SET)
    model = saved_model(name)
    valve = raw_data()[2]["valve"].iloc[N_TRAIN:].to_numpy()
    crosstab = pd.crosstab(pd.Series(valve, name="état réel de la vanne (%)"),
                           pd.Series(model.predict(x_test), name="prédiction").map(STATE_LABELS))
    return test_scores(model, x_test, y_test), crosstab.sort_index(ascending=False)
