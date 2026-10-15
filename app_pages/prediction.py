import pandas as pd
import streamlit as st

from app_pages.shared import (N_TRAIN, PIPELINE_SET, STATE_LABELS, raw_data, saved_model,
                              saved_model_test_report)
from src import pipeline
from src.data import list_models

st.title("Prédiction")
st.caption(f"Modèles entraînés dans le notebook final (models/{PIPELINE_SET}/), "
           f"features histogramme des {N_TRAIN} premiers cycles.")

names = list_models(PIPELINE_SET)
if not names:
    st.error(f"Aucun modèle dans models/{PIPELINE_SET}/ : exécute d'abord notebook_projet.ipynb.",
             icon=":material/error:")
    st.stop()

name = st.selectbox("Modèle", names,
                    index=names.index("gradient_boosting") if "gradient_boosting" in names else 0)
model = saved_model(name)
has_proba = hasattr(model, "predict_proba")
ps2, fs1, profile = raw_data()

source = st.segmented_control("Cycle à diagnostiquer", ["dataset", "fichiers"], default="dataset",
                              format_func={"dataset": "Cycle du jeu de données",
                                           "fichiers": "Fichiers PS2 et FS1"}.get, required=True)


def show_result(prediction: int, score: float, valve: int | None = None):
    """Carte du diagnostic d'un cycle, avec l'état réel s'il est connu."""
    score_label = f"{100 * score:.1f} %" if has_proba else f"{score:+.2f}"
    with st.container(horizontal=True):
        st.metric("État prédit", STATE_LABELS[prediction], border=True, height="stretch",
                  icon=":material/check_circle:" if prediction else ":material/warning:")
        st.metric("Score « optimale »", score_label, border=True, height="stretch",
                  help="Probabilité que la vanne soit optimale" if has_proba
                  else "Distance à la frontière de décision (positive = optimale)")
        if valve is not None:
            correct = prediction == int(valve == 100)
            st.metric("État réel", f"vanne à {valve} %", border=True, height="stretch",
                      delta="prédiction correcte" if correct else "prédiction fausse",
                      delta_color="normal" if correct else "inverse",
                      delta_arrow="off")


if source == "dataset":
    cycle = st.number_input("Numéro du cycle", min_value=0, max_value=len(profile) - 1,
                            value=N_TRAIN, step=1)
    if cycle < N_TRAIN:
        st.caption(":material/info: Cycle d'entraînement : le modèle l'a déjà vu.")
    ps2_cycle, fs1_cycle = ps2.iloc[[cycle]], fs1.iloc[[cycle]]
    show_result(pipeline.predict_cycle(ps2_cycle, fs1_cycle, model),
                pipeline.cycle_score(ps2_cycle, fs1_cycle, model),
                valve=int(profile["valve"].iloc[cycle]))
else:
    st.caption(f"Un cycle par ligne, {ps2.shape[1]} valeurs PS2 et {fs1.shape[1]} valeurs FS1, "
               "même format que data/raw/. Les valeurs manquantes sont remplacées par la moyenne du cycle.")
    ps2_col, fs1_col = st.columns(2)
    ps2_file = ps2_col.file_uploader("Fichier PS2", type=["txt", "csv", "tsv"])
    fs1_file = fs1_col.file_uploader("Fichier FS1", type=["txt", "csv", "tsv"])
    if ps2_file and fs1_file:
        ps2_up = pd.read_csv(ps2_file, sep=None, engine="python", header=None)
        fs1_up = pd.read_csv(fs1_file, sep=None, engine="python", header=None)
        if ps2_up.shape[1] != ps2.shape[1] or fs1_up.shape[1] != fs1.shape[1]:
            st.error(f"Format attendu : {ps2.shape[1]} colonnes pour PS2 et {fs1.shape[1]} pour FS1 "
                     f"(reçu {ps2_up.shape[1]} et {fs1_up.shape[1]}).", icon=":material/error:")
        elif len(ps2_up) != len(fs1_up):
            st.error(f"PS2 et FS1 n'ont pas le même nombre de cycles ({len(ps2_up)} et {len(fs1_up)}).",
                     icon=":material/error:")
        elif len(ps2_up) == 1:
            show_result(pipeline.predict_cycle(ps2_up, fs1_up, model),
                        pipeline.cycle_score(ps2_up, fs1_up, model))
        else:
            rows = [(pipeline.predict_cycle(ps2_up.iloc[[i]], fs1_up.iloc[[i]], model),
                     pipeline.cycle_score(ps2_up.iloc[[i]], fs1_up.iloc[[i]], model))
                    for i in range(len(ps2_up))]
            table = pd.DataFrame(rows, columns=["prédiction", "score"]).rename_axis("cycle")
            table["prédiction"] = table["prédiction"].map(STATE_LABELS)
            st.dataframe(table, alt="Diagnostic de chaque cycle des fichiers chargés")
            st.download_button("Télécharger les prédictions", table.to_csv(), "predictions.csv",
                               icon=":material/download:")

st.subheader("Performance sur le jeu de test")
scores, crosstab = saved_model_test_report(name)
with st.container(horizontal=True):
    st.metric("F1 macro", f"{scores['f1_macro']:.3f}", border=True)
    st.metric("Accuracy", f"{scores['accuracy']:.3f}", border=True)
    st.metric("Balanced accuracy", f"{scores['balanced_acc']:.3f}", border=True)
    st.metric("AUC", f"{scores['auc']:.3f}", border=True)
st.dataframe(crosstab, alt="Nombre de cycles de test par état réel de la vanne et par prédiction")
