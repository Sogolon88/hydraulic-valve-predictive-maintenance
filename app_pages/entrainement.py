import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app_pages.shared import FEATURE_LABELS, dataset
from src import pipeline
from src.evaluation import plot_confusion_matrices, plot_roc_curves
from src.features import FEATURE_SETS

SEARCHES = {"aucun": "Réglages par défaut", "grid": "Grid search", "random": "Random search"}
DEFAULT_MODELS = ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting"]
COLUMNS = {"train_f1_macro": "F1 train", "cv_f1_macro": "F1 CV", "cv_std": "écart-type CV",
           "f1_macro": "F1 test", "auc": "AUC test", "accuracy": "accuracy test",
           "balanced_acc": "balanced acc. test", "precision_macro": "précision test",
           "recall_macro": "rappel test", "fit_time_s": "entraînement (s)"}

st.title("Entraînement et comparaison")
st.caption("Construit les features, découpe train / test, sélectionne les features sur le train, "
           "puis compare les modèles par validation croisée (F1 macro) et sur le test.")

with st.form("config"):
    features_col, split_col = st.columns(2)
    feature_set = features_col.segmented_control(
        "Jeu de features", FEATURE_SETS, default="histogramme", format_func=FEATURE_LABELS.get,
        required=True)
    split = split_col.segmented_control(
        "Découpage", list(pipeline.SPLITS), default="enonce", format_func=pipeline.SPLITS.get,
        required=True)
    names = st.multiselect("Modèles", list(pipeline.candidate_models()), default=DEFAULT_MODELS)
    search_col, size_col = st.columns(2)
    search = search_col.segmented_control(
        "Hyperparamètres", list(SEARCHES), default="aucun", format_func=SEARCHES.get, required=True,
        help="Réglage par validation croisée sur le train, pour les modèles qui ont une grille. "
             "Le grid search sur plusieurs modèles peut prendre quelques minutes.")
    test_size = size_col.slider("Part du test (découpage aléatoire)", 0.1, 0.4, 0.2, 0.05,
                                help="Ignoré avec le découpage de l'énoncé.")
    submitted = st.form_submit_button("Lancer l'entraînement", type="primary",
                                      icon=":material/play_arrow:")

if submitted:
    if not names:
        st.warning("Choisis au moins un modèle.", icon=":material/warning:")
        st.stop()
    start = time.perf_counter()
    with st.status("Entraînement en cours…", expanded=True) as status:
        st.write(f"Features « {FEATURE_LABELS[feature_set]} » et découpage…")
        x_train, x_test, y_train, y_test, dropped = dataset(feature_set, split, test_size)
        st.write(f"{x_train.shape[1]} features gardées ({len(dropped)} supprimées), "
                 f"{len(x_train)} cycles d'entraînement, {len(x_test)} de test.")
        st.write(f"Entraînement de {len(names)} modèle(s)…")
        results, fitted, params = pipeline.train_models(
            names, x_train, y_train, x_test, y_test, search=None if search == "aucun" else search)
        status.update(label=f"Entraînement terminé en {time.perf_counter() - start:.0f} s",
                      state="complete", expanded=False)
    st.session_state.training = {
        "config": f"{FEATURE_LABELS[feature_set]} · {pipeline.SPLITS[split]} · {SEARCHES[search]}",
        "results": results, "fitted": fitted, "params": params,
        "x_test": x_test, "y_test": y_test, "features": list(x_train.columns), "dropped": dropped}

run = st.session_state.get("training")
if run is None:
    st.info("Choisis un jeu de features et des modèles, puis lance l'entraînement.",
            icon=":material/info:")
    st.stop()

results = run["results"]
best = results.iloc[0]
st.subheader("Résultats")
st.caption(run["config"])
with st.container(horizontal=True):
    st.metric("Meilleur modèle (F1 test)", results.index[0], border=True)
    st.metric("F1 macro test", f"{best['f1_macro']:.3f}", border=True)
    st.metric("F1 macro CV", f"{best['cv_f1_macro']:.3f}", help="Validation croisée sur le train",
              border=True)
    st.metric("AUC test", f"{best['auc']:.3f}", border=True)

st.dataframe(results[list(COLUMNS)].rename(columns=COLUMNS),
             column_config={label: st.column_config.NumberColumn(format="%.3f")
                            for label in COLUMNS.values()},
             alt="Scores de chaque modèle en validation croisée et sur le test")

if run["params"]:
    with st.expander("Meilleurs hyperparamètres", icon=":material/tune:"):
        st.dataframe(pd.DataFrame({"modèle": list(run["params"]),
                                   "paramètres": [str(p) for p in run["params"].values()]}),
                     hide_index=True, alt="Meilleurs hyperparamètres trouvés par modèle")

with st.expander(f"Features gardées ({len(run['features'])})", icon=":material/list:"):
    st.write(", ".join(run["features"]))
    if run["dropped"]:
        st.caption("Supprimées (constantes ou corrélées sur le train) : " + ", ".join(run["dropped"]))

st.subheader("Matrices de confusion et courbes ROC")
shown = st.multiselect("Modèles à afficher", list(run["fitted"]), default=list(results.index[:3]),
                       max_selections=4, key="shown_models")
if shown:
    models = {name: run["fitted"][name] for name in shown}
    fig = plot_confusion_matrices(models, run["x_test"], run["y_test"], show=False)
    st.pyplot(fig, alt="Matrices de confusion sur le test des modèles choisis")
    plt.close(fig)

    roc_col, _ = st.columns([3, 2])
    fig = plot_roc_curves(models, run["x_test"], run["y_test"], show=False)
    roc_col.pyplot(fig, alt="Courbes ROC sur le test des modèles choisis")
    plt.close(fig)
