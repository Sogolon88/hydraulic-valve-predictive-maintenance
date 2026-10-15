"""Point d'entrée unique de la pipeline : chargement, features, entraînement, prédiction.
Les fonctions reprennent les étapes de notebook_projet.ipynb ; l'interface Streamlit (app.py)
n'appelle que ce module."""
from functools import lru_cache

import pandas as pd

from src.config import N_TRAIN, PIPELINE_SET, PROJECT_ROOT
from src.data import load_data, load_model, make_binary_target, split_data, split_first_n
from src.evaluation import get_scores
from src.features import build_feature_set, build_histogramme, histogram_edges, select_features
from src.model import get_baseline_models, get_ensemble_models, get_param_grids
from src.train import compare_models, tune_model

PROFILE_COLUMNS = ["cooler", "valve", "pump", "accumulator", "stable"]
SPLITS = {"enonce": f"Énoncé : {N_TRAIN} premiers cycles", "aleatoire": "Aléatoire stratifié"}


# -------données
def load_raw():
    """Capteurs bruts et profil : (PS2, FS1, profile), un cycle par ligne."""
    ps2 = load_data("data/raw/PS2.txt")
    fs1 = load_data("data/raw/FS1.txt")
    profile = load_data("data/raw/profile.txt")
    profile.columns = PROFILE_COLUMNS
    return ps2, fs1, profile


def make_target(profile: pd.DataFrame) -> pd.Series:
    """Cible : 1 si la vanne est à 100 % (optimale), 0 sinon."""
    return make_binary_target(profile["valve"], positive_value=100, name="valve_optimal")


def prepare_dataset(feature_set: str, ps2: pd.DataFrame, fs1: pd.DataFrame, y: pd.Series,
                    split: str = "enonce", n_train: int = N_TRAIN, test_size: float = 0.2,
                    random_state: int = 42):
    """Features d'un jeu de FEATURE_SETS, découpage train/test, puis sélection sur le train.
    split : "enonce" (n_train premiers cycles, bornes des histogrammes fixées sur le train)
    ou "aleatoire" (mélange stratifié, test_size).
    Retourne (x_train, x_test, y_train, y_test, colonnes supprimées)."""
    if split == "enonce":
        X = build_feature_set(feature_set, ps2, fs1, n_fit=n_train)
        x_train, x_test, y_train, y_test = split_first_n(X, y, n_train=n_train)
    elif split == "aleatoire":
        X = build_feature_set(feature_set, ps2, fs1)
        x_train, x_test, y_train, y_test = split_data(X, y, test_size=test_size,
                                                      random_state=random_state)
    else:
        raise ValueError(f"Découpage inconnu : {split}. Choix : {list(SPLITS)}")

    x_train, x_test, dropped = select_features(x_train, x_test)
    return x_train, x_test, y_train, y_test, dropped


# -------entraînement
def candidate_models(random_state: int = 42) -> dict:
    """Modèles simples et modèles d'ensemble, réglages par défaut (random_forest une seule fois)."""
    return {**get_baseline_models(random_state), **get_ensemble_models(random_state)}


def tunable_models() -> list[str]:
    """Modèles qui ont une grille d'hyperparamètres."""
    return list(get_param_grids())


def train_models(names: list[str], x_train, y_train, x_test, y_test, search: str | None = None,
                 n_iter: int = 10, cv: int = 5, random_state: int = 42):
    """Entraîne et compare les modèles demandés.
    search : None (réglages par défaut), "grid" ou "random" : les modèles qui ont une grille
    sont d'abord réglés par validation croisée sur le train, les autres gardent leurs réglages.
    Retourne (tableau trié par f1_macro sur le test, modèles entraînés, meilleurs paramètres)."""
    models = candidate_models(random_state)
    unknown = set(names) - set(models)
    if unknown:
        raise ValueError(f"Modèles inconnus : {sorted(unknown)}. Choix : {list(models)}")

    selected, best_params = {}, {}
    for name in names:
        if search and name in get_param_grids():
            selected[name], best_params[name], _ = tune_model(
                name, x_train, y_train, search=search, n_iter=n_iter, cv=cv,
                random_state=random_state)
        else:
            selected[name] = models[name]

    results, fitted = compare_models(selected, x_train, y_train, x_test, y_test,
                                     cv=cv, random_state=random_state)
    return results, fitted, best_params


# -------prédiction d'un cycle
@lru_cache(maxsize=None)
def _training_edges(bins: int):
    """Bornes des histogrammes (PS2, FS1) : min/max des N_TRAIN premiers cycles, comme à l'entraînement."""
    return tuple(histogram_edges(load_data(f"data/raw/{s}.txt"), bins, n_fit=N_TRAIN)
                 for s in ("PS2", "FS1"))


@lru_cache(maxsize=None)
def _kept_columns(feature_set: str):
    """Colonnes gardées après sélection, lues dans data/processed/<feature_set>/x_train.csv."""
    return pd.read_csv(PROJECT_ROOT / "data/processed" / feature_set / "x_train.csv", nrows=0).columns


def cycle_features(ps2: pd.DataFrame, fs1: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """Features histogramme d'UN cycle, avec les mêmes colonnes que celles gardées à l'entraînement."""
    if len(ps2) != 1 or len(fs1) != 1:
        raise ValueError("ps2 et fs1 doivent contenir un seul cycle (une ligne)")

    # valeurs manquantes -> moyenne du cycle
    ps2 = ps2.fillna(ps2.mean(axis=1).iloc[0])
    fs1 = fs1.fillna(fs1.mean(axis=1).iloc[0])

    edges_ps2, edges_fs1 = _training_edges(bins)
    X = pd.concat([build_histogramme(ps2, edges=edges_ps2, prefix="ps2_"),
                   build_histogramme(fs1, edges=edges_fs1, prefix="fs1_")], axis=1)
    return X[_kept_columns(PIPELINE_SET)]


def _as_model(model):
    """Un nom devient le modèle enregistré dans models/PIPELINE_SET/ ; un modèle est gardé tel quel."""
    return load_model(model, PIPELINE_SET) if isinstance(model, str) else model


def predict_cycle(ps2: pd.DataFrame, fs1: pd.DataFrame, model="gradient_boosting") -> int:
    """Prédit l'état de la vanne pour un cycle : 1 = optimale, 0 = sinon.
    ps2, fs1 : DataFrames d'une ligne. Valeurs manquantes remplacées par la moyenne du cycle.
    model : nom d'un modèle de models/PIPELINE_SET/ ou modèle déjà chargé."""
    return int(_as_model(model).predict(cycle_features(ps2, fs1))[0])


def cycle_score(ps2: pd.DataFrame, fs1: pd.DataFrame, model="gradient_boosting") -> float:
    """Score de la classe « optimale » pour un cycle : probabilité si le modèle en donne une,
    sinon distance à la frontière (SVC)."""
    return float(get_scores(_as_model(model), cycle_features(ps2, fs1))[0])
