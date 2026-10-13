import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.evaluation import test_scores
from src.model import get_baseline_models, get_param_grids


def _make_cv(cv, random_state: int = 42):
    """Un entier devient une CV stratifiée et mélangée ; un objet CV est gardé tel quel."""
    if isinstance(cv, int):
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    return cv


# -------hyperparamètres
def tune_model(name: str, x_train, y_train, param_grid: dict | None = None,
               search: str = "grid", n_iter: int = 30, cv: int = 5,
               scoring: str = "f1_macro", random_state: int = 42, n_jobs: int = -1):
    """Recherche d'hyperparamètres d'un modèle de base, sur le train uniquement.

    search : "grid" (toutes les combinaisons) ou "random" (n_iter tirages).
    Retourne (meilleur modèle réentraîné, meilleurs paramètres, tableau trié des résultats).
    """
    models = get_baseline_models(random_state)
    if name not in models:
        raise ValueError(f"Modèle inconnu : {name}. Choix : {list(models)}")
    param_grid = param_grid or get_param_grids()[name]
    splitter = _make_cv(cv, random_state)

    if search == "grid":
        searcher = GridSearchCV(models[name], param_grid, cv=splitter,
                                scoring=scoring, n_jobs=n_jobs)
    elif search == "random":
        searcher = RandomizedSearchCV(models[name], param_grid, n_iter=n_iter, cv=splitter,
                                      scoring=scoring, n_jobs=n_jobs, random_state=random_state)
    else:
        raise ValueError("search doit valoir 'grid' ou 'random'")

    searcher.fit(x_train, np.ravel(y_train))

    table = pd.DataFrame(searcher.cv_results_)
    table = table[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    table = table.sort_values("rank_test_score").reset_index(drop=True)
    return searcher.best_estimator_, searcher.best_params_, table


def tune_all(x_train, y_train, x_test, y_test, names: list[str] | None = None,
             search: str = "grid", n_iter: int = 30, cv: int = 5,
             scoring: str = "f1_macro", random_state: int = 42):
    """Règle les hyperparamètres de chaque modèle (CV sur le train), puis évalue
    UNE SEULE FOIS les meilleurs sur le test.
    Retourne (tableau de comparaison, modèles réglés)."""
    y_test = np.ravel(y_test)
    names = names or list(get_baseline_models(random_state))

    rows, tuned = [], {}
    for name in names:
        start = time.perf_counter()
        model, params, table = tune_model(name, x_train, y_train, search=search, n_iter=n_iter,
                                      cv=cv, scoring=scoring, random_state=random_state)
        elapsed = time.perf_counter() - start

        best_cv = table.loc[0, "mean_test_score"]
        rows.append({"model": name, "cv_f1_macro": best_cv,
                     **test_scores(model, x_test, y_test),
                     "search_time_s": elapsed, "best_params": params})
        tuned[name] = model

    results = pd.DataFrame(rows).set_index("model").sort_values("f1_macro", ascending=False)
    return results, tuned

