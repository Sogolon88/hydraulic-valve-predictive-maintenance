import numpy as np
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _pipe(clf, scale: bool) -> Pipeline:
    """Pipeline à une étape 'clf' (et 'scaler' si le modèle est sensible à l'échelle).
    La standardisation est ajustée sur le train seulement, y compris en validation croisée."""
    steps = [("scaler", StandardScaler())] if scale else []
    return Pipeline(steps + [("clf", clf)])


def get_baseline_models(random_state: int = 42, class_weight: str | None = "balanced") -> dict:
    """Modèles de base. Les hyperparamètres se règlent avec 'clf__<nom>'."""
    return {
        "knn": _pipe(KNeighborsClassifier(n_neighbors=5), scale=True),
        "decision_tree": _pipe(
            DecisionTreeClassifier(random_state=random_state, class_weight=class_weight), scale=False),
        "logistic_regression": _pipe(
            LogisticRegression(max_iter=2000, class_weight=class_weight), scale=True),
        "naive_bayes": _pipe(GaussianNB(), scale=False),
        "svm": _pipe(SVC(class_weight=class_weight), scale=True),
        "random_forest": _pipe(
            RandomForestClassifier(n_estimators=200, random_state=random_state,
                                   n_jobs=-1, class_weight=class_weight), scale=False),
    }


def _members(random_state: int, class_weight: str | None) -> list:
    """Modèles de base variés (linéaire, distance, marge, arbres) combinés par le vote et le stacking.
    SVC avec probability=True pour que le vote 'soft' puisse moyenner des probabilités."""
    base = get_baseline_models(random_state, class_weight)
    svm = _pipe(SVC(class_weight=class_weight, probability=True, random_state=random_state),
                scale=True)
    return [("logistic_regression", base["logistic_regression"]), ("knn", base["knn"]),
            ("svm", svm), ("random_forest", base["random_forest"])]


def get_ensemble_models(random_state: int = 42, class_weight: str | None = "balanced") -> dict:
    """Modèles d'ensemble, en trois familles :
    - bagging : beaucoup d'arbres entraînés indépendamment sur des tirages du train, puis vote ;
    - boosting : des modèles simples ajoutés un par un, chacun corrigeant les erreurs des précédents ;
    - vote et stacking : combinaison de modèles de natures différentes.
    Pas de n_jobs sur bagging, vote et stacking : ils paralléliseraient avec des processus,
    très lents à démarrer sous Windows. Les forêts gardent n_jobs=-1 (threads)."""
    tree = DecisionTreeClassifier(random_state=random_state, class_weight=class_weight)
    return {
        "bagging": _pipe(BaggingClassifier(tree, n_estimators=200, random_state=random_state),
                         scale=False),
        "random_forest": get_baseline_models(random_state, class_weight)["random_forest"],
        "extra_trees": _pipe(ExtraTreesClassifier(n_estimators=300, random_state=random_state,
                                                  n_jobs=-1, class_weight=class_weight), scale=False),
        "adaboost": _pipe(AdaBoostClassifier(n_estimators=200, random_state=random_state),
                          scale=False),
        "gradient_boosting": _pipe(HistGradientBoostingClassifier(random_state=random_state,
                                                                  class_weight=class_weight),
                                   scale=False),
        "vote_soft": VotingClassifier(_members(random_state, class_weight), voting="soft"),
        "stacking": StackingClassifier(_members(random_state, class_weight),
                                       final_estimator=LogisticRegression(max_iter=2000), cv=5),
    }


def get_param_grids() -> dict:
    """Grilles d'hyperparamètres par défaut des modèles de base et des modèles de boosting."""
    return {
        "knn": {
            "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
        "decision_tree": {
            "clf__max_depth": [None, 3, 5, 8, 12],
            "clf__min_samples_leaf": [1, 3, 5, 10],
            "clf__criterion": ["gini", "entropy"],
        },
        "logistic_regression": {
            "clf__C": [0.01, 0.1, 1, 10, 100],
        },
        "naive_bayes": {
            "clf__var_smoothing": list(np.logspace(-11, -5, 7)),
        },
        "svm": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", 0.01, 0.1, 1],
            "clf__kernel": ["rbf"],
        },
        "random_forest": {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 8, 16],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", 0.5],
        },
        "gradient_boosting": {
            "clf__learning_rate": [0.03, 0.1, 0.3],
            "clf__max_depth": [None, 3, 6],
            "clf__max_iter": [100, 300],
        },
        "adaboost": {
            "clf__n_estimators": [50, 200, 500],
            "clf__learning_rate": [0.1, 0.5, 1.0],
        },
    }
