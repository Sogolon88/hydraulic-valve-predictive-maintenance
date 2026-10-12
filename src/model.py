import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


def get_param_grids() -> dict:
    """Grilles d'hyperparamètres par défaut pour chaque modèle de base."""
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
    }
