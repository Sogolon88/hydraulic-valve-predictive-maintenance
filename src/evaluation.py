import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.plots import finish


def get_scores(model, x):
    """Score de la classe 1 : probabilité si disponible, sinon distance à la frontière (SVC)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def test_scores(model, x_test, y_test) -> dict:
    pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_acc": balanced_accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro"),
        "recall_macro": recall_score(y_test, pred, average="macro"),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "auc": roc_auc_score(y_test, get_scores(model, x_test)),
    }


def plot_confusion_matrices(models: dict, x_test, y_test, save_as=None, show: bool = True):
    """Une matrice de confusion par modèle, côte à côte.
    show=False : renvoie la figure au lieu de l'afficher."""
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)
    for ax, (name, model) in zip(axes[0], models.items()):
        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, colorbar=False)
        ax.set_title(name)
    return finish(save_as, show)


def plot_roc_curves(models: dict, x_test, y_test, save_as=None, show: bool = True):
    """Courbes ROC de tous les modèles sur un même graphique (AUC dans la légende).
    show=False : renvoie la figure au lieu de l'afficher."""
    ax = plt.subplots(figsize=(6, 5))[1]
    for name, model in models.items():
        RocCurveDisplay.from_predictions(y_test, get_scores(model, x_test), name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="hasard")
    ax.legend()
    return finish(save_as, show)
