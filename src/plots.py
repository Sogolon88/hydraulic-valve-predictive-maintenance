from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import PROJECT_ROOT
from src.data import class_distribution


def save_figure(name: str, dpi: int = 150) -> Path:
    """Enregistre la figure courante dans figures/<name> (à appeler avant plt.show())."""
    path = PROJECT_ROOT / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    return path


def finish(save_as: str | None, show: bool = True):
    """Enregistre la figure courante si save_as est donné, puis l'affiche (notebook)
    ou la renvoie (show=False, pour Streamlit : st.pyplot(fig))."""
    if save_as:
        save_figure(save_as)
    if show:
        plt.show()
        return None
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_class_distribution(y: pd.Series, save_as: str | None = None) -> pd.DataFrame:
    """Affiche la répartition des classes (et l'enregistre si save_as est donné).
    Retourne le tableau des effectifs et pourcentages."""
    table = class_distribution(y)
    table["n"].plot.bar(rot=0)
    plt.title("Répartition des classes")
    plt.ylabel("Nombre de cycles")
    finish(save_as)
    return table


def plot_correlation_matrix(X: pd.DataFrame, save_as: str | None = None, annot: bool = True,
                            show: bool = True):
    """Heatmap de la matrice de corrélation, avec les valeurs."""
    n = X.shape[1]
    size = max(6, 0.55 * n)
    plt.figure(figsize=(size + 2, size))
    sns.heatmap(X.corr(), annot=annot, fmt=".2f", annot_kws={"size": 7},
                cmap="coolwarm", vmin=-1, vmax=1)
    return finish(save_as, show)


def plot_histograms(X: pd.DataFrame, save_as: str | None = None, figsize=(15, 10)):
    """Histogramme de chaque feature."""
    X.hist(figsize=figsize)
    finish(save_as)


def plot_boxplots(X: pd.DataFrame, save_as: str | None = None, figsize=(18, 6)):
    """Boîte à moustaches de chaque feature."""
    X.boxplot(figsize=figsize, rot=90)
    finish(save_as)
