from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Racine du projet, indépendante du dossier d'exécution
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def centrer_reduire(x_train: pd.DataFrame, x_test: pd.DataFrame | None = None):
    """Standardise les features : fit sur le train, transform sur le train et le test."""
    scaler = StandardScaler()

    x_train_s = pd.DataFrame(
        scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index
    )
    if x_test is None:
        return x_train_s, scaler

    x_test_s = pd.DataFrame(
        scaler.transform(x_test), columns=x_test.columns, index=x_test.index
    )
    return x_train_s, x_test_s, scaler


def save_figure(name: str, dpi: int = 150) -> Path:
    """Enregistre la figure courante dans figures/<name> (à appeler avant plt.show())."""
    import matplotlib.pyplot as plt

    path = PROJECT_ROOT / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    return path


def save_models(models: dict) -> None:
    """Enregistre chaque modèle (pipeline complet) dans models/<nom>.pkl."""
    import joblib

    folder = PROJECT_ROOT / "models"
    folder.mkdir(exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, folder / f"{name}.pkl")


def load_model(name: str):
    """Recharge un modèle enregistré depuis models/<nom>.pkl."""
    import joblib

    return joblib.load(PROJECT_ROOT / "models" / f"{name}.pkl")


def split_data(x: pd.DataFrame, y: pd.Series, test_size: float = 0.15,
               random_state: int = 42, stratify: bool = True):
    """Mélange puis sépare en train/test.
    stratify=True garde la même proportion de classes dans les deux jeux.
    Retourne x_train, x_test, y_train, y_test."""
    if len(x) != len(y):
        raise ValueError(f"x ({len(x)}) et y ({len(y)}) n'ont pas la même longueur")

    return train_test_split(
        x, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y if stratify else None,
    )


def class_distribution(y: pd.Series) -> pd.DataFrame:
    """Effectif et pourcentage de chaque classe."""
    counts = y.value_counts().sort_index()
    return pd.DataFrame({"n": counts, "%": (100 * counts / len(y)).round(1)})


def make_binary_target(values: pd.Series, positive_value, name: str | None = None) -> pd.Series:
    """Cible binaire : 1 si values == positive_value (ou est dans la liste), 0 sinon."""
    if isinstance(positive_value, (list, tuple, set)):
        y = values.isin(positive_value).astype(int)
    else:
        y = (values == positive_value).astype(int)
    y.name = name or values.name
    return y


def missing_summary(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Nombre de valeurs manquantes par jeu de données."""
    return pd.DataFrame({
        "valeurs_manquantes": {k: int(df.isna().sum().sum()) for k, df in datasets.items()},
        "cellules": {k: int(df.size) for k, df in datasets.items()},
    })


import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def feature_reduction_stats(data: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """
    Résume chaque cycle (chaque ligne du DataFrame) par un ensemble
    de caractéristiques statistiques.

    data : DataFrame de forme (n_cycles, n_points), un cycle par ligne.
    prefix : préfixe ajouté aux noms de colonnes (ex. "PS2", "FS1").
    """
    values = data.to_numpy(dtype=float)

    q25, q75 = np.percentile(values, [25, 75], axis=1)

    features = pd.DataFrame(
        {
            "min": values.min(axis=1),
            "max": values.max(axis=1),
            "mean": values.mean(axis=1),
            "std": values.std(axis=1),
            "median": np.median(values, axis=1),
            "q25": q25,
            "q75": q75,
            "iqr": q75 - q25,
            "rms": np.sqrt(np.mean(values**2, axis=1)),
            "skew": skew(values, axis=1),
            "kurtosis": kurtosis(values, axis=1),
        },
        index=data.index,
    )

    features = features.fillna(0.0)

    if prefix:
        features = features.add_prefix(f"{prefix}_")

    return features

def build_frequence(data: pd.DataFrame, fs: float, prefix: str = "") -> pd.DataFrame:
    fft = np.abs(np.fft.rfft(data.values, axis=1))[:, 1:]        # sans la composante continue
    freqs = np.fft.rfftfreq(data.shape[1], d=1 / fs)[1:]

    features = pd.DataFrame({
        "amplitude": fft.max(axis=1),
        "dominant_freq": freqs[fft.argmax(axis=1)],
        "energie": (fft ** 2).sum(axis=1),
        "centroide": (fft * freqs).sum(axis=1) / fft.sum(axis=1),
    }, index=data.index)

    return features.add_prefix(prefix)


def build_histogramme(data: pd.DataFrame, bins: int = 20, edges=None, prefix: str = "") -> pd.DataFrame:
    """Chaque cycle (ligne) devient un histogramme : proportion de mesures dans chaque intervalle.
    Les intervalles sont communs à tous les cycles (min -> max de data, sans utiliser y)."""
    if edges is None:
        edges = np.linspace(data.values.min(), data.values.max(), bins + 1)

    counts = np.array([np.histogram(row, bins=edges)[0] for row in data.values])
    features = pd.DataFrame(counts / data.shape[1],
                            columns=[f"bin_{i}" for i in range(len(edges) - 1)],
                            index=data.index)
    return features.add_prefix(prefix)


def build_features(sensors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calcule les features de chaque capteur et les concatène en colonnes.
    sensors : {"ps2": PS2, "fs1": FS1, ...} (un cycle par ligne)."""
    return pd.concat(
        [feature_reduction_stats(df, name) for name, df in sensors.items()],
        axis=1,
    )


def drop_constant_features(X: pd.DataFrame, threshold: float = 1e-3):
    """Supprime les features constantes ou quasi constantes (écart-type < threshold).
    Retourne (X sans ces colonnes, liste des colonnes supprimées)."""
    dropped = X.columns[X.std() < threshold].tolist()
    return X.drop(columns=dropped), dropped


def drop_correlated_features(X: pd.DataFrame, threshold: float = 0.95):
    """Garde une seule feature par groupe de features corrélées (|corr| > threshold).
    Retourne (X sans les colonnes redondantes, liste des colonnes supprimées)."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if (upper[c] > threshold).any()]
    return X.drop(columns=dropped), dropped


def count_outliers_iqr(X: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Nombre et pourcentage de valeurs aberrantes par feature (règle de l'IQR :
    hors de [Q1 - k*IQR ; Q3 + k*IQR])."""
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    n = ((X < q1 - k * iqr) | (X > q3 + k * iqr)).sum()
    return (
        pd.DataFrame({"n_outliers": n, "%": (100 * n / len(X)).round(1)})
        .sort_values("n_outliers", ascending=False)
    )

import pandas as pd
from pathlib import Path


from src.utils import PROJECT_ROOT


def load_data(path: str | Path, header=None, **kwargs) -> pd.DataFrame:
    """Charge un fichier de données selon son extension."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()

    if ext in {".csv", ".txt", ".tsv", ".dat"}:
        # sep=None + engine="python" : détecte le séparateur (tab, ;, ',', espace)
        return pd.read_csv(path, sep=None, engine="python", header=header, **kwargs)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, header=header, **kwargs)
    if ext == ".json":
        return pd.read_json(path, **kwargs)
    if ext == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if ext == ".feather":
        return pd.read_feather(path, **kwargs)
    if ext == ".pkl":
        return pd.read_pickle(path, **kwargs)

    raise ValueError(f"Format non supporté : {ext}")


def save_data(x_train, x_test, y_train, y_test, name: str) -> Path:
    """Enregistre les jeux déjà séparés dans data/processed/<name>/ (4 fichiers CSV)."""
    path = PROJECT_ROOT / "data" / "processed" / name
    path.mkdir(parents=True, exist_ok=True)

    def as_frame(y):
        return y.to_frame() if isinstance(y, pd.Series) else y

    x_train.to_csv(path / "x_train.csv", index=False)
    x_test.to_csv(path / "x_test.csv", index=False)
    as_frame(y_train).to_csv(path / "y_train.csv", index=False)
    as_frame(y_test).to_csv(path / "y_test.csv", index=False)

    return path


def load_processed(name: str):
    """Charge x_train, x_test, y_train, y_test depuis data/processed/<name>/.
    y est renvoyé sous forme de Series (une seule colonne)."""
    path = PROJECT_ROOT / "data" / "processed" / name
    x_train = pd.read_csv(path / "x_train.csv")
    x_test = pd.read_csv(path / "x_test.csv")
    y_train = pd.read_csv(path / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(path / "y_test.csv").iloc[:, 0]
    return x_train, x_test, y_train, y_test


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import class_distribution, save_figure


def _finish(save_as: str | None):
    if save_as:
        save_figure(save_as)
    plt.show()


def plot_class_distribution(y: pd.Series, save_as: str | None = None) -> pd.DataFrame:
    """Affiche la répartition des classes (et l'enregistre si save_as est donné).
    Retourne le tableau des effectifs et pourcentages."""
    table = class_distribution(y)
    table["n"].plot.bar(rot=0)
    plt.title("Répartition des classes")
    plt.ylabel("Nombre de cycles")
    _finish(save_as)
    return table


def plot_correlation_matrix(X: pd.DataFrame, save_as: str | None = None, annot: bool = True):
    """Heatmap de la matrice de corrélation, avec les valeurs."""
    n = X.shape[1]
    size = max(6, 0.55 * n)
    plt.figure(figsize=(size + 2, size))
    sns.heatmap(X.corr(), annot=annot, fmt=".2f", annot_kws={"size": 7},
                cmap="coolwarm", vmin=-1, vmax=1)
    _finish(save_as)


def plot_histograms(X: pd.DataFrame, save_as: str | None = None, figsize=(15, 10)):
    """Histogramme de chaque feature."""
    X.hist(figsize=figsize)
    _finish(save_as)


def plot_boxplots(X: pd.DataFrame, save_as: str | None = None, figsize=(18, 6)):
    """Boîte à moustaches de chaque feature."""
    X.boxplot(figsize=figsize, rot=90)
    _finish(save_as)
