from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROJECT_ROOT


# -------fichiers de données
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


# -------modèles enregistrés
def save_models(models: dict, name_folder) -> None:
    """Enregistre chaque modèle (pipeline complet) dans models/<name_folder>/<nom>.pkl."""
    import joblib

    folder = PROJECT_ROOT / "models" / name_folder
    folder.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, folder / f"{name}.pkl")


def load_model(name: str, name_folder):
    """Recharge un modèle enregistré depuis models/<name_folder>/<nom>.pkl."""
    import joblib

    return joblib.load(PROJECT_ROOT / "models" / name_folder / f"{name}.pkl")


def list_models(name_folder) -> list[str]:
    """Noms des modèles enregistrés dans models/<name_folder>/."""
    return sorted(p.stem for p in (PROJECT_ROOT / "models" / name_folder).glob("*.pkl"))


# -------cible et découpage
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


def split_first_n(x: pd.DataFrame, y: pd.Series, n_train: int = 2000):
    """Découpage imposé par l'énoncé : les n_train premiers cycles pour construire le modèle,
    les suivants pour le test final. Pas de mélange, l'ordre des cycles est conservé.
    Retourne x_train, x_test, y_train, y_test."""
    if len(x) != len(y):
        raise ValueError(f"x ({len(x)}) et y ({len(y)}) n'ont pas la même longueur")
    if not 0 < n_train < len(x):
        raise ValueError(f"n_train doit être entre 1 et {len(x) - 1}")

    return x.iloc[:n_train], x.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]


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
