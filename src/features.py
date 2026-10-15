import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


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


FEATURE_SETS = ["stats", "spectre", "stats_spectre", "histogramme"]


def histogram_edges(data: pd.DataFrame, bins: int = 20, n_fit: int | None = None) -> np.ndarray:
    """Bornes communes des histogrammes : min -> max des n_fit premiers cycles (None = tous)."""
    ref = data.values[:n_fit]
    return np.linspace(ref.min(), ref.max(), bins + 1)


def build_feature_set(name: str, ps2: pd.DataFrame, fs1: pd.DataFrame,
                      n_fit: int | None = None, bins: int = 20) -> pd.DataFrame:
    """Features brutes (avant sélection) d'une des méthodes de FEATURE_SETS,
    construites comme dans les notebooks d'exploration.
    n_fit : seuls les n_fit premiers cycles servent à fixer les bornes des histogrammes
    (None = tous les cycles)."""
    if name == "stats":
        X = build_features({"ps2": ps2, "fs1": fs1})
        X["qmm"] = X["ps2_mean"] / X["fs1_mean"]
        return X
    if name == "spectre":
        return pd.concat([build_frequence(ps2, fs=100, prefix="ps2_"),
                          build_frequence(fs1, fs=10, prefix="fs1_")], axis=1)
    if name == "stats_spectre":
        return pd.concat([build_features({"ps2": ps2, "fs1": fs1}),
                          build_frequence(ps2, fs=100, prefix="ps2_"),
                          build_frequence(fs1, fs=10, prefix="fs1_")], axis=1)
    if name == "histogramme":
        return pd.concat([build_histogramme(ps2, edges=histogram_edges(ps2, bins, n_fit), prefix="ps2_"),
                          build_histogramme(fs1, edges=histogram_edges(fs1, bins, n_fit), prefix="fs1_")],
                         axis=1)
    raise ValueError(f"Méthode inconnue : {name}. Choix : {FEATURE_SETS}")


def select_features(x_train: pd.DataFrame, x_test: pd.DataFrame,
                    constant_threshold: float = 1e-3, corr_threshold: float = 0.85):
    """Sélection des features décidée sur le train seulement, puis appliquée au test.
    Retourne (x_train, x_test, liste des colonnes supprimées)."""
    x_train, constant = drop_constant_features(x_train, constant_threshold)
    x_train, correlated = drop_correlated_features(x_train, corr_threshold)
    return x_train, x_test[x_train.columns], constant + correlated
