import numpy as np
import pandas as pd
import pytest

import src.pipeline as pipeline
from src.config import PIPELINE_SET, PROJECT_ROOT
from src.data import load_data

rng = np.random.default_rng(0)
ps2 = pd.DataFrame(rng.normal(150, 5, size=(40, 600)))
fs1 = pd.DataFrame(rng.normal(8, 1, size=(40, 60)))
y = pd.Series([0, 1] * 20)


@pytest.mark.parametrize("split", list(pipeline.SPLITS))
def test_prepare_dataset(split):
    x_train, x_test, y_train, y_test, dropped = pipeline.prepare_dataset(
        "stats", ps2, fs1, y, split=split, n_train=30, test_size=0.25)

    assert len(x_train) == len(y_train) == 30 and len(x_test) == len(y_test) == 10
    assert list(x_test.columns) == list(x_train.columns)   # sélection décidée sur le train
    assert not set(dropped) & set(x_train.columns)

    with pytest.raises(ValueError):
        pipeline.prepare_dataset("stats", ps2, fs1, y, split="inconnu")


def test_train_models_avec_et_sans_reglage():
    x_train, x_test, y_train, y_test, _ = pipeline.prepare_dataset("stats", ps2, fs1, y, n_train=30)

    results, fitted, params = pipeline.train_models(["knn", "naive_bayes"], x_train, y_train,
                                                    x_test, y_test, cv=3)
    assert set(results.index) == set(fitted) == {"knn", "naive_bayes"} and params == {}

    _, _, params = pipeline.train_models(["knn", "vote_soft"], x_train, y_train, x_test, y_test,
                                         search="random", n_iter=2, cv=3)
    assert set(params) == {"knn"}                          # vote_soft n'a pas de grille

    with pytest.raises(ValueError):
        pipeline.train_models(["inconnu"], x_train, y_train, x_test, y_test)


def test_cycle_features_reproduit_les_features_d_entrainement():
    dossier = PROJECT_ROOT / "data/processed" / PIPELINE_SET
    if not (dossier / "x_train.csv").exists():
        pytest.skip("jeu histogramme absent")

    PS2, FS1 = load_data("data/raw/PS2.txt"), load_data("data/raw/FS1.txt")
    train = pd.concat([pd.read_csv(dossier / "x_train.csv"), pd.read_csv(dossier / "x_test.csv")])

    for i in range(5):
        feats = pipeline.cycle_features(PS2.iloc[[i]], FS1.iloc[[i]])
        ecart = np.abs(train.values - feats.values).max(axis=1)
        assert (ecart < 1e-9).any()            # ce cycle existe dans le jeu d'entraînement ou de test


def test_predict_cycle_valeurs_manquantes():
    if not (PROJECT_ROOT / "models" / PIPELINE_SET / "gradient_boosting.pkl").exists():
        pytest.skip("modèle absent")

    PS2, FS1 = load_data("data/raw/PS2.txt"), load_data("data/raw/FS1.txt")
    ps2_cycle, fs1_cycle = PS2.iloc[[0]].copy(), FS1.iloc[[0]].copy()
    ps2_cycle.iloc[0, :50] = np.nan                                     # 50 valeurs manquantes

    attendu = pipeline.cycle_features(ps2_cycle.fillna(ps2_cycle.mean(axis=1).iloc[0]), fs1_cycle)
    assert pipeline.cycle_features(ps2_cycle, fs1_cycle).equals(attendu)  # moyenne du cycle
    assert pipeline.predict_cycle(ps2_cycle, fs1_cycle) in (0, 1)
    assert 0 <= pipeline.cycle_score(ps2_cycle, fs1_cycle) <= 1

    with pytest.raises(ValueError):
        pipeline.predict_cycle(PS2.iloc[:2], FS1.iloc[:2])               # un seul cycle attendu
