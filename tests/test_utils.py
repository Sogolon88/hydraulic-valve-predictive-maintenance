# tests/test_utils.py
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import src.data as src_data
import src.utils as utils
from src.model import get_baseline_models


def test_save_and_load_model(tmp_path, monkeypatch):
    monkeypatch.setattr(src_data, "PROJECT_ROOT", tmp_path)   # écrit dans un dossier temporaire

    x, y = make_classification(100, 4, random_state=0)
    model = get_baseline_models()["knn"].fit(x, y)

    src_data.save_models({"knn": model}, "stats")

    assert (tmp_path / "models" / "stats" / "knn.pkl").exists()           # fichier créé
    assert src_data.list_models("stats") == ["knn"]
    loaded = src_data.load_model("knn", "stats")
    assert np.array_equal(loaded.predict(x), model.predict(x))


def test_split_data():
    x = pd.DataFrame({"a": range(200)})
    y = pd.Series([0, 1] * 100)

    x_train, x_test, y_train, y_test = utils.split_data(x, y, test_size=0.15)

    assert len(x_test) == 30                              # 15 %
    assert set(x_train.index).isdisjoint(x_test.index)    # pas de fuite train/test
    assert y_test.mean() == 0.5                           # classes conservées


def test_split_data_longueurs_differentes():
    with pytest.raises(ValueError):
        utils.split_data(pd.DataFrame({"a": range(10)}), pd.Series(range(9)))


def test_split_first_n():
    x = pd.DataFrame({"a": range(10)})
    y = pd.Series(range(10))

    x_train, x_test, y_train, y_test = utils.split_first_n(x, y, n_train=7)

    assert x_train["a"].tolist() == list(range(7))        # les 7 premiers, dans l'ordre
    assert y_test.tolist() == [7, 8, 9]                   # les suivants pour le test

    with pytest.raises(ValueError):
        utils.split_first_n(x, y, n_train=10)             # test vide


def test_build_feature_set():
    rng = np.random.default_rng(0)
    ps2 = pd.DataFrame(rng.normal(150, 5, size=(6, 600)))
    fs1 = pd.DataFrame(rng.normal(8, 1, size=(6, 60)))

    for name in utils.FEATURE_SETS:
        X = utils.build_feature_set(name, ps2, fs1, n_fit=4)
        assert len(X) == 6 and not X.isna().any().any()

    with pytest.raises(ValueError):
        utils.build_feature_set("inconnu", ps2, fs1)


def test_select_features_decide_sur_le_train():
    x_train = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8], "c": [5.0, 5, 5, 5]})
    x_test = pd.DataFrame({"a": [9.0], "b": [0.0], "c": [7.0]})

    x_train_s, x_test_s, dropped = utils.select_features(x_train, x_test)

    assert sorted(dropped) == ["b", "c"]                  # c constante, b copie de a
    assert list(x_test_s.columns) == list(x_train_s.columns) == ["a"]


def test_make_binary_target():
    y = utils.make_binary_target(pd.Series([100, 90, 73, 100]), 100)
    assert y.tolist() == [1, 0, 0, 1]


def test_build_histogramme():
    data = pd.DataFrame(np.random.default_rng(0).normal(size=(5, 200)), index=[10, 11, 12, 13, 14])

    features = utils.build_histogramme(data, bins=10, prefix="ps2_")

    assert features.shape == (5, 10)                      # 10 intervalles par cycle
    assert features.index.tolist() == data.index.tolist() # index conservé
    assert np.allclose(features.sum(axis=1), 1)           # proportions : somme = 1
    assert list(features.columns)[0] == "ps2_bin_0"
