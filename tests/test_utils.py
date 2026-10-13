# tests/test_utils.py
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import src.utils as utils
from src.model import get_baseline_models


def test_save_and_load_model(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "PROJECT_ROOT", tmp_path)   # écrit dans un dossier temporaire

    x, y = make_classification(100, 4, random_state=0)
    model = get_baseline_models()["knn"].fit(x, y)

    utils.save_models({"knn": model})

    assert (tmp_path / "models" / "knn.pkl").exists()           # fichier créé
    loaded = utils.load_model("knn")
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

