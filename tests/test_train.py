import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.model import get_baseline_models
from src.train import tune_model

x, y = make_classification(200, 5, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


def test_baseline_models():
    models = get_baseline_models()
    assert len(models) == 6
    for model in models.values():
        model.fit(x_train, y_train)
        assert len(model.predict(x_test)) == len(y_test)


def test_tune_model():
    grid = {"clf__n_neighbors": [3, 5]}
    model, params, table = tune_model("knn", x_train, y_train, param_grid=grid, cv=3)
    assert params["clf__n_neighbors"] in [3, 5]
    assert len(table) == 2


def test_tune_model_inconnu():
    with pytest.raises(ValueError):
        tune_model("inconnu", x_train, y_train)
