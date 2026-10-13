import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.model import get_baseline_models, get_ensemble_models
from src.train import compare_models, tune_model

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


def test_tune_model_ensemble():
    grid = {"clf__learning_rate": [0.1, 0.3]}
    model, params, table = tune_model("gradient_boosting", x_train, y_train, param_grid=grid, cv=3)
    assert params["clf__learning_rate"] in [0.1, 0.3]


def test_tune_model_inconnu():
    with pytest.raises(ValueError):
        tune_model("inconnu", x_train, y_train)


def test_ensemble_models():
    models = get_ensemble_models()
    assert {"bagging", "adaboost", "vote_soft", "stacking"} <= set(models)
    for model in models.values():
        model.fit(x_train, y_train)
        assert len(model.predict(x_test)) == len(y_test)


def test_compare_models():
    models = {k: v for k, v in get_baseline_models().items() if k in ("knn", "naive_bayes")}
    results, fitted = compare_models(models, x_train, y_train, x_test, y_test, cv=3)
    assert set(results.index) == set(fitted) == {"knn", "naive_bayes"}
    assert {"train_f1_macro", "cv_f1_macro", "cv_std", "f1_macro", "auc"} <= set(results.columns)
