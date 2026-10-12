from sklearn.datasets import make_classification

import src.evaluation as evaluation
from src.model import get_baseline_models

x, y = make_classification(200, 5, random_state=0)


def test_scores_entre_0_et_1():
    model = get_baseline_models()["knn"].fit(x, y)
    scores = evaluation.test_scores(model, x, y)

    assert set(scores) == {"accuracy", "balanced_acc", "precision_macro",
                           "recall_macro", "f1_macro", "auc"}
    assert all(0 <= v <= 1 for v in scores.values())


def test_get_scores_proba_et_svm():
    models = get_baseline_models()
    for name in ["knn", "svm"]:            # predict_proba / decision_function
        score = evaluation.get_scores(models[name].fit(x, y), x)
        assert len(score) == len(y)
