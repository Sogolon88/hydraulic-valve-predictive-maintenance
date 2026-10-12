# Hydraulic valve predictive maintenance

Prédiction de l'état de la vanne d'un système hydraulique à partir de deux capteurs seulement :
**PS2** (pression) et **FS1** (débit). Cible : état de la vanne (`data/raw/profile.txt`).

## Structure

```
data/
  raw/            PS2.txt, FS1.txt, profile.txt (jamais modifiés)
  processed/      jeux train/test issus de l'extraction de features
notebooks/        exploration.ipynb, model_training.ipynb
src/  utils.py        chemins, chargement/sauvegarde, features, découpage, graphiques d'exploration
  model.py        configuration des modèles et grilles d'hyperparamètres
  train.py        entraînement, comparaison, recherche d'hyperparamètres
  evaluation.py   scores de test, matrices de confusion, courbes ROC/AUC
models/           modèles entraînés
figures/          graphiques du rapport
results/          métriques et prédictions
docs/             énoncé du projet
```

## Utilisation

```bash
pip install -r requirements.txt
pip install -e .      # une seule fois : rend `src` importable partout
jupyter notebook notebooks/
```

Dans un notebook : `from src.utils import load_data, save_data, load_processed`
(les chemins relatifs sont résolus depuis la racine du projet).

Modèles réglés : `save_models(tuned)` les enregistre dans `models/`, `load_model("random_forest")` les recharge.
