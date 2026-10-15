# Hydraulic valve predictive maintenance

Prédiction de l'état de la vanne d'un système hydraulique à partir de deux capteurs seulement :
**PS2** (pression) et **FS1** (débit). Cible : état de la vanne (`data/raw/profile.txt`).

## Structure

```
data/
  raw/            PS2.txt, FS1.txt, profile.txt (jamais modifiés)
  processed/      jeux train/test issus de l'extraction de features
notebooks/        exploration.ipynb, model_training.ipynb
src/  pipeline.py     point d'entrée unique : données, features, entraînement, prédiction d'un cycle
  config.py       chemins et constantes (N_TRAIN, PIPELINE_SET)
  data.py         chargement/sauvegarde des données et des modèles, cible, découpage
  features.py     extraction et sélection des features
  plots.py        graphiques d'exploration
  utils.py        ré-exporte les fonctions ci-dessus (imports des notebooks)
  model.py        configuration des modèles et grilles d'hyperparamètres
  train.py        entraînement, comparaison, recherche d'hyperparamètres
  evaluation.py   scores de test, matrices de confusion, courbes ROC/AUC
models/           modèles entraînés
figures/          graphiques du rapport
results/          métriques et prédictions
app.py, app_pages/  interface Streamlit
docs/             énoncé du projet
```

## Utilisation

```bash
pip install -r requirements.txt
pip install -e .      # une seule fois : rend `src` importable partout
jupyter notebook notebooks/
streamlit run app.py  # interface : données, entraînement, prédiction
```

La page Prédiction utilise les modèles enregistrés par `notebook_projet.ipynb` dans `models/histogramme_enonce/`.

Dans un notebook : `from src.utils import load_data, save_data, load_processed`
(les chemins relatifs sont résolus depuis la racine du projet).

Modèles réglés : `save_models(tuned)` les enregistre dans `models/`, `load_model("random_forest")` les recharge.
