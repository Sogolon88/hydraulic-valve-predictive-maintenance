from pathlib import Path

# Racine du projet, indépendante du dossier d'exécution
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Découpage de l'énoncé (notebook_projet.ipynb) : les 2000 premiers cycles pour construire le modèle
N_TRAIN = 2000

# Modèles et features utilisés pour la prédiction d'un cycle (models/ et data/processed/)
PIPELINE_SET = "histogramme_enonce"
