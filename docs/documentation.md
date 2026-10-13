# Documentation du projet

## 1. Ce que l'on cherche à faire

Une machine hydraulique tourne par cycles de 60 secondes. Pendant chaque cycle, deux capteurs enregistrent des mesures :

- PS2 : la pression, en bar, 100 mesures par seconde, donc 6000 valeurs par cycle.
- FS1 : le débit, en litres par minute, 10 mesures par seconde, donc 600 valeurs par cycle.

Pour chaque cycle, on connaît aussi l'état de la vanne : 100 %, 90 %, 80 % ou 73 %.

Le but est de construire un modèle qui, à partir de PS2 et FS1, prédit pour un cycle si la vanne est optimale (100 %) ou non. C'est une classification binaire :

- y = 1 : vanne à 100 %
- y = 0 : vanne à 90, 80 ou 73 %

## 2. Les données

Les fichiers bruts sont dans `data/raw/` et ne sont jamais modifiés.

| Fichier | Taille | Contenu |
|---|---|---|
| `PS2.txt` | 2205 lignes, 6000 colonnes | une ligne = un cycle de pression |
| `FS1.txt` | 2205 lignes, 600 colonnes | une ligne = un cycle de débit |
| `profile.txt` | 2205 lignes, 5 colonnes | état du système pour chaque cycle |

Les 5 colonnes de `profile.txt`, dans l'ordre : refroidisseur, vanne, pompe, accumulateur, stabilité. Seule la colonne vanne (indice 1) nous intéresse.

Il n'y a aucune valeur manquante et aucune variable catégorielle.

Les classes sont équilibrées : environ 51 % de cycles avec la vanne à 100 %, 49 % sinon.

## 3. Organisation du dossier

```
data/raw/          fichiers bruts
data/processed/    jeux train et test déjà préparés, un sous dossier par méthode
notebooks/         les notebooks Jupyter
src/utils.py       chargement, création des features, sélection, découpage, graphiques
src/model.py       liste des modèles et grilles d'hyperparamètres
src/train.py       entraînement, validation croisée, recherche d'hyperparamètres
src/evaluation.py  métriques, matrices de confusion, courbes ROC
tests/             tests unitaires (pytest)
figures/           images enregistrées pour le rapport
docs/              énoncé et cette documentation
```

## 4. Le problème de départ : trop de colonnes

Un cycle contient 6600 valeurs brutes. On ne les donne pas directement au modèle. On résume chaque cycle par quelques nombres, appelés features. Trois façons de résumer ont été testées, chacune dans son notebook.

### Méthode 1 : statistiques (notebook `exploration.ipynb`)

Pour chaque capteur et chaque cycle, on calcule : minimum, maximum, moyenne, écart type, médiane, quartiles 25 % et 75 %, écart interquartile, moyenne quadratique, asymétrie, aplatissement. On ajoute `qmm`, le rapport entre la pression moyenne et le débit moyen.

Résultat enregistré dans `data/processed/stats/`.

### Méthode 2 : statistiques et fréquences (notebook `exploration_frequence.ipynb`)

Mêmes statistiques que la méthode 1, plus 4 valeurs tirées de la transformée de Fourier de chaque capteur : amplitude maximale, fréquence dominante, énergie, centre de gravité du spectre.

Résultat enregistré dans `data/processed/stats_spectre/`.

### Méthode 3 : histogrammes (notebook `exploration_histogramme.ipynb`)

On découpe la plage de valeurs de chaque capteur en 20 intervalles. Pour chaque cycle, on calcule la part des mesures tombée dans chaque intervalle. Exemple : si 30 % des mesures de pression d'un cycle sont entre 120 et 125 bar, la colonne correspondante vaut 0.30.

Résultat enregistré dans `data/processed/histogramme/`.

Le dossier `data/processed/spectre/` existe aussi, mais aucun notebook actuel ne le produit.

## 5. Les étapes communes aux trois notebooks d'exploration

1. Charger PS2, FS1 et profile avec `load_data`.
2. Construire `X`, le tableau des features (une ligne par cycle).
3. Construire `y` avec `make_binary_target(profile[1], positive_value=100)`.
4. Supprimer les colonnes presque constantes avec `drop_constant_features` (écart type inférieur à 0.001).
5. Supprimer les colonnes redondantes avec `drop_correlated_features` (corrélation absolue supérieure à 0.85, on garde une colonne par paire).
6. Découper en train et test avec `split_data(X, y, test_size=0.2)` : 1764 cycles en train, 441 en test, découpage aléatoire mais stratifié (même proportion de 0 et de 1 des deux côtés), toujours le même grâce à `random_state=42`.
7. Enregistrer les 4 tableaux avec `save_data(...)` dans `data/processed/<nom>/`.

## 6. Modélisation (notebook `model_training.ipynb`)

1. Recharger un jeu préparé : `x_train, x_test, y_train, y_test = load_processed("stats")`.
2. Lancer `tune_all(...)`. Pour chacun des 6 modèles (k plus proches voisins, arbre de décision, régression logistique, Naive Bayes, SVM, forêt aléatoire), cette fonction :
   - teste toutes les combinaisons d'hyperparamètres de la grille définie dans `src/model.py`,
   - note chaque combinaison par validation croisée à 5 plis sur le train uniquement, avec le F1 macro,
   - garde la meilleure combinaison, réentraîne le modèle sur tout le train,
   - l'évalue une seule fois sur le test.
3. Afficher les matrices de confusion et les courbes ROC.
4. Enregistrer les modèles dans `models/` avec `save_models`.

Les modèles sensibles à l'échelle des valeurs (k plus proches voisins, régression logistique, SVM) sont placés dans un pipeline avec un `StandardScaler`. Le scaler apprend la moyenne et l'écart type sur le train seulement, ce qui évite de faire fuiter des informations du test.

Métriques calculées sur le test : accuracy, accuracy équilibrée, précision, rappel, F1 (moyennes macro), AUC.

## 7. Comment réutiliser un jeu préparé dans un nouveau notebook

```python
from src.utils import load_processed

x_train, x_test, y_train, y_test = load_processed("histogramme")
```

Les noms possibles sont `stats`, `stats_spectre`, `histogramme` et `spectre`. `y_train` et `y_test` sont renvoyés sous forme de Series, prêts à être donnés à `fit`.

Pour que `from src... import ...` fonctionne, le notebook doit être lancé avec la racine du projet dans le chemin Python.

## 8. Points à corriger ou à discuter en équipe

1. Découpage imposé par l'énoncé. L'énoncé demande d'utiliser les 2000 premiers cycles pour construire le modèle et les 205 derniers comme test final. Les notebooks font aujourd'hui un découpage aléatoire 80 / 20. Il faudra soit remplacer ce découpage, soit garder les 205 derniers cycles à part pour une évaluation finale.
2. Sélection des features avant le découpage. Les colonnes constantes et corrélées sont retirées en regardant toutes les données, test compris. L'effet est faible car `y` n'est pas utilisé, mais l'idéal est de le faire sur le train seulement.
3. Taille du test. Le README et la valeur par défaut de `split_data` parlent de 15 %, les notebooks utilisent 20 %.
4. Installation. Le README demande `pip install -e .`, mais le dépôt ne contient ni `pyproject.toml` ni `setup.py`, donc cette commande échoue.
