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
notebook_projet.ipynb  notebook final : tout le projet, du problème au modèle retenu
data/raw/          fichiers bruts
data/processed/    jeux train et test déjà préparés, un sous dossier par méthode
                   (<nom>/ : découpage aléatoire, <nom>_enonce/ : découpage de l'énoncé)
notebooks/         les notebooks de travail (exploration, modèles, ensembles, découpage)
models/            modèles entraînés, un sous dossier par jeu de features
src/utils.py       chargement, création des features, sélection, découpage, graphiques
src/model.py       liste des modèles et grilles d'hyperparamètres
src/train.py       entraînement, validation croisée, recherche d'hyperparamètres
src/evaluation.py  métriques, matrices de confusion, courbes ROC
tests/             tests unitaires (pytest)
figures/           images enregistrées pour le rapport
docs/              énoncé et cette documentation
```

## 4. Le problème de départ : trop de colonnes

Un cycle contient 6600 valeurs brutes. On ne les donne pas directement au modèle. On résume chaque cycle par quelques nombres, appelés features. Quatre façons de résumer ont été testées.

### Méthode 1 : statistiques (notebook `exploration.ipynb`)

Pour chaque capteur et chaque cycle, on calcule : minimum, maximum, moyenne, écart type, médiane, quartiles 25 % et 75 %, écart interquartile, moyenne quadratique, asymétrie, aplatissement. On ajoute `qmm`, le rapport entre la pression moyenne et le débit moyen.

Résultat enregistré dans `data/processed/stats/`.

### Méthode 2 : statistiques et fréquences (notebook `exploration_frequence.ipynb`)

Mêmes statistiques que la méthode 1, plus 4 valeurs tirées de la transformée de Fourier de chaque capteur : amplitude maximale, fréquence dominante, énergie, centre de gravité du spectre.

Résultat enregistré dans `data/processed/stats_spectre/`.

### Méthode 3 : histogrammes (notebook `exploration_histogramme.ipynb`)

On découpe la plage de valeurs de chaque capteur en 20 intervalles. Pour chaque cycle, on calcule la part des mesures tombée dans chaque intervalle. Exemple : si 30 % des mesures de pression d'un cycle sont entre 120 et 125 bar, la colonne correspondante vaut 0.30.

Résultat enregistré dans `data/processed/histogramme/`.

### Méthode 4 : fréquences seules

Seulement les 4 valeurs de la transformée de Fourier, pour chaque capteur. Aucun notebook d'exploration ne la produit aujourd'hui, mais `build_feature_set("spectre", ...)` la reconstruit.

Résultat enregistré dans `data/processed/spectre/`.

La fonction `build_feature_set(nom, PS2, FS1)` de `src/utils.py` construit les features brutes de n'importe laquelle des 4 méthodes, exactement comme les notebooks d'exploration. La liste des noms est dans `FEATURE_SETS`.

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
4. Enregistrer les modèles avec `save_models(modeles, "stats")`, dans `models/stats/`. On les recharge avec `load_model("svm", "stats")`.

Les modèles sensibles à l'échelle des valeurs (k plus proches voisins, régression logistique, SVM) sont placés dans un pipeline avec un `StandardScaler`. Le scaler apprend la moyenne et l'écart type sur le train seulement, ce qui évite de faire fuiter des informations du test.

Métriques calculées sur le test : accuracy, accuracy équilibrée, précision, rappel, F1 (moyennes macro), AUC.

## 7. Ensemble learning (notebook `ensemble_learning.ipynb`)

Un modèle d'ensemble combine plusieurs modèles pour prendre une seule décision. `get_ensemble_models()` dans `src/model.py` en fournit 7, en trois familles :

| Famille | Modèles | Principe |
|---|---|---|
| Bagging | `bagging`, `random_forest`, `extra_trees` | beaucoup d'arbres entraînés chacun sur un tirage du train, puis vote |
| Boosting | `adaboost`, `gradient_boosting` | des modèles simples ajoutés un par un, chacun corrige les erreurs des précédents |
| Combinaison | `vote_soft`, `stacking` | régression logistique, k plus proches voisins, SVM et forêt aléatoire décident ensemble |

`compare_models(modeles, x_train, y_train, x_test, y_test)` dans `src/train.py` compare des modèles sans recherche d'hyperparamètres. Pour chacun, il donne le score sur le train, le score en validation croisée et les scores sur le test. Un grand écart entre le score sur le train et celui de la validation croisée signale un surapprentissage.

`tune_model` accepte aussi `gradient_boosting` et `adaboost`, qui ont leur grille dans `get_param_grids()`.

## 8. Découpage de l'énoncé (notebook `decoupage_enonce.ipynb`)

L'énoncé impose : les 2000 premiers cycles pour construire le modèle, les 205 derniers pour le test final. Ce notebook reconstruit les 4 jeux de features de cette façon et les enregistre dans `data/processed/<nom>_enonce/`.

Trois fonctions de `src/utils.py` servent à ça :

1. `build_feature_set(nom, PS2, FS1, n_fit=2000)` : les bornes des histogrammes sont fixées sur les 2000 premiers cycles seulement.
2. `split_first_n(X, y, n_train=2000)` : découpage sans mélange, dans l'ordre des cycles.
3. `select_features(x_train, x_test)` : colonnes constantes et corrélées choisies sur le train seulement, puis mêmes colonnes gardées dans le test.

Point important : les 205 derniers cycles ne ressemblent pas aux 2000 premiers. Dans le test, la vanne est optimale dans 36 % des cycles (contre 53 % dans le train), le refroidisseur est toujours à 100 % et l'accumulateur ne prend que 2 valeurs sur 4. Le test est donc plus difficile qu'un découpage aléatoire, et les scores y sont un peu plus bas.

## 9. Le notebook final (`notebook_projet.ipynb`, à la racine)

Il reprend tout le projet dans l'ordre, avec le découpage de l'énoncé : le problème, les données, le découpage, la construction et la sélection des features, les valeurs aberrantes, le choix des modèles, l'optimisation des hyperparamètres, l'ensemble learning, l'évaluation finale sur les 205 cycles et l'interprétation du modèle retenu. Il se suffit à lui même : il repart des fichiers bruts.

## 10. Comment réutiliser un jeu préparé dans un nouveau notebook

```python
from src.utils import load_processed

x_train, x_test, y_train, y_test = load_processed("histogramme")
```

Les noms possibles sont `stats`, `stats_spectre`, `histogramme` et `spectre` (découpage aléatoire), ou les mêmes suivis de `_enonce` (découpage de l'énoncé). `y_train` et `y_test` sont renvoyés sous forme de Series, prêts à être donnés à `fit`.

Pour que `from src... import ...` fonctionne, le projet doit être installé une fois dans l'environnement du notebook :

```bash
.venv/Scripts/python.exe -m pip install -e .
```

## 11. Points à discuter en équipe

1. Découpage. Les notebooks d'exploration et `model_training.ipynb` utilisent encore le découpage aléatoire 80 / 20. Le découpage de l'énoncé est appliqué dans `decoupage_enonce.ipynb` et `notebook_projet.ipynb`. Les modèles enregistrés dans `models/` viennent du découpage aléatoire.
2. Taille du test aléatoire. Le README et la valeur par défaut de `split_data` parlent de 15 %, les notebooks utilisent 20 %.
