# 🏠 House Price Prediction - Pipeline ML Bout-en-Bout

Ce projet implémente une solution complète de machine learning pour la prédiction des prix immobiliers. Il intègre un pipeline de prétraitement automatisé, une optimisation des hyperparamètres et un déploiement via une application web interactive.

## 🌟 Points Forts du Projet
* **Pipeline Scikit-Learn :** Automatisation du nettoyage (Imputation), de la mise à l'échelle (StandardScaler) et de l'encodage (OneHotEncoder).
* **Optimisation Robuste :** Recherche par grille (**GridSearchCV**) pour identifier les meilleurs paramètres du modèle Random Forest.
* **Gestion des Données :** Traitement différencié des variables numériques et catégorielles.
* **Déploiement :** Interface utilisateur fluide avec **Streamlit**.

## 🛠️ Stack Technique
* **Langage :** Python 3.x
* **Librairies ML :** Scikit-Learn, Pandas, NumPy, Joblib
* **Interface :** Streamlit
* **Qualité :** Logging intégré pour le suivi de l'entraînement.

## 📂 Structure des Fichiers
* `train.py` : Script d'entraînement complet (Prétraitement, GridSearch, Évaluation, Sauvegarde).
* `app.py` : Application web Streamlit pour la prédiction en temps réel.
* `data.csv` : Jeu de données source (contenant la cible `MEDV`).
* `model.pkl` : Modèle sérialisé après entraînement.

## ⚙️ Utilisation

### 1. Installation
```bash
git clone [https://github.com/solaymanaitali/Projet-House-Price-Prediction.git](https://github.com/solaymanaitali/Projet-House-Price-Prediction.git)
cd Projet-House-Price-Prediction
pip install -r requirements.txt
