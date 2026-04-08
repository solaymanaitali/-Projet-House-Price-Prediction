import pandas as pd
import joblib
import logging
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------
# LOGGING
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")

# -----------------------
# LOAD DATA
# -----------------------
logging.info("Chargement des données...")
df = pd.read_csv(DATA_PATH)

if "MEDV" not in df.columns:
    raise Exception("La colonne MEDV est manquante")

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# -----------------------
# TRAIN / TEST SPLIT
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# PREPROCESSING
# -----------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

from sklearn.impute import SimpleImputer

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # remplace les valeurs manquantes par la médiane
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # remplace les valeurs manquantes par la valeur la plus fréquente
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

# -----------------------
# TRAIN
# -----------------------
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_absolute_error")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Évaluation
pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

logging.info(f"MAE: {mae:.2f}")
logging.info(f"R2: {r2:.3f}")

# Sauvegarde du modèle optimisé
joblib.dump(best_model, "model.pkl")
logging.info("Modèle sauvegardé avec succès")