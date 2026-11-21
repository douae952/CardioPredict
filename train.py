
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Charger les données nettoyées
df = pd.read_csv("cardio_clean.csv")

# Séparer les variables
X = df.drop("cardio", axis=1)
y = df["cardio"]

# Division des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Modèle XGBoost
model_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

# Entraînement
model_xgb.fit(X_train, y_train)

# Évaluation
print("Accuracy :", model_xgb.score(X_test, y_test))

# Sauvegarde
joblib.dump(
    {
        "model": model_xgb,
        "features": X_train.columns.tolist()
    },
    "xgb_cardio.pkl"
)

print("Modèle sauvegardé avec succès !")
