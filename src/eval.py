"""Etape 4 du pipeline : evaluer.

Lit   : models/model.pkl  +  les quatre fichiers de data/split/
Ecrit : reports/metrics.json

On mesure sur l'entrainement ET sur le test. L'ecart entre les deux est le
signal de surapprentissage : un modele qui excelle sur ce qu'il a vu et
flanche sur le reste a memorise au lieu d'apprendre.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

SPLIT = Path("data/split")
MODELE = Path("models/model.pkl")
SORTIE = Path("reports/metrics.json")
CIBLE = "SalePrice"


def mesurer(modele, X, y):
    """RMSE et R2. np.sqrt plutot que root_mean_squared_error : marche avec toutes
    les versions de scikit-learn, y compris celles anterieures a 1.4."""
    prediction = modele.predict(X)
    return np.sqrt(mean_squared_error(y, prediction)), r2_score(y, prediction)


def main():
    with open(MODELE, "rb") as f:
        modele = pickle.load(f)

    X_train = pd.read_csv(SPLIT / "X_train.csv")
    y_train = pd.read_csv(SPLIT / "y_train.csv")[CIBLE]
    X_test = pd.read_csv(SPLIT / "X_test.csv")
    y_test = pd.read_csv(SPLIT / "y_test.csv")[CIBLE]

    rmse_train, r2_train = mesurer(modele, X_train, y_train)
    rmse_test, r2_test = mesurer(modele, X_test, y_test)

    metriques = {
        "rmse_train": round(float(rmse_train), 2),
        "r2_train": round(float(r2_train), 4),
        "rmse_test": round(float(rmse_test), 2),
        "r2_test": round(float(r2_test), 4),
        "ecart_rmse": round(float(rmse_test - rmse_train), 2),
        "n_variables": int(X_train.shape[1]),
    }

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(json.dumps(metriques, indent=2), encoding="utf-8")

    print(f"Entrainement   RMSE = {rmse_train:>10,.0f} $    R2 = {r2_train:.3f}")
    print(f"Test           RMSE = {rmse_test:>10,.0f} $    R2 = {r2_test:.3f}")
    print(f"Ecart RMSE          {rmse_test - rmse_train:>10,.0f} $")
    print(f"Ecrit  : {SORTIE}")


if __name__ == "__main__":
    main()
