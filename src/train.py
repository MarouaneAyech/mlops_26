"""Etape 3 du pipeline : entrainer.

Lit   : data/split/X_train.csv, y_train.csv
Ecrit : models/model.pkl

Ce script n'affiche aucune metrique : mesurer n'est pas son travail.
C'est eval.py qui s'en charge, sur un jeu que ce script n'a jamais vu.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

ENTREE = Path("data/split")
SORTIE = Path("models/model.pkl")
CIBLE = "SalePrice"


def entrainer(X_train, y_train):
    modele = LinearRegression()
    modele.fit(X_train, y_train)
    return modele


def main():
    X_train = pd.read_csv(ENTREE / "X_train.csv")
    y_train = pd.read_csv(ENTREE / "y_train.csv")[CIBLE]

    modele = entrainer(X_train, y_train)

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    with open(SORTIE, "wb") as f:
        pickle.dump(modele, f)

    print(f"Entraine sur {X_train.shape[0]} biens et {X_train.shape[1]} variables")
    print(f"Ecrit    : {SORTIE}")


if __name__ == "__main__":
    main()
