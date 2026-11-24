"""Etape 2 du pipeline : decouper.

Lit   : data/processed/dataset.csv  +  params.yaml (split.*)
Ecrit : data/split/X_train.csv, y_train.csv, X_test.csv, y_test.csv

Deux operations en une : separer les variables de la cible, puis decouper
en entrainement et test. C'est ici, et seulement ici, que vit l'alea du projet.
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ENTREE = Path("data/processed/dataset.csv")
SORTIE = Path("data/split")
CIBLE = "SalePrice"


def decouper(dataset, test_size, n_bins, seed):
    """Separe X et y, puis decoupe en stratifiant par tranche de prix."""
    X = dataset.drop(columns=[CIBLE])
    y = dataset[CIBLE]

    # Les prix sont tres inegalement repartis : on stratifie pour que les biens
    # chers et les biens modestes soient representes des deux cotes.
    tranches = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")

    return train_test_split(X, y, test_size=test_size, stratify=tranches, random_state=seed)


def main():
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    p = params["split"]

    dataset = pd.read_csv(ENTREE)
    X_train, X_test, y_train, y_test = decouper(
        dataset, p["test_size"], p["n_bins"], p["seed"]
    )

    SORTIE.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(SORTIE / "X_train.csv", index=False)
    X_test.to_csv(SORTIE / "X_test.csv", index=False)
    y_train.to_frame().to_csv(SORTIE / "y_train.csv", index=False)
    y_test.to_frame().to_csv(SORTIE / "y_test.csv", index=False)

    print(f"Graine : {p['seed']}   (sans elle, ce decoupage changerait a chaque execution)")
    print(f"Ecrit  : {SORTIE}/  —  entrainement {X_train.shape[0]} biens, test {X_test.shape[0]} biens")


if __name__ == "__main__":
    main()
