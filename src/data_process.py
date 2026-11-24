"""Etape 1 du pipeline : preparer les donnees.

Lit   : data/raw/data.csv  +  params.yaml (process.corr_threshold)
Ecrit : data/processed/dataset.csv

Par rapport au notebook initial, cette etape ajoute une chose : elle ne garde
que les variables suffisamment correlees au prix.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ENTREE = Path("data/raw/data.csv")
SORTIE = Path("data/processed/dataset.csv")
CIBLE = "SalePrice"


def preparer(df, seuil):
    """Ne garde que le numerique, les lignes completes, et les variables liees au prix."""
    num = df.select_dtypes(include=[np.number]).drop(columns=["Id"], errors="ignore")
    num = num.dropna()

    correlations = num.corr()[CIBLE].abs()
    retenues = correlations[correlations >= seuil].index.tolist()
    if CIBLE not in retenues:
        retenues.append(CIBLE)

    return num[retenues]


def main():
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    seuil = params["process"]["corr_threshold"]

    df = pd.read_csv(ENTREE)
    dataset = preparer(df, seuil)

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(SORTIE, index=False)

    print(f"Lu     : {ENTREE}  ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
    print(f"Seuil  : correlation absolue avec {CIBLE} >= {seuil}")
    print(f"Ecrit  : {SORTIE}  ({dataset.shape[0]} lignes, {dataset.shape[1]} colonnes)")
    print(f"         variables retenues : {', '.join(c for c in dataset.columns if c != CIBLE)}")


if __name__ == "__main__":
    main()
