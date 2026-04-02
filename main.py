"""
run.py — Point d'entrée utilisateur
Lancez ce fichier et choisissez le nombre de communautés de sujets.

Usage :
    python run.py            # saisie interactive
    python run.py --clusters 8  # passage direct en argument
"""

import argparse
from src.model.embedding import run


def main():
    parser = argparse.ArgumentParser(
        description="Clustering de prises de parole par communautés de sujets."
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="Nombre de clusters K-Means (ex: --clusters 8)",
    )
    args = parser.parse_args()

    if args.clusters is not None:
        n_clusters = args.clusters
    else:
        print("=" * 50)
        print("  Clustering de communautés de sujets")
        print("=" * 50)
        while True:
            try:
                n_clusters = int(input("\nNombre de communautés souhaité : "))
                if n_clusters < 2:
                    print("  ⚠  Veuillez choisir au moins 2 clusters.")
                else:
                    break
            except ValueError:
                print("  ⚠  Veuillez entrer un entier valide.")

    run(n_clusters)


if __name__ == "__main__":
    main()
