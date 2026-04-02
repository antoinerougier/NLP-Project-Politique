"""
pipeline.py — Vectorisation TF-IDF, extraction d'entités et clustering K-Means
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import nltk
import spacy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARQUET_PATH = "output/prises_de_parole.parquet"
OUTPUT_CSV = "output/embeddings.csv"
OUTPUT_GRAPH = "output/graphique.png"
MIN_MOTS = 6
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def charger_donnees(path: str, min_mots: int) -> pd.DataFrame:
    """Charge le parquet et filtre les textes trop courts."""
    df = pd.read_parquet(path)
    df = df[df["nb_mots"] > min_mots].reset_index(drop=True)
    print(f"  {len(df)} prises de parole chargées (nb_mots > {min_mots})")
    return df


def vectoriser(textes: pd.Series, stop_words: list) -> tuple:
    """Retourne (matrice_sparse, vectoriseur)."""
    vec = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words=stop_words,
        ngram_range=NGRAM_RANGE,
    )
    matrice = vec.fit_transform(textes)
    print(f"  Matrice TF-IDF : {matrice.shape[0]} docs × {matrice.shape[1]} features")
    return matrice, vec


def extraire_entites(texte: str, nlp) -> list:
    """Retourne une liste de tuples (texte_entité, label)."""
    doc = nlp(texte)
    return [(ent.text, ent.label_) for ent in doc.ents]


def clustering_kmeans(matrice, n_clusters: int) -> np.ndarray:
    """Applique K-Means et retourne les labels."""
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(matrice)
    print(f"  K-Means terminé : {n_clusters} clusters")
    return labels


def reduire_2d(matrice) -> np.ndarray:
    """Réduit la matrice TF-IDF en 2D via SVD tronquée (LSA)."""
    svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    return svd.fit_transform(matrice)


def sauvegarder_graphique(coords_2d: np.ndarray, labels: np.ndarray, path: str):
    """Génère et sauvegarde le scatter plot des clusters."""
    n_clusters = len(np.unique(labels))
    cmap = plt.get_cmap("tab20", n_clusters)
    colors = [cmap(i) for i in range(n_clusters)]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            s=18,
            alpha=0.65,
            color=colors[cluster_id],
            linewidths=0,
        )

    # Centroïdes
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cx, cy = coords_2d[mask, 0].mean(), coords_2d[mask, 1].mean()
        ax.scatter(
            cx,
            cy,
            s=120,
            color=colors[cluster_id],
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
        )
        ax.text(
            cx,
            cy,
            str(cluster_id),
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="white",
            zorder=6,
        )

    # Légende
    patches = [
        mpatches.Patch(color=colors[i], label=f"Cluster {i}") for i in range(n_clusters)
    ]
    ax.legend(
        handles=patches,
        loc="upper right",
        framealpha=0.25,
        facecolor="#1e2130",
        edgecolor="#444",
        labelcolor="white",
        fontsize=8,
        ncol=max(1, n_clusters // 10),
    )

    ax.set_title(
        f"Communautés de sujets — K-Means ({n_clusters} clusters)",
        color="white",
        fontsize=14,
        pad=14,
    )
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.set_xlabel("Composante LSA 1", color="#888", fontsize=9)
    ax.set_ylabel("Composante LSA 2", color="#888", fontsize=9)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Graphique sauvegardé → {path}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def run(n_clusters: int):
    """Lance le pipeline complet avec le nombre de clusters choisi."""

    print("\n[1/5] Chargement des données…")
    df = charger_donnees(PARQUET_PATH, MIN_MOTS)

    print("\n[2/5] Vectorisation TF-IDF…")
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

    stop_fr = stopwords.words("french")
    matrice, _ = vectoriser(df["texte"], stop_fr)

    print("\n[3/5] Extraction des entités (spaCy)…")
    nlp = spacy.load("fr_core_news_md")
    df["entites"] = df["texte"].apply(lambda t: extraire_entites(t, nlp))

    print("\n[4/5] Clustering K-Means…")
    labels = clustering_kmeans(matrice, n_clusters)
    df["cluster"] = labels

    # Stocker le vecteur TF-IDF (représentation dense — attention mémoire !)
    print("  Conversion matrice → vecteurs denses…")
    df["vecteur"] = list(matrice.toarray())

    # Réorganiser les colonnes
    df = df[["texte", "vecteur", "entites", "cluster"]]

    print("\n[5/5] Sauvegarde…")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  DataFrame sauvegardé → {OUTPUT_CSV}")

    coords_2d = reduire_2d(matrice)
    sauvegarder_graphique(coords_2d, labels, OUTPUT_GRAPH)

    print("\n✓ Pipeline terminé.\n")
    return df
