"""
Extraction des prises de parole depuis les fichiers XML de comptes rendus
de l'Assemblée nationale.

Usage :
    # Fichier unique
    python extract_prises_de_parole.py --file chemin/vers/fichier.xml

    # Dossier complet
    python extract_prises_de_parole.py --folder chemin/vers/dossier/

    # Avec export CSV
    python extract_prises_de_parole.py --folder ./xml/ --output resultats.csv

    # Filtrer uniquement les vraies prises de parole (avec orateur identifié)
    python extract_prises_de_parole.py --folder ./xml/ --speakers-only
"""

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import sys

NS = "http://schemas.assemblee-nationale.fr/referentiel"


def t(tag: str) -> str:
    """Retourne le tag avec namespace."""
    return f"{{{NS}}}{tag}"


def get_full_text(elem) -> str:
    """Extrait le texte complet d'un élément, y compris les sous-balises (italique, etc.)."""
    if elem is None:
        return ""
    return " ".join(elem.itertext()).strip()


def parse_date(raw: str) -> str:
    """Convertit '20210201160000000' en '2021-02-01'."""
    try:
        return datetime.strptime(raw[:8], "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:
        return raw


def extract_metadata(root) -> dict:
    """Extrait les métadonnées de la séance."""
    meta = {}
    meta["uid"] = getattr(root.find(t("uid")), "text", "")
    meta["seanceRef"] = getattr(root.find(t("seanceRef")), "text", "")

    # Bloc metadonnees
    md = root.find(t("metadonnees"))
    if md is not None:
        meta["dateSeance"] = parse_date(
            getattr(md.find(t("dateSeance")), "text", "") or ""
        )
        meta["dateSeanceJour"] = getattr(md.find(t("dateSeanceJour")), "text", "") or ""
        meta["numSeance"] = getattr(md.find(t("numSeance")), "text", "") or ""
        meta["typeAssemblee"] = getattr(md.find(t("typeAssemblee")), "text", "") or ""
        meta["legislature"] = getattr(md.find(t("legislature")), "text", "") or ""
        meta["session"] = getattr(md.find(t("session")), "text", "") or ""
    else:
        meta["dateSeance"] = ""
        meta["dateSeanceJour"] = ""
        meta["numSeance"] = ""
        meta["typeAssemblee"] = ""
        meta["legislature"] = ""
        meta["session"] = ""

    return meta


def extract_paragraphes(root, metadata: dict) -> list[dict]:
    """
    Parcourt tous les éléments <paragraphe> du XML et retourne
    une liste de dicts prêts pour un DataFrame.
    """
    rows = []

    for para in root.iter(t("paragraphe")):
        attrib = para.attrib

        # --- Orateur (premier seulement si plusieurs) ---
        nom_orateur = ""
        id_orateur = ""
        qualite_orateur = ""
        for or_elem in para.findall(f".//{t('orateur')}"):
            nom = or_elem.find(t("nom"))
            id_ = or_elem.find(t("id"))
            qual = or_elem.find(t("qualite"))
            nom_orateur = (nom.text or "").strip() if nom is not None else ""
            id_orateur = (id_.text or "").strip() if id_ is not None else ""
            qualite_orateur = (qual.text or "").strip() if qual is not None else ""
            break  # un seul orateur principal par prise de parole

        # --- Texte ---
        texte_elem = para.find(t("texte"))
        texte = get_full_text(texte_elem)
        stime = texte_elem.get("stime", "") if texte_elem is not None else ""

        row = {
            # Métadonnées séance
            "uid_seance": metadata.get("uid", ""),
            "seanceRef": metadata.get("seanceRef", ""),
            "dateSeance": metadata.get("dateSeance", ""),
            "dateSeanceJour": metadata.get("dateSeanceJour", ""),
            "numSeance": metadata.get("numSeance", ""),
            "typeAssemblee": metadata.get("typeAssemblee", ""),
            "legislature": metadata.get("legislature", ""),
            "session": metadata.get("session", ""),
            # Identifiants du paragraphe
            "id_syceron": attrib.get("id_syceron", ""),
            "ordre_absolu_seance": attrib.get("ordre_absolu_seance", ""),
            "ordinal_prise": attrib.get("ordinal_prise", ""),
            "valeur_ptsodj": attrib.get("valeur_ptsodj", ""),
            # Identifiants acteur
            "id_acteur": attrib.get("id_acteur", ""),
            "id_mandat": attrib.get("id_mandat", ""),
            # Classification du paragraphe
            "code_grammaire": attrib.get("code_grammaire", ""),
            "code_style": attrib.get("code_style", ""),
            "code_parole": attrib.get("code_parole", ""),
            "roledebat": attrib.get("roledebat", ""),
            # Orateur
            "nom_orateur": nom_orateur,
            "id_orateur": id_orateur,
            "qualite_orateur": qualite_orateur,
            # Contenu
            "stime": stime,  # horodatage audio (secondes)
            "texte": texte,
            "nb_caracteres": len(texte),
            "nb_mots": len(texte.split()) if texte else 0,
        }

        rows.append(row)

    return rows


def process_file(filepath: str | Path) -> pd.DataFrame:
    """Parse un fichier XML et retourne un DataFrame."""
    filepath = Path(filepath)
    tree = ET.parse(filepath)
    root = tree.getroot()
    metadata = extract_metadata(root)
    rows = extract_paragraphes(root, metadata)
    df = pd.DataFrame(rows)
    # Conversion de types
    for col in ["ordre_absolu_seance", "ordinal_prise", "nb_caracteres", "nb_mots"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["stime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def process_folder(folder: str | Path, pattern: str = "*.xml") -> pd.DataFrame:
    """
    Itère sur tous les fichiers XML d'un dossier et retourne
    un DataFrame consolidé.
    """
    folder = Path(folder)
    xml_files = sorted(folder.glob(pattern))

    if not xml_files:
        print(f"⚠️  Aucun fichier XML trouvé dans : {folder}")
        return pd.DataFrame()

    print(f"📂 {len(xml_files)} fichier(s) XML trouvé(s) dans {folder}")
    dfs = []
    errors = []

    for i, fp in enumerate(xml_files, 1):
        try:
            df = process_file(fp)
            dfs.append(df)
            print(f"  [{i}/{len(xml_files)}] ✅ {fp.name} → {len(df)} paragraphes")
        except Exception as e:
            errors.append((fp.name, str(e)))
            print(f"  [{i}/{len(xml_files)}] ❌ {fp.name} → {e}")

    if errors:
        print(f"\n⚠️  {len(errors)} fichier(s) en erreur :")
        for name, err in errors:
            print(f"   - {name}: {err}")

    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        print(
            f"\n✅ Total : {len(result)} paragraphes extraits depuis {len(dfs)} fichier(s)"
        )
        return result

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Extraction des prises de parole depuis les XML de l'Assemblée nationale"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Chemin vers un fichier XML unique")
    group.add_argument("--folder", help="Chemin vers un dossier de fichiers XML")
    parser.add_argument(
        "--output",
        default="prises_de_parole.csv",
        help="Fichier de sortie CSV (défaut: prises_de_parole.csv)",
    )
    parser.add_argument(
        "--speakers-only",
        action="store_true",
        help="Garder uniquement les paragraphes avec un orateur identifié",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Ne pas exporter en CSV, afficher seulement un aperçu",
    )
    args = parser.parse_args()

    # Extraction
    if args.file:
        print(f"📄 Traitement du fichier : {args.file}")
        df = process_file(args.file)
    else:
        df = process_folder(args.folder)

    if df.empty:
        print("Aucune donnée extraite.")
        sys.exit(1)

    # Filtre optionnel
    if args.speakers_only:
        before = len(df)
        df = df[df["nom_orateur"] != ""].copy()
        print(f"🔍 Filtre orateurs : {before} → {len(df)} paragraphes")

    # Aperçu
    print("\n--- Aperçu du DataFrame ---")
    print(
        df[["dateSeance", "nom_orateur", "roledebat", "code_grammaire", "texte"]]
        .head(10)
        .to_string()
    )
    print(f"\nShape : {df.shape}")
    print(f"\nColonnes : {list(df.columns)}")

    # Export
    if not args.no_export:
        df.to_parquet(args.output, engine="pyarrow", index=False)
        print(f"\n💾 Export CSV : {args.output}")

    return df


if __name__ == "__main__":
    df = main()
