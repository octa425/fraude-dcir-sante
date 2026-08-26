# ============================================================
# ref_tables.py — REF_ACTES et REF_SPECIALITES
# Projet Detection anomalies DCIR
# Auteur : Octavien YAMESSE
# ============================================================

import pandas as pd
import os
from config import DATA_RAW_DIR

def generer_ref_specialites():
    """
    Table de reference des specialites medicales.
    Contient les seuils adaptatifs par specialite
    utilises pour la detection d'anomalies.
    """
    data = [
        # PRASPE | LIBELLE              | PRACAT | actes_j_moy | P99 | moy_dep | tarif_ref
        ("01", "Medecine generale",        1,  25, 45, 0.00,  26.50),
        ("03", "Cardiologie",              1,  15, 28, 0.35,  65.00),
        ("05", "Dermatologie",             1,  14, 25, 0.45,  55.00),
        ("07", "Gynecologie",              1,  12, 22, 0.30,  50.00),
        ("15", "Ophtalmologie",            1,  16, 30, 0.40,  60.00),
        ("19", "Chirurgie dentaire",       4,  12, 22, 0.25,  85.00),
        ("24", "Soins infirmiers",         6,  30, 55, 0.00,   9.45),
        ("26", "Kinesitherapie",           7,  20, 38, 0.00,  45.60),
        ("33", "Psychiatrie",              1,   8, 14, 0.20,  55.00),
        ("50", "Pharmacie",               50,   0,  0, 0.00,   0.00),
    ]

    df = pd.DataFrame(data, columns=[
        "PRASPE_COD", "PRASPE_LIB", "PRACAT_COD",
        "ACTES_JOUR_NORMAL", "ACTES_JOUR_SEUIL_P99",
        "TAUX_DEPASSEMENT_MOYEN", "TARIF_REF_MOYEN"
    ])

    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    df.to_csv(f"{DATA_RAW_DIR}REF_SPECIALITES.csv", index=False)
    print(f"REF_SPECIALITES : {len(df)} specialites")
    return df


def generer_ref_actes():
    """
    Table de reference des actes medicaux.
    SPECIALITES_COMPAT = liste des codes specialite
    autorises a realiser cet acte (separateur ;).
    Indispensable pour le scenario S3.
    """
    data = [
        # ACT_COD | LIBELLE                    | CATEGORIE         | TARIF | SPECIALITES_COMPAT
        ("C001", "Consultation MG",              "CONSULTATION",     26.50, "01;22;23"),
        ("C002", "Consultation cardiologue",     "CONSULTATION",     30.00, "03"),
        ("C003", "Consultation dermatologue",    "CONSULTATION",     30.00, "05"),
        ("C004", "Consultation ophtalmo",        "CONSULTATION",     30.00, "15"),
        ("C005", "Consultation gyneco",          "CONSULTATION",     30.00, "07;70;79"),
        ("C006", "Consultation psychiatre",      "CONSULTATION",     45.73, "33;75"),
        ("C007", "Consultation pediatre",        "CONSULTATION",     30.00, "12"),
        ("D001", "Consultation dentiste",        "DENTAIRE",         26.50, "04;19"),
        ("D002", "Extraction dentaire",          "DENTAIRE",         50.00, "04;19"),
        ("D003", "Couronne dentaire",            "DENTAIRE",        120.00, "04;19"),
        ("K001", "Seance kinesitherapie AMK",    "KINESITHERAPIE",   45.60, "07;26"),
        ("K002", "Seance kine respiratoire",     "KINESITHERAPIE",   38.40, "26"),
        ("I001", "Soins infirmiers AIS1",        "SOINS_INFIRMIERS",  3.15, "06;24"),
        ("I002", "Soins infirmiers AIS3",        "SOINS_INFIRMIERS",  9.45, "06;24"),
        ("I003", "Prise de sang IDE",            "SOINS_INFIRMIERS",  4.20, "06;24"),
        ("A001", "Acte gynecologique",           "ACTE_TECHNIQUE",   30.00, "07;70;79"),
        ("A002", "Echographie cardiaque",        "ACTE_TECHNIQUE",   78.96, "03"),
        ("A003", "Fond d'oeil",                  "ACTE_TECHNIQUE",   22.96, "15"),
        ("V001", "Visite MG a domicile",         "VISITE",           35.75, "01;22;23"),
    ]

    df = pd.DataFrame(data, columns=[
        "ACT_COD", "ACT_LIBELLE", "ACT_CATEGORIE",
        "TARIF_REFERENCE", "SPECIALITES_COMPAT"
    ])

    df.to_csv(f"{DATA_RAW_DIR}REF_ACTES.csv", index=False)
    print(f"REF_ACTES : {len(df)} actes")
    return df


if __name__ == "__main__":
    ref_spe = generer_ref_specialites()
    ref_act = generer_ref_actes()
    print("Referentiels generes avec succes !")
