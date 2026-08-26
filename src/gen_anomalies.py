# ============================================================
# gen_anomalies.py — Injection des anomalies S1 a S5
# Projet Detection anomalies DCIR
# Auteur : Octavien YAMESSE
# ============================================================

import pandas as pd
import numpy as np
import os
from datetime import date, timedelta
from config import DATA_RAW_DIR, ANOMALIES, RANDOM_SEED

np.random.seed(RANDOM_SEED)

def selectionner_praticiens_anomaux(df_praticiens):
    """
    Selectionne les praticiens qui recevront
    des anomalies injectees.
    Retourne un dict {scenario: [PRANUM_PRA]}
    """
    suspects = {}

    # S1 : Suractivite → MG, infirmiers, kines, specialistes
    mg = df_praticiens[df_praticiens["PRASPE_PRA"] == "01"]["PRANUM_PRA"].tolist()
    inf = df_praticiens[df_praticiens["PRASPE_PRA"] == "24"]["PRANUM_PRA"].tolist()
    kine = df_praticiens[df_praticiens["PRASPE_PRA"] == "26"]["PRANUM_PRA"].tolist()
    spe = df_praticiens[df_praticiens["PRASPE_PRA"].isin(["03","05","15"])]["PRANUM_PRA"].tolist()

    suspects["S1"] = (
        np.random.choice(mg, 6, replace=False).tolist() +
        np.random.choice(inf, 4, replace=False).tolist() +
        np.random.choice(kine, 3, replace=False).tolist() +
        np.random.choice(spe, 2, replace=False).tolist()
    )

    # S2 : Actes fantomes → MG, specialistes, infirmiers, dentistes
    dent = df_praticiens[df_praticiens["PRASPE_PRA"] == "19"]["PRANUM_PRA"].tolist()
    suspects["S2"] = (
        np.random.choice(mg, 4, replace=False).tolist() +
        np.random.choice(spe, 3, replace=False).tolist() +
        np.random.choice(inf, 2, replace=False).tolist() +
        np.random.choice(dent, 1, replace=False).tolist()
    )

    # S3 : Incoherence acte/specialite → specialistes, dentistes, infirmiers, kines
    suspects["S3"] = (
        np.random.choice(spe, 4, replace=False).tolist() +
        np.random.choice(dent, 3, replace=False).tolist() +
        np.random.choice(inf, 3, replace=False).tolist() +
        np.random.choice(kine, 2, replace=False).tolist()
    )

    # S4 : Anomalie tarifaire → specialistes S2, dentistes, MG, kines
    spe_s2 = df_praticiens[
        (df_praticiens["PRASPE_PRA"].isin(["03","05","15"])) &
        (df_praticiens["CNVMTF_PRA"] == "3")
    ]["PRANUM_PRA"].tolist()
    if len(spe_s2) < 5:
        spe_s2 = spe

    suspects["S4"] = (
        np.random.choice(spe_s2, min(5, len(spe_s2)), replace=False).tolist() +
        np.random.choice(dent, 4, replace=False).tolist() +
        np.random.choice(mg, 3, replace=False).tolist() +
        np.random.choice(kine, 3, replace=False).tolist()
    )

    # S5 : Anomalie relationnelle → MG prescripteurs
    suspects["S5"] = np.random.choice(mg, 13, replace=False).tolist()

    print("Praticiens selectionnes pour anomalies :")
    for s, lst in suspects.items():
        print(f"  {s} : {len(lst)} praticiens")

    return suspects


def injecter_s1_suractivite(df_prestations, suspects_s1):
    """
    S1 : Multiplier le volume d'actes par jour
    par un facteur entre 1.8 et 2.5
    pendant une periode de 1 a 3 mois.
    """
    print("Injection S1 (suractivite)...")
    nouvelles_lignes = []

    for pranum in suspects_s1:
        # Periode d'anomalie
        debut_mois = np.random.randint(1, 10)
        nb_mois = np.random.randint(1, 4)
        date_debut_ano = date(2023, debut_mois, 1)
        date_fin_mois = debut_mois + nb_mois - 1
        if date_fin_mois > 12:
            date_fin_mois = 12
        date_fin_ano = date(2023, date_fin_mois, 28)

        # Facteur de suractivite
        facteur = np.random.uniform(1.8, 2.5)

        # Prestations existantes de ce praticien
        mask = (
            (df_prestations["PRANUM_PRA"] == pranum) &
            (pd.to_datetime(df_prestations["EXE_SOI_DTD"]).dt.date >= date_debut_ano) &
            (pd.to_datetime(df_prestations["EXE_SOI_DTD"]).dt.date <= date_fin_ano)
        )
        prs_pra = df_prestations[mask].copy()

        if len(prs_pra) == 0:
            continue

        # Dupliquer les actes (facteur - 1) fois
        nb_actes_supplementaires = int(len(prs_pra) * (facteur - 1))
        if nb_actes_supplementaires > 0:
            echantillon = prs_pra.sample(
                n=min(nb_actes_supplementaires, len(prs_pra)),
                replace=True
            ).copy()
            echantillon["PRS_ID"] = [
                f"PRS_S1_{pranum}_{i}"
                for i in range(len(echantillon))
            ]
            echantillon["origine_simulation"] = "S1_SURACTIVITE"
            nouvelles_lignes.append(echantillon)

    if nouvelles_lignes:
        df_s1 = pd.concat(nouvelles_lignes, ignore_index=True)
        df_prestations = pd.concat([df_prestations, df_s1], ignore_index=True)
        print(f"  S1 : {len(df_s1):,} actes supplementaires injectes")

    return df_prestations


def injecter_s2_actes_fantomes(df_prestations, df_beneficiaires, df_praticiens, suspects_s2):
    """
    S2 : Trois types d'anomalies :
    a) Actes post-mortem (acte apres deces du patient)
    b) Beneficiaire fictif (BEN_IDT_ANO absent de IR_BEN_R)
    c) Praticien radie (statut_actif=0)
    """
    print("Injection S2 (actes fantomes)...")
    nouvelles_lignes = []

    # S2a : Actes post-mortem
    decedes = df_beneficiaires[
        df_beneficiaires["BEN_DCD_DTE"].notna()
    ].head(5)

    for _, ben in decedes.iterrows():
        date_deces = pd.to_datetime(ben["BEN_DCD_DTE"]).date()
        if date_deces >= date(2023, 12, 31):
            continue

        pranum = np.random.choice(suspects_s2[:3])
        nb_actes_post = np.random.randint(5, 20)

        for i in range(nb_actes_post):
            jours_apres = np.random.randint(1, 60)
            date_acte = date_deces + timedelta(days=jours_apres)
            if date_acte > date(2023, 12, 31):
                date_acte = date(2023, 12, 31)

            nouvelles_lignes.append({
                "PRS_ID": f"PRS_S2a_{pranum}_{i}",
                "PRANUM_PRA": pranum,
                "PFS_PRE_NUM": None,
                "BEN_IDT_ANO": ben["BEN_IDT_ANO"],
                "EXE_SOI_DTD": date_acte,
                "PRS_NAT_COD": "C",
                "ACT_COD": "C001",
                "BSE_REM_PRU": 26.50,
                "HON_MNT": 26.50,
                "DEP_MNT": 0.0,
                "BSE_REM_MNT": 18.55,
                "RGO_REM_TAU": 0.70,
                "EXE_LIE_COD": 1,
                "PRS_ACT_QTE": 1,
                "PRS_ORD_NUM": None,
                "FLX_DIS_DTD": date_acte + timedelta(days=5),
                "origine_simulation": "S2_POST_MORTEM",
            })

    # S2b : Beneficiaires fictifs (serie BEN999xxxx)
    for idx, pranum in enumerate(suspects_s2[3:7]):
        for i in range(np.random.randint(10, 25)):
            ben_fictif = f"BEN999{str(idx * 100 + i).zfill(4)}"
            jours = np.random.randint(0, 365)
            date_acte = date(2023, 1, 1) + timedelta(days=jours)

            nouvelles_lignes.append({
                "PRS_ID": f"PRS_S2b_{pranum}_{i}",
                "PRANUM_PRA": pranum,
                "PFS_PRE_NUM": None,
                "BEN_IDT_ANO": ben_fictif,
                "EXE_SOI_DTD": date_acte,
                "PRS_NAT_COD": "C",
                "ACT_COD": "C001",
                "BSE_REM_PRU": 26.50,
                "HON_MNT": 26.50,
                "DEP_MNT": 0.0,
                "BSE_REM_MNT": 18.55,
                "RGO_REM_TAU": 0.70,
                "EXE_LIE_COD": 1,
                "PRS_ACT_QTE": 1,
                "PRS_ORD_NUM": None,
                "FLX_DIS_DTD": date_acte + timedelta(days=5),
                "origine_simulation": "S2_BEN_FICTIF",
            })

    if nouvelles_lignes:
        df_s2 = pd.DataFrame(nouvelles_lignes)
        df_prestations = pd.concat([df_prestations, df_s2], ignore_index=True)
        print(f"  S2 : {len(df_s2):,} actes fantomes injectes")

    return df_prestations


def injecter_s3_incoherence(df_prestations, df_praticiens, suspects_s3):
    """
    S3 : Injecter des actes incompatibles avec la specialite.
    Un dermatologue qui facture des actes gynecologiques, etc.
    """
    print("Injection S3 (incoherence acte/specialite)...")

    # Actes incompatibles par specialite
    actes_incompatibles = {
        "03": ["A001", "I001", "K001"],  # Cardio → actes gyneco, soins inf, kine
        "05": ["A001", "K001", "D001"],  # Dermato → gyneco, kine, dentaire
        "15": ["A001", "K001", "I001"],  # Ophtalmo → gyneco, kine, soins inf
        "19": ["K001", "C001", "I001"],  # Dentiste → kine, MG, soins inf
        "24": ["C001", "D001", "K001"],  # Infirmier → MG, dentaire, kine
        "26": ["C001", "D001", "I001"],  # Kine → MG, dentaire, soins inf
        "07": ["D001", "I001", "K001"],  # Gyneco → dentaire, soins inf, kine
    }

    tarifs_incompatibles = {
        "A001": 30.00, "I001": 3.15, "K001": 45.60,
        "D001": 26.50, "C001": 26.50,
    }

    nouvelles_lignes = []

    for pranum in suspects_s3:
        pra = df_praticiens[df_praticiens["PRANUM_PRA"] == pranum]
        if len(pra) == 0:
            continue
        praspe = str(pra.iloc[0]["PRASPE_PRA"])

        if praspe not in actes_incompatibles:
            continue

        actes_incompat = actes_incompatibles[praspe]

        # Prestations existantes de ce praticien
        prs_pra = df_prestations[
            df_prestations["PRANUM_PRA"] == pranum
        ]

        # Injecter 10% a 20% d'actes incompatibles
        nb_incompat = int(len(prs_pra) * np.random.uniform(0.10, 0.20))

        for i in range(nb_incompat):
            act_incompat = np.random.choice(actes_incompat)
            tarif = tarifs_incompatibles.get(act_incompat, 26.50)

            jours = np.random.randint(0, 365)
            date_acte = date(2023, 1, 1) + timedelta(days=jours)

            nouvelles_lignes.append({
                "PRS_ID": f"PRS_S3_{pranum}_{i}",
                "PRANUM_PRA": pranum,
                "PFS_PRE_NUM": None,
                "BEN_IDT_ANO": f"BEN{str(np.random.randint(1, 10000)).zfill(7)}",
                "EXE_SOI_DTD": date_acte,
                "PRS_NAT_COD": act_incompat[0],
                "ACT_COD": act_incompat,
                "BSE_REM_PRU": tarif,
                "HON_MNT": tarif,
                "DEP_MNT": 0.0,
                "BSE_REM_MNT": round(tarif * 0.70, 2),
                "RGO_REM_TAU": 0.70,
                "EXE_LIE_COD": 1,
                "PRS_ACT_QTE": 1,
                "PRS_ORD_NUM": None,
                "FLX_DIS_DTD": date_acte + timedelta(days=5),
                "origine_simulation": "S3_INCOHERENCE",
            })

    if nouvelles_lignes:
        df_s3 = pd.DataFrame(nouvelles_lignes)
        df_prestations = pd.concat([df_prestations, df_s3], ignore_index=True)
        print(f"  S3 : {len(df_s3):,} actes incompatibles injectes")

    return df_prestations


def injecter_s4_anomalie_tarifaire(df_prestations, df_praticiens, suspects_s4):
    """
    S4 : Multiplier les honoraires par un facteur 2.0 a 4.0
    par rapport au tarif de reference.
    """
    print("Injection S4 (anomalie tarifaire)...")

    for pranum in suspects_s4:
        facteur = np.random.uniform(2.0, 4.0)

        mask = df_prestations["PRANUM_PRA"] == pranum
        df_prestations.loc[mask, "HON_MNT"] = (
            df_prestations.loc[mask, "BSE_REM_PRU"] * facteur
        ).round(2)
        df_prestations.loc[mask, "DEP_MNT"] = (
            df_prestations.loc[mask, "HON_MNT"] -
            df_prestations.loc[mask, "BSE_REM_PRU"]
        ).round(2)
        df_prestations.loc[mask, "origine_simulation"] = "S4_TARIFAIRE"

    print(f"  S4 : {len(suspects_s4)} praticiens avec honoraires anormaux")
    return df_prestations


def injecter_s5_reseau(df_pharmacie, df_praticiens, suspects_s5):
    """
    S5 : Concentrer 85-95% des prescriptions
    d'un MG vers une seule pharmacie.
    """
    print("Injection S5 (anomalie relationnelle)...")

    pharmacies = df_praticiens[
        df_praticiens["PRACAT_PRA"] == 50
    ]["PRANUM_PRA"].tolist()

    if not pharmacies:
        print("  S5 : aucune pharmacie disponible")
        return df_pharmacie

    for pranum in suspects_s5:
        # Pharmacie complice
        pharma_complice = np.random.choice(pharmacies)

        # Concentration anormale : 85-95%
        concentration = np.random.uniform(0.85, 0.95)

        # Prescriptions de ce praticien
        mask = df_pharmacie["PFS_PRE_NUM"] == pranum
        nb_total = mask.sum()

        if nb_total == 0:
            continue

        # Selectionner les prescriptions a rediriger
        indices = df_pharmacie[mask].index
        nb_a_rediriger = int(nb_total * concentration)
        indices_rediriger = np.random.choice(
            indices, size=nb_a_rediriger, replace=False
        )

        df_pharmacie.loc[indices_rediriger, "PRANUM_EXE"] = pharma_complice
        df_pharmacie.loc[indices_rediriger, "origine_simulation"] = "S5_RESEAU"

    print(f"  S5 : {len(suspects_s5)} praticiens avec concentration pharmacie")
    return df_pharmacie


if __name__ == "__main__":
    print("Test injection anomalies...")
    df_pra = pd.read_csv(f"{DATA_RAW_DIR}IR_PSA_R.csv")
    suspects = selectionner_praticiens_anomaux(df_pra)
    print("Suspects selectionnes :", suspects)
