# ============================================================
# gen_prestations.py — Generation ER_PRS_F
# Projet Detection anomalies DCIR
# Auteur : Octavien YAMESSE
# ============================================================

import pandas as pd
import numpy as np
import os
from datetime import date, timedelta
from config import (DATA_RAW_DIR, DATE_DEBUT, DATE_FIN,
                    FACTEURS_MENSUELS, FACTEURS_HEBDO, RANDOM_SEED)

np.random.seed(RANDOM_SEED)

# Actes par specialite
ACTES_PAR_SPECIALITE = {
    "01": {"actes": ["C001", "V001"], "actes_tech": [], "moy_jour": 25, "std_jour": 5},
    "03": {"actes": ["C002"], "actes_tech": ["A002"], "moy_jour": 15, "std_jour": 4},
    "05": {"actes": ["C003"], "actes_tech": [], "moy_jour": 14, "std_jour": 3},
    "07": {"actes": ["C005"], "actes_tech": ["A001"], "moy_jour": 12, "std_jour": 3},
    "15": {"actes": ["C004"], "actes_tech": ["A003"], "moy_jour": 16, "std_jour": 4},
    "19": {"actes": ["D001", "D002", "D003"], "actes_tech": [], "moy_jour": 12, "std_jour": 3},
    "24": {"actes": ["I001", "I002", "I003"], "actes_tech": [], "moy_jour": 30, "std_jour": 8},
    "26": {"actes": ["K001", "K002"], "actes_tech": [], "moy_jour": 20, "std_jour": 5},
    "33": {"actes": ["C006"], "actes_tech": [], "moy_jour": 8, "std_jour": 2},
    "50": {"actes": [], "actes_tech": [], "moy_jour": 0, "std_jour": 0},
}

TARIFS = {
    "C001": 26.50, "C002": 30.00, "C003": 30.00,
    "C004": 30.00, "C005": 30.00, "C006": 45.73,
    "C007": 30.00, "D001": 26.50, "D002": 50.00,
    "D003": 120.00, "K001": 45.60, "K002": 38.40,
    "I001": 3.15,  "I002": 9.45,  "I003": 4.20,
    "A001": 30.00, "A002": 78.96, "A003": 22.96,
    "V001": 35.75,
}

def generer_dates_activite(annee=2023):
    """Genere toutes les dates de l'annee."""
    dates = []
    d = date(annee, 1, 1)
    while d <= date(annee, 12, 31):
        dates.append(d)
        d += timedelta(days=1)
    return dates

def calculer_nb_actes_jour(praspe, cnvmtf, jour):
    """
    Calcule le nombre d'actes pour un jour donne
    en tenant compte des facteurs saisonniers et hebdo.
    """
    if praspe not in ACTES_PAR_SPECIALITE:
        return 0

    params = ACTES_PAR_SPECIALITE[praspe]
    moy = params["moy_jour"]
    std = params["std_jour"]

    if moy == 0:
        return 0

    # Facteur mensuel et hebdomadaire
    facteur_mois = FACTEURS_MENSUELS.get(jour.month, 1.0)
    facteur_jour = FACTEURS_HEBDO.get(jour.weekday(), 1.0)

    # Nombre d'actes avec bruit gaussien
    nb = int(np.random.normal(moy, std) * facteur_mois * facteur_jour)
    return max(0, nb)

def calculer_montants(act_cod, cnvmtf, praspe):
    """
    Calcule BSE_REM_PRU, HON_MNT, DEP_MNT selon le secteur.
    """
    tarif_base = TARIFS.get(act_cod, 26.50)
    bse_rem_pru = tarif_base

    if cnvmtf == "1":
        # Secteur 1 : pas de depassement
        dep_mnt = 0.0
        hon_mnt = bse_rem_pru
    elif cnvmtf == "3":
        # Secteur 2 : depassement variable selon specialite
        taux_dep = max(0, np.random.normal(0.35, 0.15))
        dep_mnt = round(bse_rem_pru * taux_dep, 2)
        hon_mnt = round(bse_rem_pru + dep_mnt, 2)
    else:
        dep_mnt = 0.0
        hon_mnt = bse_rem_pru

    # Taux de remboursement
    rgo_rem_tau = 0.70  # par defaut
    bse_rem_mnt = round(bse_rem_pru * rgo_rem_tau, 2)

    return bse_rem_pru, hon_mnt, dep_mnt, bse_rem_mnt, rgo_rem_tau

def generer_er_prs_f(df_praticiens, df_beneficiaires):
    """
    Genere la table des prestations ER_PRS_F.
    Une ligne = une prestation facturee.
    """
    prestations = []
    compteur = 1
    dates_annee = generer_dates_activite()

    # Beneficiaires disponibles
    ben_list = df_beneficiaires["BEN_IDT_ANO"].tolist()

    print("Generation ER_PRS_F en cours...")

    for _, pra in df_praticiens.iterrows():
        pranum = pra["PRANUM_PRA"]
        praspe = str(pra["PRASPE_PRA"])
        cnvmtf = str(pra["CNVMTF_PRA"])

        if praspe not in ACTES_PAR_SPECIALITE:
            continue
        if ACTES_PAR_SPECIALITE[praspe]["moy_jour"] == 0:
            continue

        params = ACTES_PAR_SPECIALITE[praspe]
        actes_possibles = params["actes"] + params["actes_tech"]

        if not actes_possibles:
            continue

        # File de patients de ce praticien
        nb_patients = max(20, int(np.random.poisson(200)))
        nb_patients = min(nb_patients, len(ben_list))
        patients_praticien = np.random.choice(
            ben_list, size=nb_patients, replace=False
        )

        for jour in dates_annee:
            nb_actes = calculer_nb_actes_jour(praspe, cnvmtf, jour)

            if nb_actes == 0:
                continue

            for _ in range(nb_actes):
                # Choisir un acte
                act_cod = np.random.choice(actes_possibles)

                # Choisir un patient
                ben_idt = np.random.choice(patients_praticien)

                # Calculer les montants
                bse_rem_pru, hon_mnt, dep_mnt, bse_rem_mnt, rgo_rem_tau = \
                    calculer_montants(act_cod, cnvmtf, praspe)

                # Date de mise en paiement (0 a 30 jours apres)
                delai_paiement = np.random.randint(0, 31)
                flx_dis_dtd = jour + timedelta(days=delai_paiement)

                # Lieu d'execution
                exe_lie_cod = int(pra["EXE_LIE_COD"])

                prestations.append({
                    "PRS_ID": f"PRS{str(compteur).zfill(9)}",
                    "PRANUM_PRA": pranum,
                    "PFS_PRE_NUM": None,
                    "BEN_IDT_ANO": ben_idt,
                    "EXE_SOI_DTD": jour,
                    "PRS_NAT_COD": act_cod[0],
                    "ACT_COD": act_cod,
                    "BSE_REM_PRU": bse_rem_pru,
                    "HON_MNT": hon_mnt,
                    "DEP_MNT": dep_mnt,
                    "BSE_REM_MNT": bse_rem_mnt,
                    "RGO_REM_TAU": rgo_rem_tau,
                    "EXE_LIE_COD": exe_lie_cod,
                    "PRS_ACT_QTE": 1,
                    "PRS_ORD_NUM": None,
                    "FLX_DIS_DTD": flx_dis_dtd,
                    "origine_simulation": "NORMAL",
                })
                compteur += 1

        if compteur % 100000 == 0:
            print(f"  {compteur:,} prestations generees...")

    df = pd.DataFrame(prestations)
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    df.to_csv(f"{DATA_RAW_DIR}ER_PRS_F.csv", index=False)

    print(f"ER_PRS_F : {len(df):,} prestations generees")
    print(f"  Montant total facture : {df['HON_MNT'].sum():,.0f} euros")
    print(f"  Montant total rembourse : {df['BSE_REM_MNT'].sum():,.0f} euros")
    return df


if __name__ == "__main__":
    df_pra = pd.read_csv(f"{DATA_RAW_DIR}IR_PSA_R.csv")
    df_ben = pd.read_csv(f"{DATA_RAW_DIR}IR_BEN_R.csv")
    df = generer_er_prs_f(df_pra, df_ben)
