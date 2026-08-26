# ============================================================
# gen_beneficiaires.py — Generation IR_BEN_R
# Projet Detection anomalies DCIR
# Auteur : Octavien YAMESSE
# ============================================================

import pandas as pd
import numpy as np
import os
from datetime import date, timedelta
from config import DATA_RAW_DIR, NB_BENEFICIAIRES, RANDOM_SEED

np.random.seed(RANDOM_SEED)

def generer_ir_ben_r():
    """
    Genere la table des beneficiaires IR_BEN_R.
    BEN_IDT_ANO = identifiant pseudonymise (remplace le NIR).
    BEN_DCD_DTE = date de deces pour le scenario S2.
    """
    beneficiaires = []

    for i in range(1, NB_BENEFICIAIRES + 1):
        ben_idt = f"BEN{str(i).zfill(7)}"

        # Sexe
        sexe = np.random.choice([1, 2], p=[0.48, 0.52])

        # Annee de naissance (distribution realiste)
        annee_naissance = int(np.random.normal(1971, 18))
        annee_naissance = max(1928, min(2005, annee_naissance))

        # Departement de residence
        dep = np.random.choice(
            ["075", "092", "093", "094"],
            p=[0.85, 0.05, 0.05, 0.05]
        )

        # Code commune
        if dep == "075":
            arrond = np.random.randint(1, 21)
            code_com = f"751{str(arrond).zfill(2)}"
        else:
            code_com = dep + "000"

        # ALD (Affection Longue Duree)
        ald = np.random.choice([0, 1], p=[0.85, 0.15])

        # CMU
        cmu = np.random.choice([0, 1], p=[0.92, 0.08])

        # Regime d'affiliation
        if cmu == 1:
            rgt_cod = 2
        else:
            rgt_cod = np.random.choice([1, 3], p=[0.97, 0.03])

        # Date d'ouverture des droits
        annees_droits = np.random.randint(2000, 2023)
        mois_droits = np.random.randint(1, 13)
        drt_aff = date(annees_droits, mois_droits, 1)

        # Date de deces (2% des beneficiaires)
        # → utilise pour le scenario S2 (actes post-mortem)
        dcd_dte = None
        if np.random.random() < 0.02:
            # Deces en 2023
            jours_deces = np.random.randint(1, 365)
            dcd_dte = date(2023, 1, 1) + timedelta(days=jours_deces)

        beneficiaires.append({
            "BEN_IDT_ANO": ben_idt,
            "BEN_SEX_COD": sexe,
            "BEN_NAI_ANN": annee_naissance,
            "BEN_RES_DPT": dep,
            "BEN_RES_COM": code_com,
            "ALD_COD": ald,
            "BEN_CMU_TOP": cmu,
            "BEN_RGT_COD": rgt_cod,
            "BEN_DRT_AFF": drt_aff,
            "BEN_DCD_DTE": dcd_dte,
        })

    df = pd.DataFrame(beneficiaires)
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    df.to_csv(f"{DATA_RAW_DIR}IR_BEN_R.csv", index=False)

    nb_decedes = df["BEN_DCD_DTE"].notna().sum()
    nb_ald = df["ALD_COD"].sum()
    nb_cmu = df["BEN_CMU_TOP"].sum()

    print(f"IR_BEN_R : {len(df)} beneficiaires generes")
    print(f"  Dont decedes : {nb_decedes} ({nb_decedes/len(df)*100:.1f}%)")
    print(f"  Dont ALD     : {nb_ald} ({nb_ald/len(df)*100:.1f}%)")
    print(f"  Dont CMU     : {nb_cmu} ({nb_cmu/len(df)*100:.1f}%)")
    return df


if __name__ == "__main__":
    df = generer_ir_ben_r()
