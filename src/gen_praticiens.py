# ============================================================
# gen_praticiens.py — Generation IR_PSA_R
# Projet Detection anomalies DCIR
# Auteur : Octavien YAMESSE
# ============================================================

import pandas as pd
import numpy as np
import os
from config import DATA_RAW_DIR, CODES_POSTAUX_PARIS, REPARTITION_PRATICIENS, RANDOM_SEED

np.random.seed(RANDOM_SEED)

def generer_ir_psa_r():
    """
    Genere la table des praticiens IR_PSA_R.
    Une ligne par praticien actif exercant a Paris.
    Inspire de la vue VPRA du SIAM.
    """
    praticiens = []
    compteur = 1

    for specialite, params in REPARTITION_PRATICIENS.items():
        nb = params["nb"]
        praspe = params["praspe"]
        pracat = params["pracat"]
        secteur1_pct = params["secteur1_pct"]

        for _ in range(nb):
            pranum = f"PRA{str(compteur).zfill(6)}"

            # Secteur conventionnel
            if secteur1_pct == 1.0:
                cnvmtf = "1"
            else:
                cnvmtf = np.random.choice(
                    ["1", "3"],
                    p=[secteur1_pct, 1 - secteur1_pct]
                )

            # Sexe (55% H pour medecins, 70% F pour infirmiers)
            if pracat == 6:
                sexe = np.random.choice([1, 2], p=[0.30, 0.70])
            else:
                sexe = np.random.choice([1, 2], p=[0.55, 0.45])

            # Classe d'age
            ages_classes = ["< 40 ans", "40-49 ans", "50-59 ans", "60-69 ans", "> 70 ans"]
            ages_proba = [0.15, 0.25, 0.30, 0.20, 0.10]
            age_cls = np.random.choice(ages_classes, p=ages_proba)

            # Annee d'installation
            annee_installation = np.random.randint(1980, 2021)

            # Statut actif (98% actifs)
            statut_actif = np.random.choice([1, 0], p=[0.98, 0.02])

            # Lieu d'exercice
            if pracat == 6:  # infirmiers : beaucoup de visites domicile
                exe_lie_cod = np.random.choice([1, 2], p=[0.40, 0.60])
            elif pracat == 7:  # kines : cabinet
                exe_lie_cod = np.random.choice([1, 2], p=[0.80, 0.20])
            else:
                exe_lie_cod = np.random.choice([1, 2, 3], p=[0.75, 0.15, 0.10])

            # Statut d'exercice
            pfs_stt_cod = np.random.choice([1, 2, 3], p=[0.85, 0.10, 0.05])

            praticiens.append({
                "PRANUM_PRA": pranum,
                "PRACAT_PRA": pracat,
                "PRASPE_PRA": praspe,
                "CNVMTF_PRA": cnvmtf,
                "PRASEX_PRA": sexe,
                "AGECLS_PRA": age_cls,
                "BDICOD_PRA": np.random.choice(CODES_POSTAUX_PARIS),
                "PFS_STT_COD": pfs_stt_cod,
                "annee_installation": annee_installation,
                "statut_actif": statut_actif,
                "EXE_LIE_COD": exe_lie_cod,
                "specialite_label": specialite,
            })

            compteur += 1

    df = pd.DataFrame(praticiens)
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    df.to_csv(f"{DATA_RAW_DIR}IR_PSA_R.csv", index=False)
    print(f"IR_PSA_R : {len(df)} praticiens generes")
    return df


if __name__ == "__main__":
    df = generer_ir_psa_r()
    print(df.groupby("specialite_label")["PRANUM_PRA"].count())
