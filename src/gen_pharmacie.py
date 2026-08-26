# ============================================================
# gen_pharmacie.py — Generation ER_PHA_F
# Projet Detection anomalies DCIR
# Auteur : Octavien YAMESSE
# ============================================================

import pandas as pd
import numpy as np
import os
from datetime import date, timedelta
from config import DATA_RAW_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)

MEDICAMENTS = [
    ("3400936789012", "Metformine 850mg",   "A10B", "A",  8.20),
    ("3400923456789", "Ramipril 5mg",       "C09A", "C", 15.50),
    ("3400934567890", "Metoprolol 100mg",   "C07A", "C", 12.30),
    ("3400945678901", "Rivaroxaban 20mg",   "B01A", "B", 85.00),
    ("3400956789012", "Warfarine 2mg",      "B01A", "B",  4.50),
    ("3400967890123", "Amoxicilline 1g",    "J01C", "J",  6.80),
    ("3400978901234", "Paracetamol 1g",     "N02B", "N",  2.10),
    ("3400989012345", "Ibuprofene 400mg",   "M01A", "M",  3.50),
    ("3400990123456", "Sertraline 50mg",    "N06A", "N", 18.90),
    ("3400901234567", "Salbutamol spray",   "R03A", "R",  4.20),
    ("3400912345678", "Levothyrox 100",     "H03A", "H",  9.80),
    ("3400913456789", "Atorvastatine 20mg", "C10A", "C", 22.50),
    ("3400924567890", "Pantoprazole 40mg",  "A02B", "A", 11.20),
    ("3400935678901", "Alprazolam 0.5mg",   "N05B", "N",  8.60),
    ("3400946789012", "Morphine 10mg",      "N02A", "N", 15.30),
]

def generer_er_pha_f(df_praticiens, df_beneficiaires, df_prestations):
    """
    Genere la table des delivrances pharmaceutiques.
    Approche : pour chaque medecin prescripteur,
    on genere directement N delivrances aleatoires.
    """
    delivrances = []
    compteur = 1
    ord_compteur = 1

    # Pharmacies disponibles
    pharmacies = df_praticiens[
        df_praticiens["PRACAT_PRA"] == 50
    ]["PRANUM_PRA"].tolist()

    if not pharmacies:
        print("ERREUR : aucune pharmacie trouvee !")
        return pd.DataFrame()

    # Prescripteurs = medecins (PRACAT=1)
    prescripteurs = df_praticiens[
        df_praticiens["PRACAT_PRA"] == 1
    ]["PRANUM_PRA"].tolist()

    print(f"  {len(prescripteurs)} prescripteurs, {len(pharmacies)} pharmacies")

    # Beneficiaires disponibles
    ben_list = df_beneficiaires["BEN_IDT_ANO"].tolist()

    print("Generation ER_PHA_F en cours...")

    for pranum in prescripteurs:
        # Pharmacie principale de ce prescripteur
        pharma_principale = np.random.choice(pharmacies)
        concentration = np.random.uniform(0.30, 0.40)

        # Nombre de delivrances pour ce praticien sur l'annee
        # Un MG avec 1000 patients → ~800 prescriptions/an
        nb_delivrances = int(np.random.poisson(800))

        for i in range(nb_delivrances):
            # Medicament aleatoire
            med = MEDICAMENTS[np.random.randint(len(MEDICAMENTS))]
            cip, nom, atc3, atc1, prix = med

            # Patient aleatoire
            ben_idt = np.random.choice(ben_list)

            # Pharmacie
            if np.random.random() < concentration:
                pharma = pharma_principale
            else:
                pharma = np.random.choice(pharmacies)

            # Dates
            jours = np.random.randint(0, 365)
            date_presc = date(2023, 1, 1) + timedelta(days=jours)
            date_deliv = date_presc + timedelta(days=np.random.randint(0, 8))
            if date_deliv > date(2023, 12, 31):
                date_deliv = date(2023, 12, 31)

            # Boites et montants
            nb_boites = np.random.randint(1, 4)
            bse_rem_mnt = round(prix * nb_boites * 0.65, 2)

            ord_id = f"ORD{str(ord_compteur).zfill(9)}"
            ord_compteur += 1

            delivrances.append({
                "DLI_ID": f"DLI{str(compteur).zfill(9)}",
                "ORD_ID": ord_id,
                "PFS_PRE_NUM": pranum,
                "PRANUM_EXE": pharma,
                "BEN_IDT_ANO": ben_idt,
                "EXE_SOI_DTD": date_deliv,
                "PRS_PRE_DTD": date_presc,
                "PHA_CIP_COD": cip,
                "PHA_NOM": nom,
                "PHA_ATC_NIV3": atc3,
                "PHA_ATC_NIV1": atc1,
                "BSE_REM_MNT": bse_rem_mnt,
                "PHA_PRIX_UNI": prix,
                "nb_boites": nb_boites,
                "PHA_QSU_PRE": nb_boites,
                "origine_simulation": "NORMAL",
            })
            compteur += 1

        if compteur % 100000 == 0:
            print(f"  {compteur:,} delivrances generees...")

    df = pd.DataFrame(delivrances)
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    df.to_csv(f"{DATA_RAW_DIR}ER_PHA_F.csv", index=False)

    print(f"ER_PHA_F : {len(df):,} delivrances generees")
    print(f"  Montant total rembourse : {df['BSE_REM_MNT'].sum():,.0f} euros")
    return df


if __name__ == "__main__":
    df_pra = pd.read_csv(f"{DATA_RAW_DIR}IR_PSA_R.csv")
    df_ben = pd.read_csv(f"{DATA_RAW_DIR}IR_BEN_R.csv")
    df_prs = pd.read_csv(f"{DATA_RAW_DIR}ER_PRS_F.csv")
    df = generer_er_pha_f(df_pra, df_ben, df_prs)
