# ============================================================
# ground_truth.py — Generation du fichier ground_truth.csv
# Projet Detection anomalies DCIR
# Auteur : Octavien YAMESSE
# RAPPEL : ce fichier ne rentre JAMAIS dans le modele ML
# ============================================================

import pandas as pd
import numpy as np
import os
from config import DATA_RAW_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)

def generer_ground_truth(df_praticiens, suspects):
    """
    Genere le fichier ground_truth.csv.
    Une ligne par praticien.
    Les colonnes S1 a S5 sont des flags booleens (0/1).
    ANOMALIE_ANY = 1 si au moins un scenario actif.
    SEVERITY = NONE / LOW / MEDIUM / HIGH.
    Ce fichier est completement separe des tables sources.
    Il ne rentre JAMAIS dans Isolation Forest.
    """

    # Inverser le dictionnaire suspects pour lookup rapide
    # {PRANUM_PRA: [scenarios]}
    praticien_scenarios = {}
    for scenario, pra_list in suspects.items():
        for pranum in pra_list:
            if pranum not in praticien_scenarios:
                praticien_scenarios[pranum] = []
            praticien_scenarios[pranum].append(scenario)

    ground_truth = []

    for _, pra in df_praticiens.iterrows():
        pranum = pra["PRANUM_PRA"]
        scenarios_actifs = praticien_scenarios.get(pranum, [])

        s1 = 1 if "S1" in scenarios_actifs else 0
        s2 = 1 if "S2" in scenarios_actifs else 0
        s3 = 1 if "S3" in scenarios_actifs else 0
        s4 = 1 if "S4" in scenarios_actifs else 0
        s5 = 1 if "S5" in scenarios_actifs else 0
        anomalie_any = 1 if any([s1, s2, s3, s4, s5]) else 0
        nb_scenarios = s1 + s2 + s3 + s4 + s5

        # Severity selon le nombre de scenarios
        if nb_scenarios == 0:
            severity = "NONE"
        elif nb_scenarios == 1:
            # S2 et S1 sont toujours HIGH
            if s2 == 1 or s1 == 1:
                severity = "HIGH"
            elif s4 == 1:
                severity = "HIGH"
            else:
                severity = "MEDIUM"
        else:
            severity = "HIGH"

        # Scenario principal pour la restitution
        if nb_scenarios == 0:
            scenario_principal = "NORMAL"
        elif nb_scenarios == 1:
            scenario_principal = scenarios_actifs[0]
        else:
            scenario_principal = "+".join(sorted(scenarios_actifs))

        ground_truth.append({
            "PRANUM_PRA": pranum,
            "S1_SURACTIVITE": s1,
            "S2_ACTE_FANTOME": s2,
            "S3_INCOHERENCE_SPECIALITE": s3,
            "S4_ANOMALIE_TARIFAIRE": s4,
            "S5_RESEAU_ATYPIQUE": s5,
            "ANOMALIE_ANY": anomalie_any,
            "nb_scenarios": nb_scenarios,
            "scenario_principal": scenario_principal,
            "SEVERITY": severity,
            "PRASPE_PRA": pra["PRASPE_PRA"],
            "specialite_label": pra["specialite_label"],
        })

    df_gt = pd.DataFrame(ground_truth)

    # Sauvegarder SEPAREMENT des tables sources
    # dans un dossier dedie
    os.makedirs("data/", exist_ok=True)
    df_gt.to_csv("data/ground_truth.csv", index=False)

    # Statistiques
    nb_anomalies = df_gt["ANOMALIE_ANY"].sum()
    nb_normaux = (df_gt["ANOMALIE_ANY"] == 0).sum()

    print(f"\nGROUND TRUTH genere :")
    print(f"  Total praticiens  : {len(df_gt)}")
    print(f"  Normaux           : {nb_normaux} ({nb_normaux/len(df_gt)*100:.1f}%)")
    print(f"  Anomalies         : {nb_anomalies} ({nb_anomalies/len(df_gt)*100:.1f}%)")
    print(f"  S1 Suractivite    : {df_gt['S1_SURACTIVITE'].sum()}")
    print(f"  S2 Actes fantomes : {df_gt['S2_ACTE_FANTOME'].sum()}")
    print(f"  S3 Incoherence    : {df_gt['S3_INCOHERENCE_SPECIALITE'].sum()}")
    print(f"  S4 Tarifaire      : {df_gt['S4_ANOMALIE_TARIFAIRE'].sum()}")
    print(f"  S5 Reseau         : {df_gt['S5_RESEAU_ATYPIQUE'].sum()}")
    print(f"\n  Sauvegarde : data/ground_truth.csv")
    print(f"  RAPPEL : ce fichier ne rentre JAMAIS dans le ML !")

    return df_gt


if __name__ == "__main__":
    df_pra = pd.read_csv(f"{DATA_RAW_DIR}IR_PSA_R.csv")
    suspects_test = {
        "S1": ["PRA000001", "PRA000002"],
        "S2": ["PRA000003"],
        "S3": [],
        "S4": ["PRA000004"],
        "S5": [],
    }
    df_gt = generer_ground_truth(df_pra, suspects_test)
    print(df_gt.head())
