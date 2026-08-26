# ============================================================
# main.py — Orchestrateur principal
# Projet Detection anomalies DCIR
# Auteur : Octavien YAMESSE
# ============================================================

import sys
import os
import time
import pandas as pd
import numpy as np

# Ajouter le dossier src au path
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_RAW_DIR, RANDOM_SEED
from ref_tables import generer_ref_specialites, generer_ref_actes
from gen_praticiens import generer_ir_psa_r
from gen_beneficiaires import generer_ir_ben_r
from gen_prestations import generer_er_prs_f
from gen_pharmacie import generer_er_pha_f
from gen_anomalies import (
    selectionner_praticiens_anomaux,
    injecter_s1_suractivite,
    injecter_s2_actes_fantomes,
    injecter_s3_incoherence,
    injecter_s4_anomalie_tarifaire,
    injecter_s5_reseau,
)
from ground_truth import generer_ground_truth

np.random.seed(RANDOM_SEED)

def verifier_qualite(df_pra, df_ben, df_prs, df_pha, df_gt):
    """
    Controle qualite apres generation.
    Verifie les cardinalites et les relations.
    """
    print("\n" + "="*50)
    print("CONTROLE QUALITE")
    print("="*50)

    # Cardinalites
    print(f"\nCardinalites :")
    print(f"  IR_PSA_R  : {len(df_pra):>8,} praticiens")
    print(f"  IR_BEN_R  : {len(df_ben):>8,} beneficiaires")
    print(f"  ER_PRS_F  : {len(df_prs):>8,} prestations")
    print(f"  ER_PHA_F  : {len(df_pha):>8,} delivrances")
    print(f"  GT        : {len(df_gt):>8,} lignes")

    # Coherence ground_truth
    assert len(df_gt) == len(df_pra), \
        "Ground truth doit avoir autant de lignes que IR_PSA_R !"

    # Verification praticiens dans prestations
    pra_dans_prs = set(df_prs["PRANUM_PRA"].unique())
    pra_dans_ir = set(df_pra["PRANUM_PRA"].unique())
    pra_inconnus = pra_dans_prs - pra_dans_ir
    print(f"\n  Praticiens dans ER_PRS_F non dans IR_PSA_R : {len(pra_inconnus)}")

    # Verification beneficiaires fictifs S2b
    ben_dans_ir = set(df_ben["BEN_IDT_ANO"].unique())
    ben_dans_prs = set(df_prs["BEN_IDT_ANO"].unique())
    ben_fictifs = [b for b in ben_dans_prs - ben_dans_ir if b.startswith("BEN999")]
    print(f"  Beneficiaires fictifs S2b : {len(ben_fictifs)}")

    # Distribution anomalies
    print(f"\nDistribution anomalies :")
    print(f"  Normaux   : {(df_gt['ANOMALIE_ANY']==0).sum()}")
    print(f"  Anomalies : {(df_gt['ANOMALIE_ANY']==1).sum()}")

    # Distribution origine_simulation
    print(f"\nOriginе simulation dans ER_PRS_F :")
    print(df_prs["origine_simulation"].value_counts().to_string())

    print("\n✅ Controle qualite termine !")


def main():
    """
    Orchestrateur principal de generation.
    Ordre strict respectant les dependances entre tables.
    """
    debut = time.time()

    print("="*60)
    print("GENERATION DES DONNEES DCIR SIMULEES")
    print("Projet Detection anomalies — Octavien YAMESSE")
    print("="*60)

    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    # ── ETAPE 1 : Referentiels ─────────────────────────────
    print("\n[1/9] Generation des referentiels...")
    df_ref_spe = generer_ref_specialites()
    df_ref_act = generer_ref_actes()

    # ── ETAPE 2 : Praticiens ───────────────────────────────
    print("\n[2/9] Generation IR_PSA_R (praticiens)...")
    df_praticiens = generer_ir_psa_r()

    # ── ETAPE 3 : Beneficiaires ────────────────────────────
    print("\n[3/9] Generation IR_BEN_R (beneficiaires)...")
    df_beneficiaires = generer_ir_ben_r()

    # ── ETAPE 4 : Prestations normales ────────────────────
    print("\n[4/9] Generation ER_PRS_F (prestations normales)...")
    df_prestations = generer_er_prs_f(df_praticiens, df_beneficiaires)

    # ── ETAPE 5 : Pharmacie normale ───────────────────────
    print("\n[5/9] Generation ER_PHA_F (pharmacie normale)...")
    df_pharmacie = generer_er_pha_f(
        df_praticiens, df_beneficiaires, df_prestations
    )

    # ── ETAPE 6 : Selection praticiens anomaux ────────────
    print("\n[6/9] Selection des praticiens anomaux...")
    suspects = selectionner_praticiens_anomaux(df_praticiens)

    # ── ETAPE 7 : Injection anomalies ─────────────────────
    print("\n[7/9] Injection des anomalies...")

    df_prestations = injecter_s1_suractivite(
        df_prestations, suspects["S1"]
    )
    df_prestations = injecter_s2_actes_fantomes(
        df_prestations, df_beneficiaires,
        df_praticiens, suspects["S2"]
    )
    df_prestations = injecter_s3_incoherence(
        df_prestations, df_praticiens, suspects["S3"]
    )
    df_prestations = injecter_s4_anomalie_tarifaire(
        df_prestations, df_praticiens, suspects["S4"]
    )
    df_pharmacie = injecter_s5_reseau(
        df_pharmacie, df_praticiens, suspects["S5"]
    )

    # ── ETAPE 8 : Sauvegarder tables finales ──────────────
    print("\n[8/9] Sauvegarde des tables finales...")
    df_prestations.to_csv(f"{DATA_RAW_DIR}ER_PRS_F.csv", index=False)
    df_pharmacie.to_csv(f"{DATA_RAW_DIR}ER_PHA_F.csv", index=False)
    print(f"  ER_PRS_F final : {len(df_prestations):,} lignes")
    print(f"  ER_PHA_F final : {len(df_pharmacie):,} lignes")

    # ── ETAPE 9 : Ground truth ────────────────────────────
    print("\n[9/9] Generation ground_truth.csv...")
    df_gt = generer_ground_truth(df_praticiens, suspects)

    # ── CONTROLE QUALITE ──────────────────────────────────
    verifier_qualite(
        df_praticiens, df_beneficiaires,
        df_prestations, df_pharmacie, df_gt
    )

    # ── RESUME FINAL ──────────────────────────────────────
    duree = time.time() - debut
    print(f"\n{'='*60}")
    print(f"GENERATION TERMINEE en {duree:.1f} secondes")
    print(f"{'='*60}")
    print(f"\nFichiers generes dans {DATA_RAW_DIR} :")
    for f in os.listdir(DATA_RAW_DIR):
        taille = os.path.getsize(f"{DATA_RAW_DIR}{f}") / 1024 / 1024
        print(f"  {f:<30} {taille:.1f} MB")
    print(f"\nGround truth : data/ground_truth.csv")
    print(f"  → NE JAMAIS utiliser dans le modele ML !")


if __name__ == "__main__":
    main()
