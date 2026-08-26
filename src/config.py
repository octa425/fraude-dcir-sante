# ============================================================
# config.py — Parametres de generation
# Projet Detection anomalies DCIR
# Auteur : Octavien YAMESSE
# ============================================================

import numpy as np
from datetime import date

# --- Graine aleatoire pour reproductibilite ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- Periode de simulation ---
DATE_DEBUT = date(2023, 1, 1)
DATE_FIN   = date(2023, 12, 31)
ANNEE      = 2023

# --- Volumetrie ---
NB_PRATICIENS     = 650
NB_BENEFICIAIRES  = 10_000

# --- Repartition praticiens par specialite ---
REPARTITION_PRATICIENS = {
    "MG":           {"nb": 150, "praspe": "01", "pracat": 1, "secteur1_pct": 0.80},
    "Cardiologue":  {"nb": 30,  "praspe": "03", "pracat": 1, "secteur1_pct": 0.40},
    "Dermatologue": {"nb": 25,  "praspe": "05", "pracat": 1, "secteur1_pct": 0.30},
    "Ophtalmo":     {"nb": 20,  "praspe": "15", "pracat": 1, "secteur1_pct": 0.30},
    "Specialistes": {"nb": 75,  "praspe": "07", "pracat": 1, "secteur1_pct": 0.40},
    "Dentiste":     {"nb": 80,  "praspe": "19", "pracat": 4, "secteur1_pct": 0.60},
    "Infirmier":    {"nb": 120, "praspe": "24", "pracat": 6, "secteur1_pct": 1.00},
    "Kine":         {"nb": 100, "praspe": "26", "pracat": 7, "secteur1_pct": 1.00},
    "Pharmacien":   {"nb": 50,  "praspe": "50", "pracat": 50,"secteur1_pct": 1.00},
}

# --- Anomalies a injecter ---
ANOMALIES = {
    "S1": 15,  # Suractivite
    "S2": 10,  # Actes fantomes
    "S3": 12,  # Incoherence acte/specialite
    "S4": 15,  # Anomalie tarifaire
    "S5": 13,  # Anomalie relationnelle
}

# --- Codes postaux Paris ---
CODES_POSTAUX_PARIS = [f"750{str(i).zfill(2)}" for i in range(1, 21)]

# --- Facteurs saisonniers par mois ---
FACTEURS_MENSUELS = {
    1: 0.90, 2: 1.10, 3: 1.05, 4: 0.95,
    5: 0.85, 6: 1.00, 7: 0.60, 8: 0.45,
    9: 1.10, 10: 1.05, 11: 1.10, 12: 0.80
}

# --- Facteurs par jour de semaine (0=Lundi, 6=Dimanche) ---
FACTEURS_HEBDO = {
    0: 1.20, 1: 1.10, 2: 1.00,
    3: 1.05, 4: 0.95, 5: 0.50, 6: 0.05
}

# --- Chemins de sortie ---
DATA_RAW_DIR = "data/raw/"
