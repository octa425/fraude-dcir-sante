# ============================================================
# isolation_forest.py — Detection d'anomalies
# Projet Detection anomalies DCIR
# Auteur : Octavien YAMESSE
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Features ML — noms en minuscules (PostgreSQL)
FEATURES_ML = [
    "actes_par_jour_moyen",
    "actes_par_jour_max",
    "ecart_type_actes_jour",
    "part_actes_weekend_pct",
    "zscore_actes_jour",
    "nb_beneficiaires_uniques",
    "actes_par_beneficiaire",
    "nb_actes_post_mortem",
    "nb_beneficiaires_fictifs",
    "montant_moyen_acte",
    "taux_depassement_pct",
    "evolution_montant_pct",
    "zscore_depassement",
    "nb_actes_distincts",
    "part_incompatibles_pct",
    "nb_pharmacies_distinctes",
    "concentration_pharmacie_pct",
]

def charger_features():
    chemin = "data/features/features_professionnels.csv"
    df = pd.read_csv(chemin)
    print(f"Features chargees : {len(df)} praticiens")
    print(f"Colonnes : {list(df.columns[:5])}...")

    nb_nan = df[FEATURES_ML].isna().sum().sum()
    print(f"Valeurs manquantes : {nb_nan} → remplacees par 0")
    df[FEATURES_ML] = df[FEATURES_ML].fillna(0)
    return df

def entrainer_isolation_forest(df):
    """
    Isolation Forest :
    - contamination=0.10 : ~10% suspects attendus
    - n_estimators=200   : 200 arbres pour la stabilite
    - StandardScaler     : normalise les features
                           avant l'entrainement
    """
    X = df[FEATURES_ML].values

    # Normalisation obligatoire
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\nFeatures normalisees : {X_scaled.shape}")

    print("Entrainement Isolation Forest...")
    modele = IsolationForest(
        n_estimators=200,
        contamination=0.10,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    modele.fit(X_scaled)

    # Score brut : negatif = plus anormal
    scores_bruts = modele.score_samples(X_scaled)
    predictions = modele.predict(X_scaled)
    anomalie_predite = (predictions == -1).astype(int)

    # Normaliser score entre 0 et 1
    # 1 = tres anormal / 0 = tres normal
    score_normalise = -scores_bruts
    score_normalise = (
        (score_normalise - score_normalise.min()) /
        (score_normalise.max() - score_normalise.min())
    )

    df["score_anomalie"] = score_normalise
    df["anomalie_predite"] = anomalie_predite
    df["score_brut"] = scores_bruts

    nb_detectes = anomalie_predite.sum()
    print(f"Anomalies detectees : {nb_detectes} ({nb_detectes/len(df)*100:.1f}%)")

    return df, modele, scaler

def evaluer_modele(df):
    """
    Evaluation UNIQUEMENT apres entrainement.
    Ground truth jamais vu pendant l'entrainement.
    """
    print("\n" + "="*50)
    print("EVALUATION DU MODELE")
    print("="*50)

    # Charger ground_truth (colonnes en MAJUSCULES)
    gt = pd.read_csv("src/data/ground_truth.csv")

    # Merger sur PRANUM_PRA (gt) avec pranum_pra (features)
    df_eval = df.merge(
        gt[["PRANUM_PRA", "ANOMALIE_ANY",
            "S1_SURACTIVITE", "S2_ACTE_FANTOME",
            "S3_INCOHERENCE_SPECIALITE",
            "S4_ANOMALIE_TARIFAIRE",
            "S5_RESEAU_ATYPIQUE",
            "SEVERITY"]],
        left_on="pranum_pra",
        right_on="PRANUM_PRA"
    )

    print(f"Merge reussi : {len(df_eval)} praticiens")

    # Evaluation globale
    vp = ((df_eval["anomalie_predite"] == 1) &
          (df_eval["ANOMALIE_ANY"] == 1)).sum()
    fp = ((df_eval["anomalie_predite"] == 1) &
          (df_eval["ANOMALIE_ANY"] == 0)).sum()
    fn = ((df_eval["anomalie_predite"] == 0) &
          (df_eval["ANOMALIE_ANY"] == 1)).sum()
    vn = ((df_eval["anomalie_predite"] == 0) &
          (df_eval["ANOMALIE_ANY"] == 0)).sum()

    precision = vp / max(vp + fp, 1)
    rappel    = vp / max(vp + fn, 1)
    f1        = 2 * precision * rappel / max(precision + rappel, 0.001)

    print(f"\nMatrice de confusion :")
    print(f"  Vrais positifs  : {vp}")
    print(f"  Faux positifs   : {fp}")
    print(f"  Faux negatifs   : {fn}")
    print(f"  Vrais negatifs  : {vn}")

    print(f"\nMetriques globales :")
    print(f"  Precision : {precision:.3f}")
    print(f"  Rappel    : {rappel:.3f}")
    print(f"  F1-score  : {f1:.3f}")

    # Evaluation par scenario
    print(f"\nDetection par scenario :")
    scenarios = [
        "S1_SURACTIVITE", "S2_ACTE_FANTOME",
        "S3_INCOHERENCE_SPECIALITE",
        "S4_ANOMALIE_TARIFAIRE", "S5_RESEAU_ATYPIQUE"
    ]
    for scenario in scenarios:
        vrais = df_eval[df_eval[scenario] == 1]
        if len(vrais) == 0:
            continue
        detectes = (vrais["anomalie_predite"] == 1).sum()
        taux = detectes / len(vrais) * 100
        print(f"  {scenario:<35} : {detectes:>2}/{len(vrais):>2} ({taux:.0f}%)")

    # Top 20 suspects
    print(f"\nTOP 20 suspects :")
    top20 = df_eval.nlargest(20, "score_anomalie")[
        ["PRANUM_PRA", "specialite_label",
         "score_anomalie", "ANOMALIE_ANY", "SEVERITY"]
    ]
    print(top20.to_string(index=False))

    return df_eval

def sauvegarder_resultats(df_eval):
    os.makedirs("data/resultats", exist_ok=True)
    df_eval.to_csv("data/resultats/scores_anomalie.csv", index=False)

    top_suspects = df_eval[
        df_eval["score_anomalie"] > 0.7
    ].sort_values("score_anomalie", ascending=False)
    top_suspects.to_csv("data/resultats/top_suspects.csv", index=False)

    print(f"\nResultats sauvegardes :")
    print(f"  scores_anomalie.csv : {len(df_eval)} praticiens")
    print(f"  top_suspects.csv    : {len(top_suspects)} suspects")

if __name__ == "__main__":
    print("="*60)
    print("ISOLATION FOREST — Detection anomalies DCIR")
    print("="*60)

    df = charger_features()
    df, modele, scaler = entrainer_isolation_forest(df)
    df_eval = evaluer_modele(df)
    sauvegarder_resultats(df_eval)
    print("\n✅ Isolation Forest termine !")
