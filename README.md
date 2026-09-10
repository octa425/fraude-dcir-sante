# Detection d'anomalies DCIR - Simulation SNDS
## Contexte
Projet de detection d'anomalies et de comportements
potentiellement suspects dans les donnees de soins de ville,
inspire du DCIR (Datamart de Consommation Inter-Regimes)
et du SIAM (Systeme d'Information de l'Assurance Maladie).
**Important :** Isolation Forest detecte des observations
atypiques. Il ne qualifie pas juridiquement une fraude.
Le terme "comportement suspect" est deliberement prefere.
## Architecture




## Tables simulees

| Table | Description | Lignes |

|-------|-------------|--------|

| IR_PSA_R | Referentiel praticiens (SIAM/VPRA) | 650 |

| IR_BEN_R | Referentiel beneficiaires | 10 000 |

| ER_PRS_F | Prestations DCIR | ~3 300 000 |

| ER_PHA_F | Pharmacie DCIR | ~240 000 |

| REF_ACTES | Referentiel actes | 19 |

| REF_SPECIALITES | Referentiel specialites | 10 |

## Les 5 scenarios d'anomalies injectees

| Scenario | Description | Nb praticiens |

|----------|-------------|---------------|

| S1 | Suractivite (actes/jour anormalement eleve) | 15 |

| S2 | Actes fantomes (post-mortem, beneficiaires fictifs) | 10 |

| S3 | Incoherence acte/specialite | 12 |

| S4 | Anomalie tarifaire vs pairs | 15 |

| S5 | Anomalie relationnelle (reseau pharmacies) | 13 |

**Total anomalies : 63 praticiens sur 650 (9.7%)**

## Generer les donnees

Les CSV ne sont pas dans le repo (trop volumineux).

Pour les generer localement :

```bash

cd src

python3 main.py

# Generation en ~12 minutes

# Fichiers generes dans data/raw/

```

## Stack technique

- Python 3.x

- pandas, numpy

- scikit-learn (Isolation Forest)

- FastAPI + Docker

- GitHub Actions (CI/CD)

- Render (deploiement)

## Auteur

Octavien YAMESSE  Data Scientist 

github.com/octa425


## Perspectives d'amelioration

### 1. Robustesse du score d'anomalie
Le score normalise entre 0 et 1 utilise
une normalisation MinMax dependante des
donnees d'entrainement actuelles.
En production sur des flux live, il faudra
figer les bornes Min/Max historiques ou
utiliser directement score_samples() natif
de scikit-learn.

### 2. Simplification du pipeline ML
Le StandardScaler pourra etre retire lors
d'un refactoring. Isolation Forest etant
base sur des arbres de decision, il est
insensible a l'echelle des variables.
Aucune perte de performance attendue.

### 3. Traitement des valeurs manquantes
Le fillna(0) actuel est justifie pour le
contexte metier des auxiliaires medicaux
(zscore depassement = 0 car secteur 1).
A surveiller si de nouvelles specialites
avec d'autres structures de donnees
manquantes entrent dans le perimetre.
