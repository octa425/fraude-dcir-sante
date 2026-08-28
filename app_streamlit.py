# ============================================================
# app_streamlit.py Dashboard Detection anomalies DCIR
# Projet Detection anomalies DCIR
# Auteur : Octavien YAMESSE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ── Configuration page ────────────────────────────────────
st.set_page_config(
    page_title="Detection Anomalies DCIR",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Chargement des donnees ────────────────────────────────
@st.cache_data
def charger_donnees():
    df = pd.read_csv("data/resultats/scores_anomalie.csv")
    if "SEVERITY" in df.columns:
        df = df.rename(columns={
            "SEVERITY": "SEVERITY",
            "ANOMALIE_ANY_x": "ANOMALIE_ANY",
            "PRANUM_PRA_x": "PRANUM_PRA_gt"
        })
    return df

df = charger_donnees()

# ── Normes par specialite ─────────────────────────────────
NORMES = {
    "MG":           {"actes_j": (20,30), "dep_pct": (0,5),   "incompat": (0,2), "pharma_conc": (30,40), "pharma_nb": (8,15),  "montant": (26,30)},
    "Cardiologue":  {"actes_j": (12,18), "dep_pct": (25,45), "incompat": (0,2), "pharma_conc": (30,40), "pharma_nb": (8,15),  "montant": (55,75)},
    "Dermatologue": {"actes_j": (11,17), "dep_pct": (30,60), "incompat": (0,2), "pharma_conc": (30,40), "pharma_nb": (8,15),  "montant": (45,65)},
    "Ophtalmo":     {"actes_j": (13,19), "dep_pct": (30,50), "incompat": (0,2), "pharma_conc": (30,40), "pharma_nb": (8,15),  "montant": (50,70)},
    "Specialistes": {"actes_j": (10,15), "dep_pct": (25,45), "incompat": (0,2), "pharma_conc": (30,40), "pharma_nb": (8,15),  "montant": (40,60)},
    "Dentiste":     {"actes_j": (9,15),  "dep_pct": (15,35), "incompat": (0,2), "pharma_conc": (30,40), "pharma_nb": (8,15),  "montant": (70,100)},
    "Infirmier":    {"actes_j": (25,35), "dep_pct": (0,0),   "incompat": (0,2), "pharma_conc": (0,0),   "pharma_nb": (0,0),   "montant": (5,12)},
    "Kine":         {"actes_j": (15,25), "dep_pct": (0,0),   "incompat": (0,2), "pharma_conc": (0,0),   "pharma_nb": (0,0),   "montant": (40,50)},
    "Pharmacien":   {"actes_j": (0,0),   "dep_pct": (0,0),   "incompat": (0,0), "pharma_conc": (0,0),   "pharma_nb": (0,0),   "montant": (0,0)},
}

def signal(valeur, norme_min, norme_max):
    if norme_max == 0 and norme_min == 0:
        return "⚪ N/A"
    if norme_min <= float(valeur) <= norme_max:
        return f"🟢 NORMAL"
    elif float(valeur) > norme_max:
        ratio = round(float(valeur) / norme_max, 1)
        return f"🔴 ANORMAL (x{ratio} vs max {norme_max})"
    else:
        return f"🟠 BAS (min {norme_min})"

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("🏥 Detection Anomalies DCIR")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Vue globale", "Top suspects",
     "Analyse par scenario", "Fiche praticien"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filtres**")

specialites = ["Toutes"] + sorted(df["specialite_label"].unique().tolist())
spe_filtre = st.sidebar.selectbox("Specialite", specialites)

seuil_score = st.sidebar.slider(
    "Seuil score anomalie",
    min_value=0.0, max_value=1.0,
    value=0.5, step=0.05
)

df_filtre = df.copy()
if spe_filtre != "Toutes":
    df_filtre = df_filtre[df_filtre["specialite_label"] == spe_filtre]

# ── PAGE 1 : Vue globale ──────────────────────────────────
if page == "Vue globale":
    st.title("🏥 Detection d'anomalies DCIR - Vue globale")
    st.markdown(
        "Simulation inspiree du DCIR/SIAM - "
        "Detection de comportements potentiellement suspects"
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total praticiens", len(df))
    with col2:
        nb_suspects = (df["score_anomalie"] >= seuil_score).sum()
        st.metric("Suspects detectes", nb_suspects,
                  delta=f"{nb_suspects/len(df)*100:.1f}%")
    with col3:
        st.metric("Precision modele", "81.7%")
    with col4:
        st.metric("Rappel modele", "77.8%")
    with col5:
        st.metric("F1-score", "79.7%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution des scores d'anomalie")
        fig = px.histogram(
            df_filtre, x="score_anomalie", nbins=30,
            color_discrete_sequence=["#2471A3"],
            labels={"score_anomalie": "Score d'anomalie",
                    "count": "Nombre de praticiens"},
        )
        fig.add_vline(x=seuil_score, line_dash="dash",
                      line_color="red",
                      annotation_text=f"Seuil : {seuil_score}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Suspects par specialite")
        suspects_spe = df_filtre[
            df_filtre["score_anomalie"] >= seuil_score
        ].groupby("specialite_label").size().reset_index(name="nb_suspects")
        fig2 = px.bar(
            suspects_spe.sort_values("nb_suspects", ascending=True),
            x="nb_suspects", y="specialite_label", orientation="h",
            color_discrete_sequence=["#E74C3C"],
            labels={"nb_suspects": "Nombre de suspects",
                    "specialite_label": "Specialite"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Performance par scenario d'anomalie")
    scenarios_data = {
        "Scenario": ["S1 Suractivite", "S2 Actes fantomes",
                     "S3 Incoherence", "S4 Tarifaire", "S5 Reseau"],
        "Vrais": [15, 10, 12, 15, 13],
        "Detectes": [14, 4, 12, 15, 6],
        "Taux (%)": [93, 40, 100, 100, 46]
    }
    df_scenarios = pd.DataFrame(scenarios_data)
    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.bar(df_scenarios, x="Scenario", y="Taux (%)",
                      color="Taux (%)", color_continuous_scale="RdYlGn",
                      range_color=[0, 100], text="Taux (%)")
        fig3.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        st.dataframe(df_scenarios, use_container_width=True, hide_index=True)

# ── PAGE 2 : Top suspects ─────────────────────────────────
elif page == "Top suspects":
    st.title("🚨 Top suspects")

    top = df_filtre.nlargest(50, "score_anomalie")[[
        "pranum_pra", "specialite_label", "secteur",
        "score_anomalie", "anomalie_predite",
        "actes_par_jour_moyen", "zscore_actes_jour",
        "taux_depassement_pct", "zscore_depassement",
        "part_incompatibles_pct", "concentration_pharmacie_pct",
        "SEVERITY"
    ]].copy()

    top["score_anomalie"] = top["score_anomalie"].round(3)
    top["zscore_actes_jour"] = top["zscore_actes_jour"].round(2)
    top["zscore_depassement"] = top["zscore_depassement"].round(2)

    st.dataframe(
        top.style.background_gradient(subset=["score_anomalie"], cmap="Reds"),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    st.subheader("Scatter : Score anomalie vs Actes/jour")
    fig4 = px.scatter(
        df_filtre, x="actes_par_jour_moyen", y="score_anomalie",
        color="specialite_label", size="score_anomalie",
        hover_data=["pranum_pra", "zscore_actes_jour"],
        labels={"actes_par_jour_moyen": "Actes par jour (moyenne)",
                "score_anomalie": "Score d'anomalie"}
    )
    fig4.add_hline(y=seuil_score, line_dash="dash",
                   line_color="red", annotation_text="Seuil suspect")
    st.plotly_chart(fig4, use_container_width=True)

# ── PAGE 3 : Analyse par scenario ─────────────────────────
elif page == "Analyse par scenario":
    st.title("🔍 Analyse par scenario")

    scenario_choisi = st.selectbox(
        "Choisir un scenario",
        ["S1 Suractivite", "S2 Actes fantomes",
         "S3 Incoherence acte/specialite",
         "S4 Anomalie tarifaire", "S5 Reseau pharmacies"]
    )

    col_map = {
        "S1 Suractivite": ("zscore_actes_jour", "Z-score actes/jour", "S1_SURACTIVITE"),
        "S2 Actes fantomes": ("nb_actes_post_mortem", "Actes post-mortem", "S2_ACTE_FANTOME"),
        "S3 Incoherence acte/specialite": ("part_incompatibles_pct", "% actes incompatibles", "S3_INCOHERENCE_SPECIALITE"),
        "S4 Anomalie tarifaire": ("zscore_depassement", "Z-score depassement", "S4_ANOMALIE_TARIFAIRE"),
        "S5 Reseau pharmacies": ("concentration_pharmacie_pct", "Concentration pharmacie (%)", "S5_RESEAU_ATYPIQUE"),
    }

    feature, label, col_gt = col_map[scenario_choisi]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Distribution : {label}")
        fig5 = px.histogram(
            df_filtre, x=feature, color=col_gt, nbins=30,
            barmode="overlay",
            color_discrete_map={0: "#2ECC71", 1: "#E74C3C"},
            labels={feature: label, col_gt: "Anomalie injectee",
                    "count": "Nb praticiens"}
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.subheader("Top 10 sur cette feature")
        top10 = df_filtre.nlargest(10, feature)[[
            "pranum_pra", "specialite_label",
            feature, "score_anomalie", col_gt
        ]]
        st.dataframe(top10, use_container_width=True, hide_index=True)

# ── PAGE 4 : Fiche praticien ──────────────────────────────
elif page == "Fiche praticien":
    st.title("👤 Fiche praticien")

    praticien_choisi = st.selectbox(
        "Choisir un praticien",
        df_filtre.sort_values("score_anomalie", ascending=False)["pranum_pra"].tolist()
    )

    pra = df_filtre[df_filtre["pranum_pra"] == praticien_choisi].iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Specialite", pra["specialite_label"])
        st.metric("Secteur", pra["secteur"])
    with col2:
        score_color = "🔴" if pra["score_anomalie"] > 0.7 else \
                      "🟠" if pra["score_anomalie"] > 0.5 else "🟢"
        st.metric("Score anomalie", f"{score_color} {pra['score_anomalie']:.3f}")
        st.metric("Anomalie detectee",
                  "OUI" if pra["anomalie_predite"] == 1 else "NON")
    with col3:
        st.metric("Scenario reel", pra.get("scenario_principal", "N/A"))
        st.metric("Severite", pra["SEVERITY"])

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Indicateurs d'activite")
        data_act = {
            "Indicateur": ["Actes/jour moyen", "Actes/jour max",
                           "Z-score actes", "Part weekend (%)"],
            "Valeur": [pra["actes_par_jour_moyen"], pra["actes_par_jour_max"],
                       pra["zscore_actes_jour"], pra["part_actes_weekend_pct"]]
        }
        st.dataframe(pd.DataFrame(data_act), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Indicateurs financiers")
        data_fin = {
            "Indicateur": ["Montant moyen acte (euro)", "Taux depassement (%)",
                           "Z-score depassement", "Evolution montant (%)"],
            "Valeur": [pra["montant_moyen_acte"], pra["taux_depassement_pct"],
                       pra["zscore_depassement"], pra["evolution_montant_pct"]]
        }
        st.dataframe(pd.DataFrame(data_fin), use_container_width=True, hide_index=True)

    st.subheader("Indicateurs de coherence et relations")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("% actes incompatibles",
                  f"{pra['part_incompatibles_pct']:.1f}%")
    with col2:
        st.metric("Concentration pharmacie",
                  f"{pra['concentration_pharmacie_pct']:.1f}%")
    with col3:
        st.metric("Nb pharmacies distinctes",
                  int(pra["nb_pharmacies_distinctes"]))

    # ── NOUVEAU : Resume interprete ───────────────────────
    st.markdown("---")
    st.subheader("📋 Resume interprete - Comparaison aux normes")
    st.caption("Chaque indicateur est compare aux valeurs normales attendues pour cette specialite.")

    spe = pra["specialite_label"]
    n = NORMES.get(spe, NORMES["Specialistes"])

    resume = {
        "Indicateur": [
            "Actes/jour moyen",
            "Actes/jour max",
            "Part weekend (%)",
            "Montant moyen acte (euro)",
            "Taux depassement (%)",
            "% actes incompatibles",
            "Concentration pharmacie (%)",
            "Nb pharmacies distinctes",
        ],
        "Valeur observee": [
            round(float(pra["actes_par_jour_moyen"]), 2),
            int(pra["actes_par_jour_max"]),
            round(float(pra["part_actes_weekend_pct"]), 2),
            round(float(pra["montant_moyen_acte"]), 2),
            round(float(pra["taux_depassement_pct"]), 2),
            round(float(pra["part_incompatibles_pct"]), 2),
            round(float(pra["concentration_pharmacie_pct"]), 2),
            int(pra["nb_pharmacies_distinctes"]),
        ],
        "Plage normale attendue": [
            f"{n['actes_j'][0]} - {n['actes_j'][1]} actes/j",
            f"< {n['actes_j'][1]*2} actes/j",
            "5% - 15%",
            f"{n['montant'][0]} - {n['montant'][1]} euros",
            f"{n['dep_pct'][0]}% - {n['dep_pct'][1]}%",
            "0% - 5%",
            f"{n['pharma_conc'][0]}% - {n['pharma_conc'][1]}%",
            f"{n['pharma_nb'][0]} - {n['pharma_nb'][1]} pharmacies",
        ],
        "Signal": [
            signal(pra["actes_par_jour_moyen"],   n["actes_j"][0],      n["actes_j"][1]),
            signal(pra["actes_par_jour_max"],      0,                    n["actes_j"][1]*2),
            signal(pra["part_actes_weekend_pct"], 5,                    15),
            signal(pra["montant_moyen_acte"],     n["montant"][0],      n["montant"][1]),
            signal(pra["taux_depassement_pct"],   n["dep_pct"][0],      n["dep_pct"][1]),
            signal(pra["part_incompatibles_pct"], 0,                    5),
            signal(pra["concentration_pharmacie_pct"], n["pharma_conc"][0], n["pharma_conc"][1]),
            signal(pra["nb_pharmacies_distinctes"], n["pharma_nb"][0],  n["pharma_nb"][1]),
        ],
    }

    df_resume = pd.DataFrame(resume)
    st.dataframe(df_resume, use_container_width=True, hide_index=True)

    # Comptage des signaux
    nb_rouge = sum(1 for s in resume["Signal"] if "ANORMAL" in s)
    nb_vert  = sum(1 for s in resume["Signal"] if "NORMAL" in s)
    nb_na    = sum(1 for s in resume["Signal"] if "N/A" in s)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Signaux anormaux 🔴", nb_rouge)
    with col2:
        st.metric("Signaux normaux 🟢", nb_vert)
    with col3:
        st.metric("Non applicables ⚪", nb_na)

    # ── Radar chart ───────────────────────────────────────
    st.subheader("Profil de risque")
    categories = ["Suractivite", "Actes fantomes",
                  "Incoherence", "Tarifaire", "Reseau"]

    def norm(val, vmax):
        return min(abs(val) / vmax, 1.0) if vmax > 0 else 0

    valeurs = [
        norm(pra["zscore_actes_jour"], 6),
        norm(pra["nb_actes_post_mortem"], 200),
        norm(pra["part_incompatibles_pct"], 20),
        norm(pra["zscore_depassement"], 10),
        norm(pra["concentration_pharmacie_pct"], 100),
    ]
    valeurs += [valeurs[0]]
    categories += [categories[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=valeurs, theta=categories, fill="toself",
        line_color="#E74C3C", fillcolor="rgba(231, 76, 60, 0.3)"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)
