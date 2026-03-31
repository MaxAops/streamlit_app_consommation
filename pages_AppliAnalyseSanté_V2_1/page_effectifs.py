import streamlit as st
import pandas as pd
from pathlib import Path

from fonctions.charts_effectifs import (
    kpis_effectifs,
    evolution_effectifs,
    repartition_categories,
    pyramide_ages,
    taux_consommation,
)

CHOIX_ANNEE = list(range(2016, 2026))


def _load_effectifs(fichier):
    """Charge et prépare le CSV effectifs."""
    try:
        df = pd.read_csv(fichier, sep=";", low_memory=False)
        # Conversion des dates principales
        for col in ["date_affiliation", "date_sortie_beneficiaire",
                    "date_entree_cat", "date_sortie_cat",
                    "date_naissance_beneficiaire"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return None


def _filtrer_effectifs(df, annees, mois_min, mois_max):
    """
    Filtre les bénéficiaires actifs sur la période sélectionnée.
    Un bénéficiaire est actif si :
      - date_affiliation <= fin de période
      - date_sortie_beneficiaire est NaT ou >= début de période
    """
    df = df.copy()
    debut = pd.Timestamp(year=min(annees), month=mois_min, day=1)
    fin   = pd.Timestamp(year=max(annees), month=mois_max, day=28) + pd.offsets.MonthEnd(0)

    mask = (
        (df["date_affiliation"] <= fin) &
        (df["date_sortie_beneficiaire"].isna() | (df["date_sortie_beneficiaire"] >= debut))
    )
    return df[mask]


def page_effectifs():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filtres Effectifs**")

    all_annees = st.sidebar.selectbox(
        "Périmètre années",
        ["Toutes les années disponibles", "Sélection manuelle"])

    if all_annees == "Sélection manuelle":
        annees = st.sidebar.multiselect(
            "Années", CHOIX_ANNEE, default=[max(CHOIX_ANNEE)])
    else:
        annees = CHOIX_ANNEE

    mois_min, mois_max = st.sidebar.slider(
        "Plage de mois", min_value=1, max_value=12, value=(1, 12))

    # ── Chargement fichier effectifs ──────────────────────────────────────────
    st.markdown("### 📂 Fichier effectifs")

    # Garder le fichier en session pour ne pas recharger à chaque interaction
    if "effectifs" not in st.session_state:
        st.session_state["effectifs"] = None

    fichier = st.file_uploader(
        "Charger le fichier effectifs (CSV séparé par ;)",
        type=["csv"], key="upload_effectifs")

    if fichier is not None:
        df_eff = _load_effectifs(fichier)
        if df_eff is not None:
            st.session_state["effectifs"] = df_eff
            st.success(f"Fichier chargé · {len(df_eff):,} lignes · "
                       f"{df_eff['id_beneficiaire'].nunique():,} bénéficiaires uniques")

    if st.session_state["effectifs"] is None:
        st.info("Chargez un fichier effectifs pour afficher les analyses.")
        return

    df_eff = st.session_state["effectifs"]

    # Filtrage temporel
    if "date_affiliation" in df_eff.columns:
        df_filtre = _filtrer_effectifs(df_eff, annees, mois_min, mois_max)
    else:
        df_filtre = df_eff.copy()
        st.warning("Colonne date_affiliation introuvable — aucun filtre temporel appliqué.")

    # Prestations (optionnel, pour taux de consommation)
    df_presta = st.session_state.get("donnees")

    # Filtrer les prestations sur la même période
    if df_presta is not None and "annee_soins" in df_presta.columns:
        df_presta_filtre = df_presta[df_presta["annee_soins"].isin(annees)]
        df_presta_filtre = df_presta_filtre[
            df_presta_filtre["mois_soins"].between(mois_min, mois_max)]
    else:
        df_presta_filtre = None

    rep = st.session_state.get("repertoire_images", ".")
    dpi = st.session_state.get("Qualité images", 100)

    st.markdown("---")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kpis_effectifs(df_filtre, df_presta_filtre)

    st.markdown("<div style='margin:1rem 0'></div>", unsafe_allow_html=True)

    # ── Bouton exécution ──────────────────────────────────────────────────────
    if "eff_cancel" not in st.session_state:
        st.session_state["eff_cancel"] = False

    col_run, col_cancel = st.columns([2, 1])
    with col_run:
        run = st.button("▶ Générer les graphiques effectifs",
                        use_container_width=True)
    with col_cancel:
        if st.button("✖ Annuler", use_container_width=True):
            st.session_state["eff_cancel"] = True

    if run:
        st.session_state["eff_cancel"] = False

    if not st.session_state["eff_cancel"] and run:

        # 1. Évolution mensuelle
        with st.spinner("Évolution mensuelle..."):
            evolution_effectifs(df_filtre, dpi, rep)

        # 2. Répartition catégories
        with st.spinner("Répartition par catégorie..."):
            repartition_categories(df_filtre, dpi, rep)

        # 3. Pyramide des âges
        with st.spinner("Pyramide des âges..."):
            pyramide_ages(df_filtre, dpi, rep)

        # 4. Taux de consommation (seulement si prestations disponibles)
        if df_presta_filtre is not None:
            with st.spinner("Taux de consommation..."):
                taux_consommation(df_filtre, df_presta_filtre, dpi, rep)
        else:
            st.info("Chargez un fichier de prestations sur la page d'accueil "
                    "pour afficher le taux de consommation.")
