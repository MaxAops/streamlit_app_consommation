import io
import streamlit as st
import numpy as np
import pandas as pd
from fonctions import workOnData

from fonctions.etude_survenance import (
    depenses_santé_famille_acte_type_beneficiaire,
    export_excel,normaliser_type_beneficiaire
)

CHOIX_ANNEE = list(range(2016, 2026))

def etude_survenance():

    st.write("test")

    if st.session_state.get("donnees") is None:
        st.warning("Veuillez d'abord charger un jeu de données.")
        return


    # ── Sidebar ───────────────────────────────────────────────────────────────
    annees_dispo = CHOIX_ANNEE

    an2 = st.sidebar.selectbox("Survenance N",   annees_dispo, index=0)
    an1_options = [a for a in annees_dispo if a < an2]
    if not an1_options:
        st.warning("Pas de survenance N-1 disponible.")
        return
    an1 = st.sidebar.selectbox("Survenance N-1", an1_options, index=0)
    mois_min, mois_max = st.sidebar.slider("Plage de mois", min_value=1, max_value=12, value=(1, 12))


    # ── Préparation données ───────────────────────────────────────────────────
    
    # ── Préparation données ───────────────────────────────────────────────────
    df_an1 = workOnData.load_data(st.session_state["donnees"], an1, mois_min, mois_max)
    df_an2 = workOnData.load_data(st.session_state["donnees"], an2, mois_min, mois_max)
    
    df_filtre = pd.concat([df_an1, df_an2], ignore_index=True)
    if "annee_survenance" in df_filtre.columns:
        df_filtre = df_filtre.drop(columns=["annee_survenance"])

    df_filtre = df_filtre.rename(columns={"annee_soins": "annee_survenance"})
    df_filtre = normaliser_type_beneficiaire(df_filtre, col_source="type_beneficiaire")



    # Vérifier colonnes nécessaires
    cols_required = ["annee_survenance", "type_beneficiaire",
                     "famille_acte_aops", "RC", "id_beneficiaire"]
    manquantes = [c for c in cols_required if c not in df_filtre.columns]
    if manquantes:
        st.error(f"Colonnes manquantes dans les données : {', '.join(manquantes)}")
        return
    
    
    # ── Calcul ───────────────────────────────────────────────────────────────
    if st.button("▶ Générer le tableau", use_container_width=True):
        try:
            with st.spinner("Calcul en cours..."):
                table = depenses_santé_famille_acte_type_beneficiaire(df_filtre)
                table = table.replace([np.inf, -np.inf], np.nan)

            st.success(f"Tableau généré pour {an1} et {an2}")

            # ── Aperçu Streamlit ──────────────────────────────────────────────
            st.dataframe(table, use_container_width=True)

            # ── Export Excel en mémoire ───────────────────────────────────────
            buf = io.BytesIO()
            export_excel(table, buf)
            buf.seek(0)

            st.download_button(
                label="⬇ Télécharger le fichier Excel",
                data=buf,
                file_name=f"etude_survenance_{an1}_{an2}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        except AssertionError as e:
            st.error(f"Erreur : {e}")
        except Exception as e:
            st.error(f"Erreur inattendue : {e}")