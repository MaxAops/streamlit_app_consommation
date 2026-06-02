#import sys
import streamlit as st

#sys.path.append(r"C:\Users\maxime.genet\Desktop\T\Mission R&D\Application santé\fonctions")

from fonctions import workOnData
from fonctions import charts
from fonctions import build_conso_tables

choix_annee=[2016,2017,2018,2019,2020,2021, 2022, 2023,2024,2025]

df = st.session_state.get("donnees")



def comparaison_survenances():
    df = st.session_state.get("donnees")

    if df is None:

        st.warning("⚠️ Aucune donnée chargée")
        return

        
    if 'cat_assure' not in df.columns:
        df['cat_assure']='actifs'

    choix_categorie = df["cat_assure"].unique()





    ID = st.sidebar.radio(
    "Sélectionnez l'identifiant à concidérer",
    ('id_beneficiaire', 'id_assuré'))
    # Créer des widgets pour permettre à l'utilisateur de choisir l'année et l'intervalle de mois
    unique_annees = choix_annee
    max_annee = max(unique_annees)

    all_annees_selected = st.sidebar.selectbox('Voulez-vous inclure uniquement des années spécifiques ? Si la réponse est oui, veuillez cocher la case ci-dessous, puis sélectionnez la ou les année(s) dans le nouveau champ.', ['Inclure toutes les années disponibles','Sélection manuelle'])
    if all_annees_selected == 'Sélection manuelle':

        default_annees = sorted([max_annee, max_annee - 1,max_annee - 2])
        annees = st.sidebar.multiselect("Sélectionnez et désélectionnez les années que vous souhaitez inclure dans l'analyse. Vous pouvez effacer la sélection actuelle en cliquant sur le bouton x correspondant sur la droite.", unique_annees, default = default_annees)
    else:
        annees=unique_annees
    mois_min, mois_max = st.sidebar.slider("Plage de mois", min_value=1, max_value=12, value=(1, 12))

    cat_assure_choose = st.sidebar.multiselect("selection categorie",choix_categorie, default = choix_categorie)

    if st.button('Cliquez ici pour exécuter'):
            cancel = st.button("Annuler")
            if not cancel:
                # Charger les données CSV à partir du fichier
                data=workOnData.load_data(st.session_state["donnees"],annees,mois_min, mois_max)
                build_conso_tables.table_evolution_kpis(data,cat_assure_choose,annees,ID,st.session_state["repertoire_images"])
                charts.proportion_cat_assure(data,cat_assure_choose,st.session_state["repertoire_images"],ID)
                charts.Evo_Cons_Moyenne(data,cat_assure_choose, st.session_state["Qualité images"],st.session_state["repertoire_images"],ID)
                charts.Evo_RC(data,cat_assure_choose,st.session_state["Qualité images"],st.session_state["repertoire_images"])
                charts.EVO_Montant(data,'RC',st.session_state["Qualité images"],st.session_state["repertoire_images"])
                charts.EVO_Consommateurs(data,cat_assure_choose,st.session_state["Qualité images"],st.session_state["repertoire_images"],ID)
                charts.EVO_Remboursement_moy(data,'RC',st.session_state["Qualité images"],st.session_state["repertoire_images"],ID)
                
            else:
                st.write("L'exécution a été annulée.")