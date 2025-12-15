#import sys
import streamlit as st

#sys.path.append(r"C:\Users\maxime.genet\Desktop\T\Mission R&D\Application santé\fonctions")

from fonctions import workOnData
from fonctions import charts

choix_annee=[2016,2017,2018,2019,2020,2021, 2022, 2023,2024,2025]


def comparaison_survenances():
    ID = st.sidebar.radio(
    "Sélectionnez l'identifiant à concidérer",
    ('id_bénéf', 'id_assuré'))
    # Créer des widgets pour permettre à l'utilisateur de choisir l'année et l'intervalle de mois
    unique_annees = choix_annee
    all_annees_selected = st.sidebar.selectbox('Voulez-vous inclure uniquement des années spécifiques ? Si la réponse est oui, veuillez cocher la case ci-dessous, puis sélectionnez la ou les année(s) dans le nouveau champ.', ['Inclure toutes les années disponibles','Sélection manuelle'])
    if all_annees_selected == 'Sélection manuelle':
        annees = st.sidebar.multiselect("Sélectionnez et désélectionnez les années que vous souhaitez inclure dans l'analyse. Vous pouvez effacer la sélection actuelle en cliquant sur le bouton x correspondant sur la droite.", unique_annees, default = unique_annees)
    else:
        annees=unique_annees
    mois_min, mois_max = st.sidebar.slider("Plage de mois", min_value=1, max_value=12, value=(1, 12))

    if st.button('Cliquez ici pour exécuter'):
            cancel = st.button("Annuler")
            if not cancel:
                # Charger les données CSV à partir du fichier
                data=workOnData.load_data(st.session_state["donnees"],annees,mois_min, mois_max)
                charts.Evo_Cons_Moyenne(data, st.session_state["Qualité images"],st.session_state["repertoire_images"],ID)
                charts.Evo_RC(data,st.session_state["Qualité images"],st.session_state["repertoire_images"])
                charts.EVO_Montant(data,'RC',st.session_state["Qualité images"],st.session_state["repertoire_images"])
                charts.EVO_Consommateurs(data,st.session_state["Qualité images"],st.session_state["repertoire_images"],ID)
                charts.EVO_Remboursement_moy(data,'RC',st.session_state["Qualité images"],st.session_state["repertoire_images"],ID)
            else:
                st.write("L'exécution a été annulée.")