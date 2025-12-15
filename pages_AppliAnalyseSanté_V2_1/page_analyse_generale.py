#import sys
import streamlit as st
#sys.path.append(r"C:\Users\maxime.genet\Desktop\T\Mission R&D\Application santé\fonctions")


from fonctions import workOnData
from fonctions import charts


choix_annee=[2016,2017,2018,2019,2020,2021, 2022, 2023,2024,2025]


def Analyse_generale():
    ID = st.sidebar.radio(
    "Sélectionnez l'identifiant à concidérer",
    ('id_bénéf', 'id_assuré'))
    # Créer des widgets pour permettre à l'utilisateur de choisir l'année et l'intervalle de mois
    annee = st.sidebar.selectbox("Année", choix_annee)
    mois_min, mois_max = st.sidebar.slider("Plage de mois", min_value=1, max_value=12, value=(1, 12))
    familles_actes= st.sidebar.multiselect(
    "Familles actes à inclure dans l'analyse",
    ['Hospitalisation',
'Consultations et visites',
'Soins courants',
'Pharmacie',
'Dentaire',
'Optique',
'Divers'],
    ['Hospitalisation',
'Consultations et visites',
'Soins courants',
'Pharmacie',
'Dentaire',
'Optique'],
)

    if st.button('Cliquez ici pour exécuter'):
            cancel = st.button("Annuler")
            if not cancel:
                # Charger les données CSV à partir du fichier
                data=workOnData.load_data(st.session_state["donnees"],annee,mois_min, mois_max,familles_actes)
                charts.PlotVentilationCouts(data, annee,st.session_state["Qualité images"],st.session_state["repertoire_images"],ID)
                charts.distributionFamilleActes(data,annee,st.session_state["repertoire_images"],st.session_state["Qualité images"])
            else:
                st.write("L'exécution a été annulée.")