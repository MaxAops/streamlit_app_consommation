#import sys
import streamlit as st

#sys.path.append(r"C:\Users\maxime.genet\Desktop\T\Mission R&D\Application santé\fonctions")

from fonctions import workOnData
from fonctions import charts

choix_annee=[2016,2017,2018,2019,2020,2021, 2022, 2023,2024,2025]


def _100p100Santé():
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

    if "Sous famille" in list(st.session_state["donnees"].columns):
        select_var = st.selectbox('Séléctionnez la variable du 100% santé', list(st.session_state["donnees"].columns))
        select_sf = st.multiselect('Séléctionnez la sous famille contenant du 100% santé', list(st.session_state["donnees"]['Sous famille'].unique()))
        # Afficher la variable sélectionnée sous forme de liste
        
        #sf = [select_sf]
    else : 
         st.write("La colonne Sous famille n'est pas présente dans le jeux de données")

    # Initialiser `st.session_state` pour stocker les données et l'annulation
    if "cancel" not in st.session_state:
        st.session_state.cancel = False
    if "data" not in st.session_state:
        st.session_state.data = None

    # Extraction des données
    if st.button('Cliquez ici pour extraire les données'):
        st.session_state.cancel = False  # Réinitialiser l’annulation
        st.session_state.data = workOnData.load_data(st.session_state["donnees"], annees, mois_min, mois_max, Sous_famille=select_sf)

    # Bouton pour annuler l'exécution
    if st.button("Annuler"):
        st.session_state.cancel = True
        st.session_state.data = None  # Supprime les données extraites

    # Affichage des résultats si l'annulation n'est pas activée et que les données sont chargées
    if not st.session_state.cancel and st.session_state.data is not None:
        data = st.session_state.data  # Récupérer les données chargées
        st.table(data[select_var].value_counts())

        # Renommage des catégories
        data = workOnData.rename_cat(data, select_var)
        # Affichage du graphique mis à jour
    sf=data['Famille acte'].unique()[0]
    charts.Panier_plot(data, ID, select_var, sf, 
                        st.session_state["Qualité images"], st.session_state["repertoire_images"])
    charts.Panier_plot_ventilation(data, ID, select_var, sf, 
                        st.session_state["Qualité images"], st.session_state["repertoire_images"])
                
            