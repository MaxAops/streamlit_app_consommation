#import sys
import streamlit as st

#sys.path.append(r"C:\Users\maxime.genet\Desktop\T\Mission R&D\Application santé\fonctions")

from fonctions import workOnData
from fonctions import charts

choix_annee=[2016,2017,2018,2019,2020,2021, 2022, 2023,2024,2025]


def etude_prix():
    # Créer des widgets pour permettre à l'utilisateur de choisir l'année et l'intervalle de mois
    unique_annees = choix_annee
    all_annees_selected = st.sidebar.selectbox('Voulez-vous inclure uniquement des années spécifiques ? Si la réponse est oui, veuillez cocher la case ci-dessous, puis sélectionnez la ou les année(s) dans le nouveau champ.', ['Inclure toutes les années disponibles','Sélection manuelle'])
    if all_annees_selected == 'Sélection manuelle':
        annees = st.sidebar.multiselect("Sélectionnez et désélectionnez les années que vous souhaitez inclure dans l'analyse. Vous pouvez effacer la sélection actuelle en cliquant sur le bouton x correspondant sur la droite.", unique_annees, default = unique_annees)
    else:
        annees=unique_annees
    mois_min, mois_max = st.sidebar.slider("Plage de mois", min_value=1, max_value=12, value=(1, 12))



    if "Sous famille" in list(st.session_state["donnees"].columns):
        col1, col2, col3 = st.columns(3)
        with col1:
            select_sf = st.multiselect('Séléctionnez la ou les sous famille(s)', list(st.session_state["donnees"]['Sous famille'].unique()))
            # Afficher la variable sélectionnée sous forme de liste
            if len(select_sf)>0:
                élément_titre=' - '.join(select_sf)
            else:
                élément_titre=select_sf[0]
            #sf = [select_sf]
        with col2:
            var=st.selectbox('Séléctionnez le type de montant : RC, Frais réels ...', ['RC','FR','R_SS','RàC'])
        with col3:
            orientation = st.radio(
            "Choisissez l'orientation du graphique",
            ["horizontal", "vertical"],
                )
    else : 
         st.write("La colonne Sous famille n'est pas présente dans le jeux de données")
    
    if "Majoration" in list(st.session_state["donnees"].columns): 
        
        on = st.toggle("Exclure les actes identifiés comme des majorations")
    else:
        on=None
    
    
    if st.button('Cliquez ici pour exécuter'):
        cancel = st.button("Annuler")
        if not cancel:
            # Charger les données CSV à partir du fichier
            data=workOnData.load_data(st.session_state["donnees"],annees,mois_min, mois_max, Sous_famille=select_sf)
            if on :
                data=data[data['Majoration'].isna()]
            if var=='RàC':
                data=workOnData.groupby_acte(data)
            if not data.empty:
                    charts.etude_composante_dépense(data,var,élément_titre,st.session_state["repertoire_images"], st.session_state["Qualité images"],orientation)                    
            else: 
                st.write("Pas de données pour cette sous famille")
        else:
            st.write("L'exécution a été annulée.")
                
            