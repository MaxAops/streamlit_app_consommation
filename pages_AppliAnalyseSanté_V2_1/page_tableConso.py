import streamlit as st
import numpy as np
import os


from fonctions import workOnData
from fonctions import build_conso_tables

choix_annee=[2016,2017,2018,2019,2020,2021, 2022, 2023,2024,2025]



def tableConso():
    st.title("Tables consommations par familles")
    ID = st.sidebar.radio(
    "Sélectionnez l'identifiant à concidérer",
    ('id_bénéf', 'id_assuré'))
    # Créer des widgets pour permettre à l'utilisateur de choisir l'année et l'intervalle de mois
    annee = st.sidebar.selectbox("Année", choix_annee)
    mois_min, mois_max = st.sidebar.slider("Plage de mois", min_value=1, max_value=12, value=(1, 12))
    
    # Détecter si on est sur Streamlit Cloud
    ON_CLOUD = os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true"

    # Choisir le backend selon l'environnement
    backend = "matplotlib" if ON_CLOUD else "chrome"

    if st.button('table par familles actes'):
            cancel = st.button("Annuler")
            if not cancel:
                # Charger les données CSV à partir du fichier
                data1=workOnData.load_data(st.session_state["donnees"],annee,mois_min, mois_max)
                data2=workOnData.load_data(st.session_state["donnees"],annee-1,mois_min, mois_max)
                table1=build_conso_tables.TableConso(data1,st.session_state["repertoire_images"],ID,backend)
                table2=build_conso_tables.TableConso(data2,st.session_state["repertoire_images"],ID,backend)
                tableVs=build_conso_tables.table_N_vs_NMoins1(table1,table2,annee,st.session_state["repertoire_images"],backend)
                
                ecart_table=len(table1)+3

                build_conso_tables.ajouter_tableau_excel(f"{st.session_state["repertoire_images"]}table_conso.xlsx", table1.rename(columns={'Famille acte':str(annee)}).set_index(str(annee),drop=True), 'Consommation par famille', 0, 0)
                build_conso_tables.ajouter_tableau_excel(f"{st.session_state["repertoire_images"]}table_conso.xlsx", table2.rename(columns={'Famille acte':str(annee-1)}).set_index(str(annee-1),drop=True), 'Consommation par famille', ecart_table, 0)
                build_conso_tables.ajouter_tableau_excel(f"{st.session_state["repertoire_images"]}table_conso.xlsx", tableVs.set_index(tableVs.columns[0],drop=True), 'Consommation par famille', ecart_table*2, 0)
            else:
                st.write("L'exécution a été annulée.")
    
    st.title("Tables consommations par sous familles")

    columns = list(st.session_state["donnees"].columns)
    # Valeur par défaut souhaitée
    Mesure_def = "Sous famille"
    # Calcul de l'index correspondant
    index_mesure = columns.index(Mesure_def) if Mesure_def in columns else 0

    # Valeur par défaut souhaitée
    boucle_def = "Famille acte"
    # Calcul de l'index correspondant
    index_boucle = columns.index(boucle_def) if boucle_def in columns else 0

    Mesure = st.selectbox('Séléctionnez la variable d\'index de la future table', list(st.session_state["donnees"].columns),index_mesure)
                    
    boucle = st.selectbox('Séléctionnez la variable sur laquelle vous voulez boucler (ex : Famille acte):', list(st.session_state["donnees"].columns),index_boucle)
    
    on_comparaison = st.toggle("Comparer N et N-1?")

    if st.button('tables par sous familles actes'):
        cancel = st.button("Annuler")
        if not cancel: ################### Exectution : 2 cas de figure : - avec comparaison N/N-1 / - Sans comparaison
            if on_comparaison: ############# Comparaison N/N-1
                data1=workOnData.load_data(st.session_state["donnees"],annee,mois_min, mois_max)
                data2=workOnData.load_data(st.session_state["donnees"],annee-1,mois_min, mois_max)
                if data1 is not None and data2 is not None:
                    ecart_table=0
                    liste_famille_acte = list(set(np.concatenate((data1[boucle].unique(), data2[boucle].unique()))))
                    for V_boucle in liste_famille_acte:
                        try:
                            data_1=data1[data1[boucle]==V_boucle]
                            data_2=data2[data2[boucle]==V_boucle]
                            if data_1 is not None and data_2 is not None:
                                table_SF_1=build_conso_tables.TableConso_par_sous_familles(data_1,st.session_state["repertoire_images"],ID,Mesure,V_boucle)
                                table_SF_2=build_conso_tables.TableConso_par_sous_familles(data_2,st.session_state["repertoire_images"],ID,Mesure,V_boucle)
                                build_conso_tables.ajouter_tableau_excel(f"{st.session_state["repertoire_images"]}table_conso.xlsx", table_SF_1.rename(columns={table_SF_1.columns[0]:str(V_boucle)}).set_index(str(V_boucle),drop=True), f"Conso par Sf {str(annee)[-2:]} vs {str(annee-1)[-2:]}", ecart_table, 0)
                                build_conso_tables.ajouter_tableau_excel(f"{st.session_state["repertoire_images"]}table_conso.xlsx", table_SF_2.rename(columns={table_SF_2.columns[0]:str(V_boucle)}).set_index(str(V_boucle),drop=True), f"Conso par Sf {str(annee)[-2:]} vs {str(annee-1)[-2:]}", ecart_table,len(table_SF_1.columns)+3)
                                ecart_table=ecart_table+max(len(table_SF_1),len(table_SF_2))+3
                                build_conso_tables.comparaison_sf_n_n_1(table_SF_1,table_SF_2,st.session_state["repertoire_images"],annee,Mesure,V_boucle)

                        except:
                            st.write(f"Données manquantes pour la catégorie : {V_boucle}")
                else:
                    st.write("Pas de donnée")


            else: ############## Pas de comparaison seulement résultat N
                data=workOnData.load_data(st.session_state["donnees"],annee,mois_min, mois_max)
                if data is not None:
                    ecart_table=0
                    for V_boucle in data[boucle].unique():
                        data_=data[data[boucle]==V_boucle]
                        table_SF=build_conso_tables.TableConso_par_sous_familles(data_,st.session_state["repertoire_images"],ID,Mesure,V_boucle)
                        build_conso_tables.ajouter_tableau_excel(f"{st.session_state["repertoire_images"]}table_conso.xlsx", table_SF.rename(columns={table_SF.columns[0]:str(V_boucle)}).set_index(str(V_boucle),drop=True), f"Conso par S_famille {str(annee)[-2:]}", ecart_table, 0)
                        ecart_table=ecart_table+len(table_SF)+3
                        
                else:
                    st.write("Pas de donnée")
        else:
            st.write("L'exécution a été annulée.")