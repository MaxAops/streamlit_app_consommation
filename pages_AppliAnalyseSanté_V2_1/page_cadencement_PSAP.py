import pandas as pd
import streamlit as st

from fonctions import workOnData

try:
    import dataframe_image as dfi # Librairie très instable problèmes récurrents
except:
    print("erreur de la librairie dfi")

def cadencement_PSAP():

    with st.expander("Cadencement & PSAP"):

        df=st.session_state["donnees"]

        col1, col2,col3 = st.columns(3)

        with col1:
            on  = st.toggle("Prestations cumulées")
        with col2:
            SurvenanceMin = st.text_input("Survenance min", df['annee_soins'].min())
            SurvenanceMax = st.text_input("Survenance max", df['annee_soins'].max())
        with col3:
            date_PSAP=st.text_input("Mois des dernière données comptable", df[df['annee_paiement']==df['annee_paiement'].max()]['mois_paiement'].max())

        df_table_cadencement = df[(df['annee_soins'] >= int(SurvenanceMin)) &
        (df['annee_soins'] <= int(SurvenanceMax))]

        pivot = pd.pivot_table(df_table_cadencement, 
                    values='RC', 
                    index=['annee_paiement', 'mois_paiement'], 
                    columns='annee_soins', 
                    aggfunc='sum', 
                    fill_value=0)
            
                
        if on:
            st.table(pivot.cumsum().applymap(lambda x: '{:,.0f}'.format(x).replace(',', ' ')))
        elif not on:
            st.table(pivot.applymap(lambda x: '{:,.0f}'.format(x).replace(',', ' ')))
        
        if df['annee_paiement'].nunique()>2:
            st.write(f"PSAP au mois numéro {date_PSAP}")
            st.table(workOnData.PSAP(pivot.cumsum(),date_PSAP))
        else:
            st.markdown("Pas assez de recule comptable pour calculer les taux de PSAP")

    with st.expander("Rendu trimestriel"):
        col4, col5,col6 = st.columns(3)
        with col4:
            comptablemin = st.text_input("Année comptable min", df['annee_paiement'].min())
            comptablemax = st.text_input("Année comptable/survenance max", df['annee_paiement'].max())

            try:
                comptablemin=int(comptablemin)
                comptablemax=int(comptablemax)
            except:
                st.markdown("Erreur de format pour les année(s) comptable")

        with col5:
            Trimestres_Surv = st.multiselect("Séléctionner le/les trimestre(s) de survenance(s) souhaité(s)",
            [1,2,3,4],default=[1,2,3,4])

            Trimestres_compt = st.multiselect("Séléctionner le/les trimestre(s) comptable(s) souhaité(s)",
            [1,2,3,4],default=[1,2,3,4])
            
        with col6:
            Identifiant_personne=st.text_input("Variable identifiant personne", 'id_bénéf')
            if 'assureur' in df.columns:
                Assureur = st.multiselect("Séléctionner le/les assureur(s) souhaité(s)",
                df['assureur'].unique(),default=df['assureur'].unique())
            else:
                Assureur=None

        st.header("Table comptable")
        table_trimestriel=workOnData.Evolution_trimestre(df,int(comptablemin),int(comptablemax),Trimestres_compt,Assureur)

        st.table(table_trimestriel)

        if st.button("Télécharger la table"):
            dfi.export(table_trimestriel,st.session_state["repertoire_images"]+"tablecomptable.png")
        