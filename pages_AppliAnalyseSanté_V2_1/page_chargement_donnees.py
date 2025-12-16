import streamlit as st
import pandas as pd
import os


from fonctions.workOnData import pad_column_with_zeros
from fonctions.workOnData import load_csv



def charger_donnees():
    fichier = st.file_uploader("Charger un fichier CSV", type=["csv"])
    if fichier is not None:
        try:
            df = load_csv(fichier)
            df["id_bénéf"] = pad_column_with_zeros(df["id_bénéf"])
            df["id_assuré"] = pad_column_with_zeros(df["id_assuré"])

            st.session_state["donnees"] = df
            st.success("Données chargées avec succès !")

        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")

    # Ajouter une option pour définir le répertoire de sortie des images

    col1, col2 = st.columns(2)
    with col1:
        repertoire = st.text_input("Entrez le chemin vers le répertoire de sortie pour les images :", value="")
    with col2:
        if st.session_state["repertoire_images"] is not None:
            repertoire=st.session_state["repertoire_images"]
            st.success(f"emplacement de stockage des images : {repertoire}")
    qualité_image = st.number_input("Entrez la résolution des images souhaitées (entre 0 et 200) :", value=100)
    if repertoire:
        if os.path.isdir(repertoire):
            st.session_state["repertoire_images"] = repertoire
        else:
            st.error("Le chemin fourni n'est pas un répertoire valide.")
    if qualité_image:
        st.session_state["Qualité images"] = qualité_image
    
    if st.sidebar.button("Supprimer données en mémoire"):
        # Vérifier que les données existent avant de les supprimer
        if "donnees" in st.session_state:
            del st.session_state["donnees"]
        if "repertoire_images" in st.session_state:
            del st.session_state["repertoire_images"]
        if "Qualité images" in st.session_state:
            del st.session_state["Qualité images"]

        # Afficher un message de confirmation
        st.success("Les données ont été supprimées de la mémoire.")
        