import streamlit as st
import pandas as pd
from pathlib import Path
import os
import shutil
import zipfile
import io


from fonctions.workOnData import pad_column_with_zeros
from fonctions.workOnData import load_csv



def init_export_dir():
    # Définir le répertoire d'export par défaut
    export_dir = Path(__file__).resolve().parents[1] / "exports" / "images"

    # Si le dossier existe déjà, supprimer son contenu
    if export_dir.exists():
        shutil.rmtree(export_dir)

    # Recréer le dossier vide
    export_dir.mkdir(parents=True, exist_ok=True)


    return export_dir



def charger_donnees():
    fichier = st.file_uploader("Charger un fichier CSV", type=["csv"])
    if fichier is not None:
        try:
            df = load_csv(fichier)
            df["id_bénéf"] = pad_column_with_zeros(df["id_bénéf"])
            df["id_assuré"] = pad_column_with_zeros(df["id_assuré"])

            st.session_state["donnees"] = df
            st.success("Données chargées avec succès !")

            export_dir = init_export_dir()
            st.session_state["repertoire_images"] = str(export_dir)

        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")

    # Ajouter une option pour définir le répertoire de sortie des images

    


    qualité_image = st.number_input("Entrez la résolution des images souhaitées (entre 0 et 200) :", value=100)
    
    if qualité_image:
        st.session_state["Qualité images"] = qualité_image


    # -------------------------------
    # Bouton pour télécharger toutes les images en ZIP
    # -------------------------------
    if st.button("Télécharger toutes les images"):
        images = [
            str(f)
            for f in Path(st.session_state["repertoire_images"]).glob("*")
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]

        if not images:
            st.warning("Aucune image à télécharger dans le répertoire export.")
            st.write(f"Répertoire actuel : {st.session_state['repertoire_images']}")
        else:
            st.write(f"Préparation du téléchargement de {len(images)} images...")
            # Créer un ZIP en mémoire
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zipf:
                for img_path in images:
                    zipf.write(img_path, arcname=Path(img_path).name)
            zip_buffer.seek(0)

            # Proposer le téléchargement
            st.download_button(
                label="Télécharger le ZIP",
                data=zip_buffer,
                file_name="images.zip",
                mime="application/zip"
            )

    
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
        