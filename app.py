import streamlit as st

import sys
sys.path.append(r"C:\Users\maxime.genet\Desktop\T\Mission R&D\Application santé\pages_AppliAnalyseSanté_V2_1")

try:
    from page_tableConso import tableConso
    from page_chargement_donnees import charger_donnees
    from page_dispersion_couts_an import Dispersion
    from page_analyse_generale import Analyse_generale
    from page_comparaison_survenances import comparaison_survenances
    from page_cadencement_PSAP import cadencement_PSAP
    from page_100p100Santé import _100p100Santé
    from page_etude_sous_famille import etude_sous_famille
    from page_etude_prix import etude_prix

except ImportError as e:
    st.error(f"Erreur lors de l'importation des modules : {e}")
    st.stop()  # Arrête l'exécution de Streamlit si l'import échoue

# Configuration de la page principale
st.set_page_config(page_title="Application Streamlit", layout="wide")


# Vérification de l'état des données dans la session state
if "donnees" not in st.session_state:
    st.session_state["donnees"] = None

if "repertoire_images" not in st.session_state:
    st.session_state["repertoire_images"] = None

# Sidebar pour naviguer entre les pages
page = st.sidebar.selectbox("Navigation", ["Charger les données", "Tables consommations", "Dispersion des coûts", "Analyse générale",
                                           "Comparaison entre survenances","Etude sous famille","Etude prix","100% santé",'Cadencements & PSAP'])

if page == "Charger les données":
    st.title("Chargement des données")
    charger_donnees()  # Charge uniquement sur cette page
    if st.session_state["donnees"] is not None:
        st.write("Aperçu des données :")
        st.dataframe(st.session_state["donnees"].head())

elif page in ["Tables consommations", "Dispersion des coûts", "Analyse générale", "Comparaison entre survenances",
              "100% santé", "Etude sous famille","Etude prix", "Cadencements & PSAP"]:
    if st.session_state["donnees"] is None:
        st.warning("Veuillez d'abord charger un jeu de données sur la page 'Charger les données'.")
    else:
        st.title(page)
        # Appelle la fonction appropriée
        if page == "Tables consommations":
            tableConso()
        elif page == "Dispersion des coûts":
            Dispersion()
        elif page == "Analyse générale":
            Analyse_generale()
        elif page == "Comparaison entre survenances":
            comparaison_survenances()
        elif page == "100% santé":
            _100p100Santé()
        elif page == "Etude sous famille":
            etude_sous_famille()
        elif page == "Etude prix":
            etude_prix()
        elif page == "Cadencements & PSAP":
            cadencement_PSAP()