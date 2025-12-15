import textwrap
import pandas as pd
import numpy as np
import streamlit as st

sorted_Famille={'Hospitalisation':1,
'Consultations et visites':2,
'Soins courants':3,
'Pharmacie':4,
'Dentaire':5,
'Optique':6,
'Divers':7}



# Fonction pour charger les données
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)



def Famille_acte_sorted(df):
    Famille_acte_sorted = []
    for i in list(sorted_Famille.keys()):
        if i in df['Famille acte'].unique():
            Famille_acte_sorted.append(i)
    return Famille_acte_sorted

def formatM(NUM):
    return('{:,.0f}'.format(NUM).replace(',', ' '))

def GetTypeBénéf(ID):
    a=''
    if 'assuré' in ID:
        a='assurés'
    elif 'bénéf' in ID:
        a='bénéficiaires'
    return a

@st.cache_data
def load_data(init_data,annee,mois_min, mois_max,famille_actes=None, Sous_famille=None):
    if not isinstance(annee, list):
        annee=[annee]
    data = init_data.copy()
    if famille_actes is not None:
        data_filtre = data[(data['RC']!=0) &
                                        (data["annee_soins"].isin(annee)) &
                                        (data["annee_paiement"]==data["annee_soins"]) &
                                        (data["mois_soins"].between(mois_min, mois_max)) &
                                        (data["mois_paiement"].between(mois_min, mois_max)) &
                                    (data["annee_soins"] == data["annee_paiement"]) & (data['Famille acte'].isin(famille_actes))]
    elif Sous_famille is not None:
        data_filtre = data[(data['RC']!=0) &
                                        (data["annee_soins"].isin(annee)) &
                                        (data["annee_paiement"]==data["annee_soins"]) &
                                        (data["mois_soins"].between(mois_min, mois_max)) &
                                        (data["mois_paiement"].between(mois_min, mois_max)) &
                                    (data["annee_soins"] == data["annee_paiement"]) & (data['Sous famille'].isin(Sous_famille))]
    
    else:
        data_filtre = data[(data['RC']!=0) &
                                        (data["annee_soins"].isin(annee)) &
                                        (data["annee_paiement"]==data["annee_soins"]) &
                                        (data["mois_soins"].between(mois_min, mois_max)) &
                                        (data["mois_paiement"].between(mois_min, mois_max)) &
                                    (data["annee_soins"] == data["annee_paiement"])]
    return data_filtre


def pad_column_with_zeros(series): # correction pour identifiants
    # Convertir la série en chaînes de caractères
    series = series.astype(str)

    # Trouver la longueur maximale de la chaîne dans la colonne
    max_length = series.apply(len).max()

    # Fonction pour compléter une chaîne avec des zéros si nécessaire
    def pad_string(s):
        # Si la chaîne finit par '.0', on la retire et on complète avec des zéros
        if s.endswith('.0'):
            s = s[:-2]  # Retirer '.0'
        return s.zfill(max_length)

    # Appliquer la fonction à chaque élément de la série
    return series.apply(pad_string)

def correction_cp(series): # correction pour codes postaux / Département
    # Convertir la série en chaînes de caractères
    series = series.astype(str)

    # Trouver la longueur maximale de la chaîne dans la colonne
    max_length = 5

    # Fonction pour compléter une chaîne avec des zéros si nécessaire
    def pad_string_cp(s):
        # Si la chaîne finit par '.0', on la retire et on complète avec des zéros
        if s.endswith('.0'):
            s = s[:-2]  # Retirer '.0'
        elif s.isalpha():
            return s
        elif len(s)<=3:
            return s.ljust(5, '0')
        elif len(s)>=4:
            return s.zfill(max_length) 

    # Appliquer la fonction à chaque élément de la série
    return series.apply(pad_string_cp)


def format_value(value):# utilisé dans le graphique : distributionFamilleActes (par bulles)
    if value >= 1000000:
        value_in_millions = value / 1000000
        formatted_value = f"{value_in_millions:.2f} M"
    else:
        formatted_value = formatM(value)

    return formatted_value

def format_string_with_linebreak(input_string, max_width=15): # utilisé dans le graphique : distributionFamilleActes (par bulles)
    wrapped_lines = textwrap.wrap(input_string, width=max_width)
    formatted_string = '\n'.join(wrapped_lines)

    return formatted_string


def calculate_max_diff(row, col, df,result_df):
    x = df.iloc[row, col]  # Valeur cible (x)
    column_values = df.iloc[:, col]  # Toutes les valeurs de la colonne
    if col==0:
        max_colx_x = (column_values.max() - x)  # max(colonne(x))- x
    else:
        max_colx_x = (column_values.max()*(1+np.mean(result_df.iloc[df.index.max()-col,:col])) - x) # max(colonne(x))*(1+mean(résidus)) - x
    return max_colx_x / x if x != 0 else None  # Éviter division par 0

def move_non_empty_to_index_0(table):
    # Pour chaque colonne, on trie les valeurs non nulles vers le début et remplit le reste par des zéros
    table = table.apply(lambda col: pd.Series([x for x in col if x != 0] + [0] * (len(col) - len([x for x in col if x != 0]))))
    return table

def PSAP(df,mois_PSAP):
    df_=df.xs(int(mois_PSAP),level='mois_paiement')
    df_=move_non_empty_to_index_0(df_)
    # Parcourir chaque case et appliquer l'opération
    result_df = pd.DataFrame(index=df_.index, columns=df_.columns)  # Nouveau DataFrame pour les résultats

    for col in range(df_.shape[1]):
        for row in range(df_.shape[0]):
            result_df.iloc[row, col] = calculate_max_diff(row, col, df_,result_df)

    return result_df.fillna(0)*100



def Evolution_trimestre(df,survenanceMin,survenanceMax,trimestres,assureur=None):
    if 'trimestre_paiement' not in df.columns:
        df['date_paiement'] = pd.to_datetime(df['date_paiement'])
        df['trimestre_paiement'] = df['date_paiement'].dt.quarter
    if assureur is None:
        table=pd.pivot_table(df[(df['annee_paiement']>=survenanceMin) & (df['annee_paiement']<=survenanceMax) &(df['trimestre_paiement'].isin(trimestres))],values='RC',index='trimestre_paiement',columns='annee_paiement',aggfunc='sum',fill_value=0)
    else:
         table=pd.pivot_table(df[(df['annee_paiement']>=survenanceMin) & (df['annee_paiement']<=survenanceMax) & (df['trimestre_paiement'].isin(trimestres)) & (df['assureur'].isin(assureur))],values='RC',index='trimestre_paiement',columns='annee_paiement',aggfunc='sum',fill_value=0)       
    table['Evolution N/N-1'] = table.apply(lambda row: str(round(((row[survenanceMax] / row[survenanceMax-1] -1) * 100),2))+'%' if row[survenanceMax] != 0 else "", axis=1)

    for col in table.columns:
        if col != 'Evolution N/N-1':  # Ne pas formater la colonne 'evolution'
            table[col] = table[col].apply(formatM)
                
    table.columns.name = 'Année comptable'
    table.index.name = None
    
    return colorize_index(table)

def colorize_index(df):
    # Appliquer les styles
    styled_df = df.style.set_table_styles(
        [{'selector': 'th:not(.index_name)','props': [('background-color', '#173A64'),('color', 'white'),('text-align', 'center'),('font-size', '14px')]},
            # Styliser l'index
            {'selector': 'th.row_heading', 
             'props': [('background-color', '#173A64'), ('color', 'white'), ('font-weight', 'bold'),('text-align', 'center')]},

            # Une ligne sur 2
            {'selector': 'tr:nth-child(even)','props': [('background-color', '#ffffff'),('color', 'black')]},
            {'selector': 'tr:nth-child(odd)','props': [('background-color', '#cccccc'),('color', 'black')]},

            # Séparation colonne () petit intervalles blancs verticales
            {'selector': 'td','props': [('color', 'black'),('border-left', '1px solid #FFFFFF'),('border-right', '1px solid #FFFFFF'),('text-align', 'right'),('font-size', '14px')]},
            #Horizontales
            {'selector': 'th','props': [('color', 'black'),('border', '1px solid #FFFFFF'),('font-size', '14px')]},
            
            # Styliser les en-têtes de colonnes
            {'selector': 'th.col_heading', 
             'props': [('background-color', '#173A64'), ('color', 'white'), ('font-weight', 'bold')]},

            # Styliser l'en-tête de la colonne 'Evolution'
            {'selector': f'th.col_heading.level0:nth-child({len(df.columns) + 1})', 
             'props': [('background-color', '#F56C26'), ('color', 'white'), ('font-weight', 'bold')]},
         {'selector': '.index_name', 
             'props': [('background-color', '#2C67AF'), ('color', 'white'), ('font-weight', 'bold')]}

        ]
    )

    return styled_df


def rename_cat(data, selected_var):
    if "new_names" not in st.session_state:
        st.session_state.new_names = {}

    categories = data[selected_var].dropna().unique()

    st.write("### Renommer plusieurs catégories")

    # Interface dynamique pour modifier les noms des catégories
    for category in categories:
        new_name = st.text_input(f"Renommer '{category}' :", 
                                 st.session_state.new_names.get(category, category))  # Pré-rempli avec la valeur existante
        if new_name != category:
            st.session_state.new_names[category] = new_name  # Enregistrer les changements

    # Bouton pour appliquer toutes les modifications en une seule fois
    if st.button("Appliquer les modifications"):
        if st.session_state.new_names or st.session_state.na:
            # Créer une copie modifiée du DataFrame pour éviter les modifications en place
            updated_data = data.copy()
            updated_data[selected_var] = updated_data[selected_var].replace(st.session_state.new_names)
            st.success("Les catégories ont été mises à jour avec succès !")
            return updated_data  # Retourner la version modifiée du DataFrame
        
        else:
            st.warning("Aucune modification détectée.")
    
    return data  # Retourne les données non modifiées si aucun changement

def groupby_acte(df):
    
    res_df=pd.pivot_table(df,values=['FR','R_SS','RC','nombre_acte'],index=['date_soins','annee_soins','code_acte','Sous famille','id_bénéf'],aggfunc='sum').reset_index()
    res_df['RàC']=res_df['FR']-res_df['R_SS']-res_df['RC']
    return res_df


def correction_dates_integrale(df_raw, col):
    
    for colonne in col:
    
        df = df_raw[[colonne]].copy()

        # Convertir toutes les valeurs en chaînes de caractères
        df[colonne] = df[colonne].astype(str)

        # Remplacer les valeurs spécifiques par des dates de référence
        df[colonne] = df[colonne].replace('2958465', '2099-12-31')

        # Identifier et convertir les valeurs numériques en dates
        numeric_mask = pd.to_numeric(df[colonne], errors='coerce').notna()
        df.loc[numeric_mask, colonne] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df.loc[numeric_mask, colonne].astype(int), 'D')
        df.loc[numeric_mask, colonne] = df.loc[numeric_mask, colonne].astype(str)

        # Supprimer les chaînes de temps
        df[colonne] = df[colonne].str.replace(" 00:00:00", "")

        # Remplacer '9999' et '2999' par '2099'
        df[colonne] = df[colonne].str.replace('9999', '2099')
        df[colonne] = df[colonne].str.replace('2999', '2099')

        # Identifier et convertir les dates au format avec tirets
        dash_mask = df[colonne].str.contains('-')
        slash_mask = df[colonne].str.contains('/')

        # Traiter les dates avec tirets
        if dash_mask.any():
            df.loc[dash_mask, colonne] = pd.to_datetime(df.loc[dash_mask, colonne])

        # Traiter les dates avec barres obliques
        if slash_mask.any():
            # Utiliser to_datetime avec coerce pour tenter les formats les plus courants
            
            date_formats = ['%d/%m/%y', '%Y-%m-%d', '%d/%m/%Y','%m/%d/%Y']

            # Convertir la colonne "dates" en objets datetime
            for format in date_formats:
                try:
                    df.loc[slash_mask, colonne] = pd.to_datetime(df.loc[slash_mask, colonne], format=format)
                    break
                except ValueError:
                    continue
        
        df_raw[colonne]=pd.to_datetime(df[colonne])
        df_raw[colonne] = df_raw[colonne].replace('2099-12-31', pd.NaT)
    
    return df_raw


def optimal_bins(data, min_bin_size=0.05, max_bin_size=0.2, initial_bins=20):
    """
    Crée des tranches optimales pour une variable continue avec des labels entiers.

    Args:
        data (array-like): Les montants à binner.
        min_bin_size (float): Proportion minimale de données par bin (ex: 0.05 = 5%).
        max_bin_size (float): Proportion maximale de données par bin.
        initial_bins (int): Nombre de bins de départ à tester.

    Returns:
        bins (list): Liste des bornes de tranches.
        labels (list): Libellés des tranches.
    """
    data = np.sort(np.array(data))
    n = len(data)

    # Étape 1 : Création de bins de base (quantiles)
    quantile_bins = np.unique(np.quantile(data, q=np.linspace(0, 1, initial_bins + 1)))

    # Étape 2 : Groupement des données par tranche
    bins = [quantile_bins[0]]
    current_bin_start = quantile_bins[0]

    for i in range(1, len(quantile_bins)):
        bin_end = quantile_bins[i]
        in_bin = (data >= current_bin_start) & (data <= bin_end)
        prop = np.sum(in_bin) / n

        if prop < min_bin_size:
            continue  # trop petit, fusionner avec le suivant
        elif prop > max_bin_size:
            # trop gros : sous-diviser cette tranche
            sub_data = data[in_bin]
            nb_sub_bins = max(2, int(prop / max_bin_size) + 1)
            sub_bins = np.unique(np.quantile(sub_data, q=np.linspace(0, 1, nb_sub_bins)))
            bins.extend(sub_bins[1:])
            current_bin_start = sub_bins[-1]
        else:
            bins.append(bin_end)
            current_bin_start = bin_end

    if bins[-1] < data[-1]:
        bins.append(data[-1])

    # Nettoyage et tri
    bins = sorted(set(bins))

    # Fonction d'arrondi intelligent
    def clean_number(x):
        return int(round(x)) if abs(round(x) - x) < 1e-6 else round(x, 2)

    # Création des labels
    labels = [
        f"[{clean_number(bins[i])}-{clean_number(bins[i+1])})"
        for i in range(len(bins) - 1)
    ]

    return bins, labels

"""
def calculer_bins_labels_equilibres(df, var, min_pct=0.05, max_pct=0.3, max_bins=20, multiple=5):

    # min_pct : proportion minimum de donnée dans une tranche
    # max_pct : proportion max de donnée dans une tranche
    # multiple : arrondi des bornes des tranche

    q1, q3 = np.percentile(df[var], [15, 85])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df_cleaned = df[(df[var] >= lower_bound) & (df[var] <= upper_bound)]
    n = len(df_cleaned)

    bin_width = 2 * iqr / np.cbrt(n)
    bin_width = max(multiple, round(bin_width / multiple) * multiple)

    # Pas de borne négative
    min_val = max(0, (df_cleaned[var].min() // multiple) * multiple)
    max_val = np.ceil(df_cleaned[var].max() / multiple) * multiple

    bins = list(range(int(min_val), int(max_val) + int(bin_width), int(bin_width)))

    if len(bins) > max_bins:
        range_val = max_val - min_val
        bin_width = max(multiple, round((range_val / max_bins) / multiple) * multiple)
        bins = list(range(int(min_val), int(max_val) + int(bin_width), int(bin_width)))

    # Histogramme initial
    counts, _ = np.histogram(df_cleaned[var], bins=bins)
    seuil_bas = min_pct * n
    seuil_haut = max_pct * n

    new_bins = [bins[0]]
    current_count = 0

    # Étape 1 – Fusionner les trop petites tranches
    for i in range(len(counts)):
        current_count += counts[i]
        if current_count >= seuil_bas:
            new_bins.append(bins[i + 1])
            current_count = 0
    if new_bins[-1] != bins[-1]:
        new_bins.append(bins[-1])

    # Étape 2 – Recompter et exploser les tranches trop pleines
    final_bins = [new_bins[0]]
    for i in range(len(new_bins) - 1):
        start = new_bins[i]
        end = new_bins[i + 1]
        mask = (df_cleaned[var] >= start) & (df_cleaned[var] < end)
        count = mask.sum()

        if count > seuil_haut:
            # On découpe en sous-bins de taille fixe
            n_sub = int(np.ceil(count / seuil_haut))
            sub_bin_width = max(multiple, round((end - start) / n_sub / multiple) * multiple)
            sub_bins = list(range(start, end, sub_bin_width))
            if sub_bins[-1] < end:
                sub_bins.append(end)
            final_bins += sub_bins[1:]  # éviter doublon
        else:
            final_bins.append(end)

    final_bins.append(np.inf)
    labels = [f"[{final_bins[i]}-{final_bins[i+1]}[" if final_bins[i+1] != np.inf else f"+{final_bins[i]}" for i in range(len(final_bins) - 1)]

    return final_bins, labels
"""

def calculer_bins_labels_equilibres(df, var, min_pct=0.05, max_pct=0.3, max_bins=20, multiple=5):
    q1, q3 = np.percentile(df[var], [15, 85])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df_cleaned = df[(df[var] >= lower_bound) & (df[var] <= upper_bound)].copy()
    n = len(df_cleaned)

    if iqr == 0 or n == 0:
        raise ValueError("Données insuffisantes ou IQR nul pour effectuer le binning.")

    bin_width = 2 * iqr / np.cbrt(n)
    bin_width = max(multiple, round(bin_width / multiple) * multiple)

    min_val = max(0, (df_cleaned[var].min() // multiple) * multiple)
    max_val = np.ceil(df_cleaned[var].max() / multiple) * multiple

    bins = list(range(int(min_val), int(max_val) + int(bin_width), int(bin_width)))

    if len(bins) > max_bins:
        range_val = max_val - min_val
        bin_width = max(multiple, round((range_val / max_bins) / multiple) * multiple)
        bins = list(range(int(min_val), int(max_val) + int(bin_width), int(bin_width)))

    counts, _ = np.histogram(df_cleaned[var], bins=bins)
    seuil_bas = min_pct * n
    seuil_haut = max_pct * n

    new_bins = [bins[0]]
    current_count = 0
    for i in range(len(counts)):
        current_count += counts[i]
        if current_count >= seuil_bas:
            new_bins.append(bins[i + 1])
            current_count = 0
    if new_bins[-1] != bins[-1]:
        new_bins.append(bins[-1])

    final_bins = [new_bins[0]]
    for i in range(len(new_bins) - 1):
        start = new_bins[i]
        end = new_bins[i + 1]
        mask = (df_cleaned[var] >= start) & (df_cleaned[var] < end)
        count = mask.sum()

        if count > seuil_haut:
            n_sub = int(np.ceil(count / seuil_haut))
            sub_bin_width = max(multiple, round((end - start) / n_sub / multiple) * multiple)
            sub_bins = list(range(start, end, sub_bin_width))
            if sub_bins[-1] < end:
                sub_bins.append(end)
            final_bins += sub_bins[1:]
        elif count > 0:
            final_bins.append(end)

    # Suppression des bins vides
    cleaned_bins = [final_bins[0]]
    for i in range(len(final_bins) - 1):
        mask = (df_cleaned[var] >= final_bins[i]) & (df_cleaned[var] < final_bins[i + 1])
        if mask.sum() > 0:
            cleaned_bins.append(final_bins[i + 1])

    cleaned_bins.append(np.inf)
    labels = [
        f"[{cleaned_bins[i]}-{cleaned_bins[i+1]}["
        if cleaned_bins[i+1] != np.inf else f"+{cleaned_bins[i]}"
        for i in range(len(cleaned_bins) - 1)
    ]

    return cleaned_bins, labels
    