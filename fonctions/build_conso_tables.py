import pandas as pd
import numpy as np
import streamlit as st
import openpyxl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
   
from fonctions.workOnData import formatM


try:
    import dataframe_image as dfi # Librairie très instable problèmes récurrents
except:
    print("erreur de la librairie dfi")
    
sorted_Famille={'Hospitalisation':1,
'Consultations et visites':2,
'Soins courants':3,
'Pharmacie':4,
'Dentaire':5,
'Optique':6,
'Divers':7}



Hospitalisation={'Honoraires':1,
                 'Actes techniques':2,
                 'Frais de séjour':3,
                 'Chambre particulière':4,
                 'Frais médicaux':5,
                 'Spécialistes':6,
                 'Généralistes':7,
                 'Frais d\'accompagnement':8,
                 'Forfaits':9}

L_Hospitalisation=sorted(Hospitalisation, key=Hospitalisation.get)


CS={'Généralistes':1,
                 'Spécialistes':2,
                 'Téléconsultation':3,
                 'Majoration':4}

L_CS=sorted(CS, key=CS.get)



Pharmacie={'Pharmacie à 15%':1,
                 'Pharmacie à 30%':2,
                 'Pharmacie à 65%':3,
                 'Pharmacie à 100%':4,
                 'Vaccins':5,
                 'Pancement':6,
                 'Aérosol':7,
                 'Pharmacie refusée':8,
                 'Autres':9}

L_Pharmacie=sorted(Pharmacie, key=Pharmacie.get)

SC={'Honoraires paramédicaux':1,
                 'Analyses':2,
                 'Actes d\'imagerie':3,
                 'Actes techniques':4,
                 'Matériel médical':5,
                 'Médecine douce':6,
                 'Transport':7,
                 'Forfaits':8,
                 'Cures Thermales':9,
                 'Audioprothèses':10}

L_SC=sorted(SC, key=SC.get)


Optique={'Verres':1,
        'Monture':2,
        'Lunettes':3,
        'Lentilles remboursées':4,
        'Lentilles non remboursées':5,
        "Chirurgie correctrice de l'œil":6,
        'Autres':7}

L_Optique=sorted(Optique, key=Optique.get)


Dentaire={'Soins dentaires':1,
                 'Inlays onlays':2,
                 'Inlays core':3,
                 'Prothèses dentaires':4,
                 'Orthodontie acceptée':5,
                 'Implantologie':6,
                 'Parodontologie':7,
                 'Dentaire refusé':8}

L_Dentaire=sorted(Dentaire, key=Dentaire.get)


# ── Design system (cohérent avec charts.py) ───────────────────────────────
_T_HEADER_BG    = "#1A2440"   # bande titre sombre
_T_SUBHD_BG     = "#2C67AF"   # sous-en-têtes colonnes
_T_SUBHD2_BG    = "#2B3885"   # variante sous-en-tête
_T_COL1_BG      = "#EEF1F9"   # fond colonne libellé (très clair)
_T_ROW_ODD      = "#FFFFFF"   # ligne impaire
_T_ROW_EVEN     = "#F4F6FB"   # ligne paire (bleu très très doux)
_T_TOTAL_BG     = "#1A2440"   # ligne Total (identique header)
_T_TOTAL_FG     = "#1A2440"
_T_TEXT         = "#1A2440"   # texte cellules
_T_TEXT_MUTED   = "#4A5568"
_T_POS          = "#1A6B3C"   # évolution positive (vert foncé)
_T_NEG          = "#B91C1C"   # évolution négative (rouge foncé)
_T_POS_BG       = "#D1FAE5"   # fond positif
_T_NEG_BG       = "#FEE2E2"   # fond négatif
_T_BORDER       = "#E2E8F0"   # séparateur très discret
_T_FONT         = 13          # taille de base

def _base_table_styles():
    """Styles CSS communs à toutes les tables."""
    return [
        # En-têtes colonnes
        {"selector": "th",
         "props": [("background-color", _T_HEADER_BG),
                   ("color", "#FFFFFF"),
                   ("font-size", f"{_T_FONT}px"),
                   ("font-weight", "600"),
                   ("text-align", "center"),
                   ("padding", "8px 10px"),
                   ("border-bottom", f"2px solid {_T_SUBHD_BG}"),
                   ("border-right", f"1px solid {_T_BORDER}"),
]},
        # Cellules
        {"selector": "td",
         "props": [("font-size", f"{_T_FONT}px"),
                   ("color", _T_TEXT),
                   ("font-weight","bold"),
                   ("padding", "6px 10px"),
                   ("text-align", "right"),
                   ("border-bottom", f"1px solid {_T_BORDER}"),
                   ("border-right", f"1px solid {_T_BORDER}")]},
        # Lignes paires
        {"selector": "tr:nth-child(even) td",
         "props": [("background-color", _T_ROW_EVEN)]},
        # Lignes impaires
        {"selector": "tr:nth-child(odd) td",
         "props": [("background-color", _T_ROW_ODD)]},
        # Hover
        {"selector": "tr:hover td",
         "props": [("background-color", "#EEF1F9")]},
        # Table générale
        {"selector": "",
         "props": [("border-collapse", "collapse"),
                   ("border-radius", "8px"),
                   ("overflow", "hidden"),
                   ("font-family", "Segoe UI, Calibri, DejaVu Sans, sans-serif")]},
    ]

def _style_last_row_total(styler, n_rows):
    """Met en valeur la ligne Total (dernière ligne)."""
    styler = styler.set_table_styles(
        {n_rows - 1: [
            {"selector": "",
             "props": [("background-color", _T_TOTAL_BG),
                       ("color", _T_TOTAL_FG),
                       ("font-weight", "bold"),
                       ("font-size","16px"),
                       ("border-top", "2px solid #4A90D9")]},
            # Forcer aussi sur td pour écraser les alternances de lignes
            {"selector": "td",
             "props": [("background-color", _T_TOTAL_BG),
                       ("color", _T_TOTAL_FG),
                       ("font-weight","bold"),
                       ("font-weight", "bold")]},
        ]}, axis=1, overwrite=False)
    return styler

def _style_label_col(styler, col_name):
    """
    Colonne libellé (première colonne).
    - En-tête (th) : fond bleu _T_SUBHD_BG, texte blanc
    - Cellules (td) : fond clair _T_COL1_BG, texte sombre _T_TEXT, aligné à gauche
    """
    styler = styler.set_table_styles(
        {col_name: [
            # Style de l'en-tête th
            {"selector": "",
             "props": [("background-color", _T_SUBHD_BG),
                       ("color", "#FFFFFF"),
                       ("text-align", "center"),
                       ("font-weight", "600")]},
            # Style des cellules td — fond très clair + texte sombre lisible
            {"selector": "td",
             "props": [("background-color", _T_COL1_BG),
                       ("color", _T_TEXT),
                       ("text-align", "center"),
                       ("font-weight", "500")]},
        ]}, overwrite=False)
    return styler

def _color_pct(val):
    """Coloration conditionnelle pour les cellules de variation en %."""
    if not isinstance(val, str):
        return ""
    try:
        v = float(val.replace("%", "").replace("+", "").replace(",", ".").replace(" ", ""))
        if v > 0.001:
            return f"background-color: {_T_POS_BG}; color: {_T_POS}; font-weight: bold"
        elif v < -0.001:
            return f"background-color: {_T_NEG_BG}; color: {_T_NEG}; font-weight: bold"
    except Exception:
        pass
    return ""  # neutre : ne pas écraser le style td


def Famille_acte_sorted(df):
    Famille_acte_sorted = []
    for i in list(sorted_Famille.keys()):
        if i in df['famille_acte_aops'].unique():
            Famille_acte_sorted.append(i)
    return Famille_acte_sorted

def TableConso(df,Emplacement_stockage,ID,backend):

    #Ordre ligne
    ### kk
    sort_by=Famille_acte_sorted(df)
    if 'Divers' in sort_by:
        sort_by.remove('Divers')
    
    annee=int(df['annee_soins'].unique())
    # Dernière modif pour gérer les làl avec une ligne base une ligne option (problème pour le RàC total)
    dfassureur= pd.pivot_table(df,values=['frais_reels','rbt_ss','RC'],index=[ID,'famille_acte_aops'],aggfunc='sum').reset_index()
    dfassureur['frais_reels']=np.where(np.abs(dfassureur['frais_reels'])<np.abs(dfassureur['rbt_ss']+dfassureur['RC']),(dfassureur['rbt_ss']+dfassureur['RC'])*np.sign(dfassureur['RC']),dfassureur['frais_reels'])
    dfassureur['RàC']=dfassureur['frais_reels']-dfassureur['rbt_ss']-dfassureur['RC']
    ##################################


    table=pd.pivot_table(dfassureur, values=['frais_reels','RàC','RC','rbt_ss'], index=['famille_acte_aops'], aggfunc=np.sum).reset_index()
    table=table[table['famille_acte_aops']!='Divers']
    table.index=table['famille_acte_aops']
    table = table.reindex(sort_by).drop(columns='famille_acte_aops')
    table.reset_index()

    table2=pd.pivot_table(dfassureur, values=[ID], index=['famille_acte_aops'], aggfunc='nunique').reset_index()
    table=pd.merge(table,table2,on='famille_acte_aops')

    table['frais_reels']=np.where(table['RàC']<0,table['rbt_ss']+table['RC'],table['frais_reels'])
    table['RàC']=table['frais_reels']-table['RC']-table['rbt_ss']
    table.rename(columns={ID: "Nombre consommants",'frais_reels':'Frais réels','RàC':'Reste à charge','rbt_ss':'Remboursement sécurité sociale','RC':'Remboursement complémentaire'},inplace=True)
    
    TT = pd.DataFrame([[dfassureur[ID].nunique(),dfassureur['frais_reels'].sum(),dfassureur['rbt_ss'].sum(),dfassureur['RC'].sum(),dfassureur['RàC'].sum()]], columns=['Nombre consommants','frais_reels','Remboursement sécurité sociale','Remboursement complémentaire','RàC'], index=['Total']).reset_index().rename(columns={'index':'famille_acte_aops','frais_reels':'Frais réels','RàC':'Reste à charge'})
    
    TT=TT[['famille_acte_aops','Frais réels','Remboursement complémentaire','Remboursement sécurité sociale','Reste à charge','Nombre consommants']]
    table=pd.concat([table,TT],axis=0)
    table['Taux de couverture']=(table['Remboursement complémentaire']+table['Remboursement sécurité sociale'])/table['Frais réels']
    table = table[['famille_acte_aops','Nombre consommants','Frais réels','Remboursement sécurité sociale','Remboursement complémentaire','Reste à charge','Taux de couverture']].reset_index(drop=True).rename(columns={'famille_acte_aops':'Famille acte'})
    tableAvantMiseEnforme=table.copy() 

    table[['Nombre consommants','Frais réels','Remboursement sécurité sociale','Remboursement complémentaire','Reste à charge']]=table[['Nombre consommants','Frais réels','Remboursement sécurité sociale','Remboursement complémentaire','Reste à charge']].map(formatM)
    table=table.rename(columns={'Famille acte': f"Survenance {annee}"})   

    n_rows = len(table.index)
    table = (table.style
             .format({'Taux de couverture': "{:.0%}"})
             .set_table_styles(_base_table_styles(), overwrite=True)
             .hide(axis='index'))
    table = _style_label_col(table, f"Survenance {annee}")
    table = table.set_properties(**{'width': '100px', 'text-align': 'right'})
    annee = int(df['annee_soins'].unique())
    st.markdown(table.to_html(), unsafe_allow_html=True)

    if 'Remboursement complémentaire' in [c for c in table.columns]:
        table = table.set_table_styles(
            {'Remboursement complémentaire': [
                {"selector": "th",
                 "props": [("background-color", _T_SUBHD2_BG),
                           ("color", "#FFFFFF"),
                           ("text-align", "center"),
                           ("font-size", "15px")]},
                {"selector": "td",
                 "props": [("background-color", "#FFFFFF"),
                           ("color", "#2B3885"),
                           ("text-align", "right"),
                           ("font-size", "15px")]}
            ]}, overwrite=False)

    # Colonne Taux de couverture : accent violet discret
    if 'Taux de couverture' in [c for c in table.columns]:
        table = table.set_table_styles(
            {'Taux de couverture': [
                {"selector": "th",
                 "props": [("background-color", "#4A3570"),
                           ("color", "#FFFFFF"),
                           ("text-align", "center"),
                           ("font-size", "15px")]},
                {"selector": "td",
                 "props": [("background-color", "#FFFFFF"),
                           ("color", "#4A3570"),
                           ("text-align", "center"),
                           ("font-size", "15px")]}
            ]}, overwrite=False)

    
    table = _style_last_row_total(table, n_rows)

    table = table.set_properties(
        subset=pd.IndexSlice[n_rows-1, :],
        **{"font-size": "16px"}
    )
    
    try:
        dfi.export(table, Emplacement_stockage+"/"+str(annee)+'_tableConso.jpg',dpi=200,table_conversion=backend)
    except:
        print("erreur de la librairie dfi")  
        
    return tableAvantMiseEnforme



def table_N_vs_NMoins1(table1,table2,annee,Emplacement_stockage,backend):

    table=(table1[['Nombre consommants', 'Frais réels',
       'Remboursement sécurité sociale', 'Remboursement complémentaire',
       'Reste à charge', 'Taux de couverture']]/table2[['Nombre consommants', 'Frais réels',
       'Remboursement sécurité sociale', 'Remboursement complémentaire',
       'Reste à charge', 'Taux de couverture']]-1)
    # Remplacer inf, -inf, et NaN par 0 temporairement
    table = table.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Mise en forme en pourcentage
    table = table.map(lambda x: '{:.1%}'.format(x))


    table=pd.concat([table2[['Famille acte']],table],axis=1).rename(columns={'Famille acte':str(annee)+' vs '+str(annee-1)})
    tableAvantMiseEnforme=table.copy()

    col_label = str(annee) + ' vs ' + str(annee-1)
    n_rows    = len(table.index)
    num_cols  = [c for c in table.columns if c != col_label]

    table = (table.style
             .set_table_styles(_base_table_styles(), overwrite=True)
             .hide(axis='index'))
    table = _style_label_col(table, col_label)


    if 'Remboursement complémentaire' in [c for c in table.columns]:
        table = table.set_table_styles(
            {'Remboursement complémentaire': [
                {"selector": "th",
                 "props": [("background-color", _T_SUBHD2_BG),
                           ("color", "#FFFFFF"),
                           ("text-align", "center"),
                           ("font-size", "15px")]},
                {"selector": "td",
                 "props": [("background-color", "#FFFFFF"),
                           ("color", "#2B3885"),
                           ("text-align", "right"),
                           ("font-size", "15px")]}
            ]}, overwrite=False)

    # Colonne Taux de couverture : accent violet discret
    if 'Taux de couverture' in [c for c in table.columns]:
        table = table.set_table_styles(
            {'Taux de couverture': [
                {"selector": "th",
                 "props": [("background-color", "#4A3570"),
                           ("color", "#FFFFFF"),
                           ("text-align", "center"),
                           ("font-size", "15px")]},
                {"selector": "td",
                 "props": [("background-color", "#FFFFFF"),
                           ("color", "#4A3570"),
                           ("text-align", "center"),
                           ("font-size", "15px")]}
            ]}, overwrite=False)

    # Coloration conditionnelle : vert/rouge sur toutes les colonnes de variation
    table = table.map(_color_pct, subset=num_cols)
    table = _style_last_row_total(table, n_rows)
    table = table.set_properties(**{'width': '100px', 'text-align': 'right'})

    st.markdown(table.to_html(), unsafe_allow_html=True)
    try:
        dfi.export(table, Emplacement_stockage+"/"+str(annee)+'_vs_'+str(annee-1)+'_'+'_tableConso.jpg',dpi=200,table_conversion=backend)
    except:
        print("erreur de la librairie dfi")
    return tableAvantMiseEnforme


def format_table_Sousfamille(table, annee):
    """
    Mise en forme moderne pour les tables sous-familles.
    Design : épuré, lisible, cohérent avec la charte graphique.
    """
    n_rows = len(table.index)

    # Base
    table = table.set_table_styles(_base_table_styles(), overwrite=True)
    table = table.hide(axis='index')

    # Colonne libellé (année) : fond bleu, texte blanc, aligné à gauche
    table = _style_label_col(table, f"Survenance {annee}")

    # Colonne Remboursement complémentaire : accent bleu foncé
    if 'Remboursement complémentaire' in [c for c in table.columns]:
        table = table.set_table_styles(
            {'Remboursement complémentaire': [
                {"selector": "th",
                 "props": [("background-color", _T_SUBHD2_BG),
                           ("color", "#FFFFFF"),
                           ("text-align", "center"),
                           ("font-size", "15px")]},
                {"selector": "td",
                 "props": [("background-color", "#FFFFFF"),
                           ("color", "#2B3885"),
                           ("text-align", "right"),
                           ("font-size", "15px")]}
            ]}, overwrite=False)

    # Colonne Taux de couverture : accent violet discret
    if 'Taux de couverture' in [c for c in table.columns]:
        table = table.set_table_styles(
            {'Taux de couverture': [
                {"selector": "th",
                 "props": [("background-color", "#4A3570"),
                           ("color", "#FFFFFF"),
                           ("text-align", "center"),
                           ("font-size", "15px")]},
                {"selector": "td",
                 "props": [("background-color", "#FFFFFF"),
                           ("color", "#4A3570"),
                           ("text-align", "center"),
                           ("font-size", "15px")]}
            ]}, overwrite=False)

    # Ligne Total (dernière)
    table = _style_last_row_total(table, n_rows)

    # Largeur colonnes
    table = table.set_properties(**{'width': '100px', 'text-align': 'center'})

    return table

def TableConso_par_sous_familles(df,Emplacement_stockage,ID,mesure,Variable_bouclée,backend="chrome"):

    st.write(f"{Variable_bouclée} : {len(df[df[mesure].isna()])} lignes n'ont pas de sous famille renseignées. Soit {formatM(df[df[mesure].isna()]['RC'].sum())}€ de remboursement complémentaire")
    
    sort_by=['Dentaire','Optique','Hospitalisation','Consultations et visites',
             'Soins courants','Pharmacie'] #,'Divers'

    if df['annee_soins'].nunique()==1:
        annee=int(df['annee_soins'].unique()[0])
    else:
        print("Annees de survenance multiple dans l'extraction")
    # Dernière modif pour gérer les làl avec une ligne base une ligne option (problème pour le RàC total)
    dfassureur= pd.pivot_table(df,values=['frais_reels','rbt_ss','RC','nb_acte'],index=[ID,mesure],aggfunc='sum').reset_index()
    dfassureur['frais_reels']=np.where(np.abs(dfassureur['frais_reels'])<np.abs(dfassureur['rbt_ss']+dfassureur['RC']),(dfassureur['rbt_ss']+dfassureur['RC'])*np.sign(dfassureur['RC']),dfassureur['frais_reels'])
    dfassureur['RàC']=dfassureur['frais_reels']-dfassureur['RC']-dfassureur['rbt_ss']
    ##################################


    table=pd.pivot_table(dfassureur, values=['frais_reels','RàC','RC','rbt_ss','nb_acte'], index=[mesure], aggfunc='sum').reset_index()
    table=table[table[mesure]!='Divers']
    table.index=table[mesure]

    if 'Dentaire' in df[mesure].unique():
        table = table.reindex(sort_by).drop(columns=mesure)

    elif Variable_bouclée=='Hospitalisation':
        table = table.reindex(L_Hospitalisation).drop(columns=mesure)

    elif Variable_bouclée=='Pharmacie':
        table = table.reindex(L_Pharmacie).drop(columns=mesure)
    
    elif Variable_bouclée=='Optique':
        table = table.reindex(L_Optique).drop(columns=mesure)
    
    elif Variable_bouclée=='Dentaire':
        table = table.reindex(L_Dentaire).drop(columns=mesure)

    elif Variable_bouclée=='Consultations et visites':
        table = table.reindex(L_CS).drop(columns=mesure)

    elif Variable_bouclée=='Soins courants':
        table = table.reindex(L_SC).drop(columns=mesure)
    
    else:
        table = table.sort_values(by='RC',ascending=False).drop(columns=mesure)


    table2=pd.pivot_table(dfassureur, values=[ID], index=[mesure], aggfunc='nunique').reset_index()
    table=pd.merge(table,table2,on=mesure)
    # Correction si RàC <0
    table['frais_reels']=np.where(table['RàC']<0,table['rbt_ss']+table['RC'],table['frais_reels'])
    table['RàC']=table['frais_reels']-table['RC']-table['rbt_ss']
    table.rename(columns={ID: "Nombre consommants",'frais_reels':'Frais réels','RàC':'Reste à charge','rbt_ss':'Remboursement sécurité sociale','RC':'Remboursement complémentaire','nb_acte':'Nombre actes'},inplace=True)
    table['Remboursement complémentaire moyen par consommant']=(table['Remboursement complémentaire']/table['Nombre consommants']).round(2)

    TT = pd.DataFrame([[dfassureur[ID].nunique(),dfassureur['frais_reels'].sum(),dfassureur['rbt_ss'].sum(),dfassureur['RC'].sum(),dfassureur['RàC'].sum(),dfassureur['nb_acte'].sum(),(dfassureur['RC'].sum()/dfassureur['nb_acte'].sum())]], columns=['Nombre consommants','frais_reels','Remboursement sécurité sociale','Remboursement complémentaire','RàC','Nombre actes','Remboursement complémentaire moyen par acte'], index=['Total']).reset_index().rename(columns={'index':mesure,'frais_reels':'Frais réels','RàC':'Reste à charge'})
    TT['Remboursement complémentaire moyen par consommant'] = (dfassureur['RC'].sum() / dfassureur[ID].nunique()).round(2)
    
    TT=TT[[mesure,'Nombre actes','Frais réels','Remboursement complémentaire','Remboursement sécurité sociale','Reste à charge','Nombre consommants','Remboursement complémentaire moyen par consommant']]
    # Harmonisation colonne

    table=pd.concat([table,TT],axis=0)
    table['Taux de couverture']=(table['Remboursement complémentaire']+table['Remboursement sécurité sociale'])/table['Frais réels']
    table = table[[mesure,'Nombre consommants','Nombre actes','Remboursement complémentaire','Reste à charge','Taux de couverture','Remboursement complémentaire moyen par consommant']].reset_index(drop=True).fillna(0) # 'Frais réels','Remboursement sécurité sociale'
    tableAvantMiseEnforme=table.copy()

    table[['Nombre consommants','Nombre actes','Remboursement complémentaire','Reste à charge']]=table[['Nombre consommants','Nombre actes','Remboursement complémentaire','Reste à charge']].map(formatM) # 'Frais réels','Remboursement sécurité sociale'
    table = table.rename(columns={mesure: f"Survenance {annee}"})   
    table['Remboursement complémentaire moyen par consommant']=table['Remboursement complémentaire moyen par consommant'].apply(lambda x: '{:.2f} €'.format(x))
    table=table.style.format({'Taux de couverture': "{:.1%}"})

    table=format_table_Sousfamille(table,annee)
    st.dataframe(table, use_container_width=True)
    try:
        dfi.export(table, Emplacement_stockage+"/"+'table_détails_'+str(mesure)+'_'+str(Variable_bouclée)+'_'+str(annee)+'.jpg',dpi=100,table_conversion=backend)
    except:
        print("erreur de la librairie dfi")
    return tableAvantMiseEnforme


def comparaison_sf_n_n_1(tn,tn_1,Emplacement_stockage,annee,mesure,Variable_bouclée,backend):
    # Définir les index
    tn_idx = tn.set_index('sous_famille')
    tn_1_idx = tn_1.set_index('sous_famille')

    # Garder seulement les index en commun
    common_idx = tn_idx.index.intersection(tn_1_idx.index)

    # Filtrer les deux DataFrames sur les index en commun
    tn_common = tn_idx.loc[common_idx]
    tn_1_common = tn_1_idx.loc[common_idx]

    # Calcul du pourcentage de variation
    res = tn_common / tn_1_common - 1

    # Formatage conditionnel
    res = res.map(
        lambda x: '{:.1%}'.format(x) if (0.00001 < abs(x) < 1000 or x == 0) else ""
    )
    table = res.reset_index().rename(columns={mesure:str(annee)+' vs '+str(annee-1)})

    tableAvantMiseEnforme = table.copy()
    col_label = str(annee) + ' vs ' + str(annee-1)
    num_cols  = [c for c in table.columns if c != col_label]
    table = (table.style
             .set_table_styles(_base_table_styles(), overwrite=True)
             .hide(axis='index'))
    if 'Remboursement complémentaire' in [c for c in table.columns]:
        table = table.set_table_styles(
            {'Remboursement complémentaire': [
                {"selector": "th",
                 "props": [("background-color", _T_SUBHD2_BG),
                           ("color", "#FFFFFF"),
                           ("text-align", "center"),
                           ("font-size", "15px")]},
                {"selector": "td",
                 "props": [("background-color", "#FFFFFF"),
                           ("color", "#2B3885"),
                           ("text-align", "right"),
                           ("font-size", "15px")]}
            ]}, overwrite=False)

    # Colonne Taux de couverture : accent violet discret
    if 'Taux de couverture' in [c for c in table.columns]:
        table = table.set_table_styles(
            {'Taux de couverture': [
                {"selector": "th",
                 "props": [("background-color", "#4A3570"),
                           ("color", "#FFFFFF"),
                           ("text-align", "center"),
                           ("font-size", "15px")]},
                {"selector": "td",
                 "props": [("background-color", "#FFFFFF"),
                           ("color", "#4A3570"),
                           ("text-align", "center"),
                           ("font-size", "15px")]}
            ]}, overwrite=False)

    table = _style_label_col(table, col_label)
    table = table.map(_color_pct, subset=num_cols)
    table = _style_last_row_total(table, len(table.index))
    table = table.set_properties(**{'width': '100px', 'text-align': 'right'})
     

    st.markdown(table.to_html(), unsafe_allow_html=True)
    
    try:
        dfi.export(table, Emplacement_stockage+"/"+'table_détails_sf_n_n_1_'+str(mesure)+'_'+str(Variable_bouclée)+'_'+str(annee)+"_vs_"+str(annee-1)+'.jpg',dpi=100,table_conversion=backend)
    except:
        print("erreur de la librairie dfi")
    
    return tableAvantMiseEnforme

def ajouter_tableau_excel(file_path, df, sheet_name, startrow, startcol):
    """
    Ajoute un DataFrame à un fichier Excel en précisant la feuille et la position.

    :param file_path: str - Chemin du fichier Excel
    :param df: pd.DataFrame - Tableau à insérer
    :param sheet_name: str - Nom de la feuille où insérer le tableau
    :param startrow: int - Ligne de départ (Excel indexé à 0)
    :param startcol: int - Colonne de départ (Excel indexé à 0)
    """

    try:
        # Vérifier si le fichier existe
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, startcol=startcol)
    except FileNotFoundError:
        # Si le fichier n'existe pas, le créer et ajouter la table
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, startcol=startcol)



def table_dentaire(df):
    
    df['sante_100'] = df['sante_100'].str.title()

    pt = pd.pivot_table(
    df,
    values=['id_beneficiaire', 'nb_acte', 'frais_reels', 'rbt_ss', 'RC'],
    index=['sante_100'],
    columns='annee_soins',
    aggfunc={
        'id_beneficiaire': 'nunique',
        'nb_acte': 'sum',
        'frais_reels': 'sum',
        'rbt_ss': 'sum',
        'RC': 'sum'
    },
    margins=True,margins_name='Total'
    )
        # 1️⃣ Calcul du RAC
    rac = (
        pt.xs('frais_reels', axis=1, level=0)
        - pt.xs('rbt_ss', axis=1, level=0)
        - pt.xs('RC', axis=1, level=0)
    )
    # 2️⃣ Recréer un MultiIndex de colonnes
    rac.columns = pd.MultiIndex.from_product(
        [['RAC'], rac.columns],
        names=pt.columns.names
    )
    # 3️⃣ Concaténation au pivot
    pt = pd.concat([pt, rac], axis=1)
    pt.loc['100% Santé', ('RAC', slice(None))] = 0

    # 1️⃣ Calcul RC moyen acte
    rc_moyen_acte = (
    pt.xs('RC', axis=1, level=0)/pt.xs('nb_acte', axis=1, level=0)
    )
    # 2️⃣ Recréer un MultiIndex de colonnes
    rc_moyen_acte.columns = pd.MultiIndex.from_product(
        [['rc_moyen_acte'], rc_moyen_acte.columns],
        names=pt.columns.names
    )
    # 3️⃣ Concaténation au pivot
    pt = pd.concat([pt, rc_moyen_acte], axis=1)


    # 1️⃣ Calcul RAC moyen acte
    rac_moyen_acte = (
    pt.xs('RAC', axis=1, level=0)/pt.xs('nb_acte', axis=1, level=0)
    )
    # 2️⃣ Recréer un MultiIndex de colonnes
    rac_moyen_acte.columns = pd.MultiIndex.from_product(
        [['rac_moyen_acte'], rac_moyen_acte.columns],
        names=pt.columns.names
    )
    # 3️⃣ Concaténation au pivot
    pt = pd.concat([pt, rac_moyen_acte], axis=1)


    # Suppression du total en colonne uniquement
    pt = pt.drop(columns='Total', level=1)
    ordre=['100% Santé','Maîtrisés','Libre','Total']
    ordre_table=[]
    for o in ordre:
        if o in pt.index:
            ordre_table.append(o)
    pt = pt.reindex(ordre_table)

    pt = pt.reindex(
        columns=[
            'id_beneficiaire',
            'nb_acte',
            'RC',
            'rc_moyen_acte',
            'rac_moyen_acte'
        ],
        level=0
    )
    pt=pt.rename(columns={'id_beneficiaire':'Consommants',
            'nb_acte':'Nombre actes',
            'RC':'Remboursement complémentaire',
            'rc_moyen_acte':'Remboursement complémentaire par acte',
            'rac_moyen_acte':'Reste à charge par acte'})
    pt.columns.names=[None, None]
    pt.index.name=None
    pt=pt.map(formatM)
    pt.index = pt.index.astype(str).str.capitalize()

    return pt


def style_pivot_dentaire(pt):

    def highlight_total(row):
        if row.name == 'Total':
            return ['background-color:#EEF1F9; color:#1A2440; font-weight:bold'] * len(row)
        else:
            return ['background-color:#FFFFFF; color:#1A2440; font-weight:bold'] * len(row)

    # 🔹 Bordures blanches par bloc level 0 (CORRIGÉ)
    border_styles = []
    cols = pt.columns
    level0 = cols.get_level_values(0)

    for label in level0.unique():
        locs = list((level0 == label).nonzero()[0])
        start, end = locs[0], locs[-1]

        border_styles = []

        level0 = pt.columns.get_level_values(0)
        n_cols = len(pt.columns)
        
        for label in level0.unique():
            locs = list((level0 == label).nonzero()[0])
            start, end = locs[0], locs[-1]  # 0-based pour python
            start_css, end_css = start, end  # CSS nth-child est 1-based
        
            # On ignore la première colonne et la dernière
            if start_css != 1 and end_css != n_cols:
                # Bordure droite uniquement sur la dernière colonne du bloc
                border_styles.append({
                    'selector':
                                f'th.col_heading.level1:nth-child({end_css}), '
                                f'td:nth-child({end_css})',
                    'props': [('border-right', '2px solid white')]
                })

    styler = (
        pt.style
        # alignement général
        .set_properties(**{
            'text-align': 'center',
            'font-weight': 'bold'
        })
        # styles
        .set_table_styles([
            # level 0 (groupes)
            {
                'selector': 'th.col_heading.level0',
                'props': [
                    ('background-color', '#173A64'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center')
                ]
            },
            # level 1 (années)
            {
                'selector': 'th.col_heading.level1',
                'props': [
                    ('background-color', '#2C67AF'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('width', '90px')
                ]
            },
            # index
            {
                'selector': 'th.row_heading',
                'props': [
                    ('font-weight', 'bold'),
                    ('text-align', 'left'),
                    ('background-color', '#173A64'),
                    ('color', '#FFFFFF')
                ]
            },
            # index Total
            {
                'selector': 'tbody tr:last-child th',
                'props': [
                    ('background-color', '#662064'),
                    ('text-align', 'left'),
                    ('color', 'white')
                ]
            },
        ] + border_styles, overwrite=False)
        # ligne TOTAL
        .apply(highlight_total, axis=1)
        # NA → tiret
        .format(na_rep='-')
    )

    return styler

def formatM_with_zero(x):
    """Formate les valeurs et remplace les 0 par '-'"""
    if pd.isna(x):
        return "-"
    
    # Si c'est déjà une chaîne (résultat de formatM), vérifier si c'est "0"
    if isinstance(x, str):
        if x in ['0', '0.0', '0,0', '0 €', '0€', '0.00', '0,00','-0']:
            return "-"
        return x
    
    # Si c'est un nombre
    if x == 0 or x == 0.0 or x==-0:
        return "-"
    
    # Sinon, appliquer le formatage normal
    return formatM(x)


def table_optique(df):

    df['sante_100'] = df['sante_100'].str.title()

    pt=pd.pivot_table(df, 
               values=['id_beneficiaire','nb_acte','frais_reels','rbt_ss','RC'],index=['sous_famille','sante_100'],columns='annee_soins',aggfunc={'id_beneficiaire':'nunique','nb_acte':'sum',
                                                                                                                                     'frais_reels':'sum','rbt_ss':'sum','RC':'sum'},margins=True,margins_name='Total')

    # 1️⃣ Calcul du RAC
    rac = (
        pt.xs('frais_reels', axis=1, level=0)
        - pt.xs('rbt_ss', axis=1, level=0)
        - pt.xs('RC', axis=1, level=0)
    )
    # 2️⃣ Recréer un MultiIndex de colonnes
    rac.columns = pd.MultiIndex.from_product(
        [['RAC'], rac.columns],
        names=pt.columns.names
    )
    # 3️⃣ Concaténation au pivot
    pt = pd.concat([pt, rac], axis=1)

    # 1️⃣ Calcul RC moyen acte
    rc_moyen_acte = (
    pt.xs('RC', axis=1, level=0)/pt.xs('nb_acte', axis=1, level=0)
    )
    # 2️⃣ Recréer un MultiIndex de colonnes
    rc_moyen_acte.columns = pd.MultiIndex.from_product(
        [['rc_moyen_acte'], rc_moyen_acte.columns],
        names=pt.columns.names
    )
    # 3️⃣ Concaténation au pivot
    pt = pd.concat([pt, rc_moyen_acte], axis=1)


    # 1️⃣ Calcul RAC moyen acte
    rac_moyen_acte = (
    pt.xs('RAC', axis=1, level=0)/pt.xs('nb_acte', axis=1, level=0)
    )
    # 2️⃣ Recréer un MultiIndex de colonnes
    rac_moyen_acte.columns = pd.MultiIndex.from_product(
        [['rac_moyen_acte'], rac_moyen_acte.columns],
        names=pt.columns.names
    )
    # 3️⃣ Concaténation au pivot
    pt = pd.concat([pt, rac_moyen_acte], axis=1)


    # Suppression du total en colonne uniquement
    pt = pt.drop(columns='Total', level=1)


    pt = pt.reindex(
        columns=[
            'id_beneficiaire',
            'nb_acte',
            'RC',
            'rc_moyen_acte',
            'rac_moyen_acte'
        ],
        level=0
    )
    pt=pt.rename(columns={'id_beneficiaire':'Consommants',
            'nb_acte':'Nombre actes',
            'RC':'Remboursement complémentaire',
            'rc_moyen_acte':'Remboursement complémentaire par acte',
            'rac_moyen_acte':'Reste à charge par acte'})
    pt.columns.names=[None, None]
    pt.index.names=[None, None]
    pt=pt.round(0)
    pt = pt.map(formatM_with_zero)
    return pt


HEADER_BG = "#1A2440"
SUBHEADER_BG = "#2C67AF"
MONTURE_BG = "#F8E1E4"
VERRES_BG = "#EAD1DC"
TOTAL_BG = "#C27BA0"
TEXT_WHITE = "white"

def style_rows(row):
    """Style les lignes selon la sous-famille"""
    sous_famille = row.name[0]  # niveau 0 du MultiIndex
    if sous_famille == "Monture":
        return ["background-color: #F8E1E4;color:#173A64; font-weight: bold"] * len(row)
    elif sous_famille == "Verres":
        return ["background-color: #EAD1DC;color:#173A64; font-weight: bold"] * len(row)
    elif sous_famille == "Total":
        return ["background-color: #C27BA0; font-weight: bold;color:#FFFFFF; border-top:2px solid white"] * len(row)
    return [""] * len(row)

def style_index_css(pt):
    """Style les cellules d'index selon la sous-famille"""
    styles = []
    for i, idx in enumerate(pt.index):
        if idx[0] == "Monture":
            styles.append({
                "selector": f"tbody tr:nth-child({i+1}) th.row_heading",
                "props": "background-color: #D86173; font-weight: bold;color:#FFFFFF"
            })
        elif idx[0] == "Verres":
            styles.append({
                "selector": f"tbody tr:nth-child({i+1}) th.row_heading",
                "props": "background-color: #9B406D; font-weight: bold;color:#FFFFFF"
            })
        elif idx[0] == "Total":
            styles.append({
                "selector": f"tbody tr:nth-child({i+1}) th.row_heading",
                "props": "background-color: #662064; font-weight: bold;color:#FFFFFF; border-top:2px solid white"
            })
    return styles


def apply_pivot_style_optique(pt):
    """Applique tous les styles au pivot table"""
    styled_pt = (
        pt.style
          .apply(style_rows, axis=1)
          .set_properties(
              text_align="center",
              width='90px'
          )
          .set_table_styles(
              style_index_css(pt)                    # couleurs index lignes
            + [
                {
                    "selector": "th.col_heading.level0",
                    "props": [
                        ("background-color", HEADER_BG),
                        ("color", TEXT_WHITE),
                        ("text-align", "center"),
                        ("font-weight", "bold"),
                        ("border", "2px solid white")
                    ],
                },
                {
                    "selector": "th.col_heading.level1",
                    "props": [
                        ("background-color", SUBHEADER_BG),
                        ("color", TEXT_WHITE),
                        ("text-align", "center"),
                        ("border", "2px solid white")
                    ],
                },
                {
                    "selector": "th.row_heading",
                    "props": [
                        ("text-align", "left"),
                        ("font-weight", "bold"),
                        ("border-right", "2px solid white")
                    ],
                },
                {
                    "selector": "tbody tr:last-child th.row_heading.level0",
                    "props": [
                        ("border-right", "none")
                    ],
                },
                {
                    "selector": "tbody tr:last-child th.row_heading.level1",
                    "props": [
                        ("border-right", "2px solid white")
                    ],
                },
                {
                    "selector": "tbody tr:first-child th.row_heading",  # ✅ Première case de l'index
                    "props": [
                        ("border-top", "2px solid white")
                    ],
                }
            ],
            overwrite=False
          )
          .format(na_rep="-")
    )
    return styled_pt


def table_100_sante(df,Emplacement_stockage,backend):
    
    if df.empty:
        st.write("Aucune donnée disponible pour le tableau 100% santé.")
        return None
    elif 'sante_100' not in df.columns:
        st.write("Aucune donnée 100% santé disponible pour le tableau.")
        return None 
    elif df['sante_100'].isna().all():
        st.write("Aucune donnée 100% santé disponible pour le tableau.")
        return None
    elif df['famille_acte_aops'].nunique() > 1:
        st.write("Le tableau 100% santé ne peut être généré que pour une seule famille d'actes à la fois.")
        return None
    elif (df['famille_acte_aops'].iloc[0] not in ['Optique', 'Dentaire','Soins courants']):
        st.write("Le tableau 100% santé n'est disponible que pour les familles d'actes Optique et Dentaire ainsi que pour la sous-famille Audioprothèses.")
        return None
    elif df['famille_acte_aops'].iloc[0]=='Dentaire':
        pt = table_dentaire(df)
        table=style_pivot_dentaire(pt)
        st.dataframe(table, use_container_width=True)
        try:
            dfi.export(table, Emplacement_stockage+"/"+'table_100_sante_dentaire.jpg',dpi=100,table_conversion=backend)
        except:
            print("erreur de la librairie dfi")
    elif df['famille_acte_aops'].iloc[0]=='Optique':
        pt = table_optique(df)
        table = apply_pivot_style_optique(pt)
        st.dataframe(table, use_container_width=True)
        try:
            dfi.export(table, Emplacement_stockage+"/"+'table_100_sante_optique.jpg',dpi=100,table_conversion=backend)
        except:
            print("erreur de la librairie dfi")
    elif df['sous_famille'].iloc[0]=='Audioprothèses':
        pt = table_optique(df)
        table = apply_pivot_style_optique(pt)
        st.dataframe(table, use_container_width=True)
        try:
            dfi.export(table, Emplacement_stockage+"/"+'table_100_sante_audioprothèses.jpg',dpi=100,table_conversion=backend)
        except:
            print("erreur de la librairie dfi")



def proportion_cat_assure(d, cat_assure, Emplacement_stockage, backend="matplotlib"):

    if not isinstance(cat_assure, list):
        cat_assure = [cat_assure]

    df = d.copy()

    table = pd.pivot_table(
        df, values='RC', index='cat_assure',
        columns='annee_soins', aggfunc='sum'
    )

    rc_porta  = table.loc[cat_assure].sum(axis=0)
    rc_total  = table.sum()
    prop_porta = rc_porta / rc_total
    years     = rc_porta.index.astype(int)

    label_cat = " - ".join(cat_assure) if len(cat_assure) > 1 else cat_assure[0]

    # ── Layout ───────────────────────────────────────────────────────────────
    from fonctions.charts import (
        _make_fig, _add_main_ax, _strip_title,
        _finalize, _apply_ax_theme, _kpi_row,
        BG, BG_AX, HEX, STRIP_DARK, C_GRID, C_ANNOT
    )
    from matplotlib.ticker import FuncFormatter, EngFormatter

    def _fmt_euro(x, _=None):
        if abs(x) >= 1_000_000: return f"{x/1_000_000:.1f} M€"
        if abs(x) >= 1_000:     return f"{x/1_000:.0f} k€"
        return f"{x:.0f} €"

    fig     = _make_fig(13, 6)
    title_h = _strip_title(fig,
        f"Montant total par survenance  ·  Part dans le remboursement global")

    kpi_h   = 0.12
    bottom  = kpi_h + 0.01
    ax1     = _add_main_ax(fig, [0.06, bottom, 0.88, 1 - title_h - 0.03 - bottom])

    # ── Barres RC ─────────────────────────────────────────────────────────────
    x     = np.arange(len(years))
    bars  = ax1.bar(x, rc_porta.values, color=HEX[2], width=0.5,
                    edgecolor=BG, linewidth=0.8, alpha=0.90, zorder=3)

    # Légère surbrillance
    

    _apply_ax_theme(ax1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(_fmt_euro))
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="x", length=0.0)

    # Annotations montants sur barres
    for bar, val in zip(bars, rc_porta.values):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() * 1.012,
                 _fmt_euro(val),
                 ha="center", va="bottom",
                 fontsize=18, fontweight="bold", color=HEX[2])

    # ── Courbe % (axe droit) ──────────────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.plot(x, prop_porta.values * 100,
             color=HEX[8], linewidth=2.5, marker="o", markersize=15,
             zorder=5, markerfacecolor="white",
             markeredgewidth=2, markeredgecolor=HEX[8])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f} %"))
    ax2.tick_params(axis="y", labelsize=15, colors=HEX[8], length=0)
    for sp in ax2.spines.values(): sp.set_visible(False)
    ax2.set_ylim(bottom=0,
                 top=max(prop_porta.values) * 100 * 1.35)
    ax2.grid(False)

    # Annotations % au-dessus des points
    for xi, pct in zip(x, prop_porta.values):
        ax2.annotate(
            f"{pct*100:.1f} %",
            xy=(xi, pct * 100),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=17, fontweight="bold",
            color=HEX[8],
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="none", alpha=0)
        )

    # ── Légende ───────────────────────────────────────────────────────────────
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    patch = mpatches.Patch(color=HEX[1], label=f"RC")
    line  = mlines.Line2D([], [], color=HEX[7], marker="o",
                          markersize=8, markerfacecolor="white",
                          markeredgewidth=1.5, label="Part dans le total")
    ax1.legend(handles=[patch, line],
               loc="lower center", bbox_to_anchor=(0.5, -0.18),
               ncol=2, fontsize=10.5, frameon=False)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    title_safe = (label_cat.replace(' ', '_')
                            .replace(',', '').replace(':', '')
                            .replace('-', '_'))
    filepath = f"{Emplacement_stockage}/RC_proportion_{title_safe}.jpg"
    _finalize(fig, filepath, 150)


def table_evolution_kpis(df, cat_assure, annees, ID, Emplacement_stockage, backend="chrome"):
    df_filtre = df[df['annee_soins'].isin(annees)].copy()
    df_filtre = df_filtre[df_filtre['cat_assure'].isin(cat_assure)]

    annee = max(annees)
    resultats = {}
    for a in annees:
        sub = df_filtre[df_filtre['annee_soins'] == a]
        if sub.empty:
            resultats[a] = {'RC': 0, 'conso': 0, 'rc_moyen': 0}
        else:
            rc_total = sub.groupby(ID)['RC'].sum().sum()
            nb_conso = sub[ID].nunique()
            rc_moyen = round(rc_total / nb_conso, 2) if nb_conso > 0 else 0
            resultats[a] = {'RC': rc_total, 'conso': nb_conso, 'rc_moyen': rc_moyen}

    def evol(v_new, v_old):
        if v_old == 0:
            return None
        return (v_new / v_old - 1)

    col1 = f"{annee-1} / {annee-2}"
    col2 = f"{annee} / {annee-1}"

    lignes = [
        {
            'Indicateur': 'Remboursement complémentaire',
            col1: evol(resultats[annee-1]['RC'],      resultats[annee-2]['RC']),
            col2: evol(resultats[annee]['RC'],         resultats[annee-1]['RC']),
        },
        {
            'Indicateur': 'Nombre de consommants',
            col1: evol(resultats[annee-1]['conso'],   resultats[annee-2]['conso']),
            col2: evol(resultats[annee]['conso'],      resultats[annee-1]['conso']),
        },
        {
            'Indicateur': 'Remboursement complémentaire moyen',
            col1: evol(resultats[annee-1]['rc_moyen'], resultats[annee-2]['rc_moyen']),
            col2: evol(resultats[annee]['rc_moyen'],   resultats[annee-1]['rc_moyen']),
        },
    ]

    table = pd.DataFrame(lignes)
    tableAvantMiseEnforme = table.copy()

    def fmt_pct(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/D"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v*100:.1f} %"

    table[col1] = table[col1].map(fmt_pct)
    table[col2] = table[col2].map(fmt_pct)
    table = table.rename(columns={'Indicateur': 'Evolution'})

    n_rows = len(table)
    styled = (table.style
              .set_table_styles(_base_table_styles(), overwrite=True)
              .hide(axis='index'))

    styled = styled.set_table_styles([
    
        {"selector": "td", 
        "props": [("text-align", "center"), ("width", "140px")]}
            ], overwrite=False)
    styled = _style_label_col(styled, 'Evolution')
    styled = styled.map(_color_pct, subset=[col1, col2])
    styled = styled.set_properties(**{'width': '140px', 'text-align': 'center'})

    st.markdown(styled.to_html(), unsafe_allow_html=True)

    try:
        dfi.export(
            styled,
            f"{Emplacement_stockage}/table_evolution_kpis_{annee}.jpg",
            dpi=150, table_conversion=backend)
    except:
        print("erreur dfi export table_evolution_kpis")

    return tableAvantMiseEnforme