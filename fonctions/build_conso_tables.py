import pandas as pd
import numpy as np
import streamlit as st
import openpyxl

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


CS={'Spécialistes':1,
                 'Généralistes':2,
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

    table[['Nombre consommants','Frais réels','Remboursement sécurité sociale','Remboursement complémentaire','Reste à charge']]=table[['Nombre consommants','Frais réels','Remboursement sécurité sociale','Remboursement complémentaire','Reste à charge']].applymap(formatM)
    table=table.rename(columns={'Famille acte':str(annee)})    

    table=table.style.format({'Taux de couverture': "{:.0%}"})
    table=table.set_table_styles({str(annee): [{'selector': '','props': [('background-color', '#2C67AF'),('text-align', 'left'),('color', 'white'),('font-size', '14px')]}]}).hide(axis='index')
    table=table.set_table_styles({max(table.index): [{'selector': '','props': [('background-color', '#173A64'),('color', 'white'),('font-weight', 'bold'),('font-size', '14px')]}]}, axis=1, overwrite=False)  
    table=table.set_table_styles({max(table.index): [{'selector': '','props': [('border', '2px solid #FFFFFF'),('font-size', '14px')]}]}, axis=1, overwrite=False)  
    table=table.set_table_styles([{'selector': 'th:not(.index_name)','props': [('background-color', '#173A64'),('color', 'white'),('text-align', 'center'),('font-size', '14px')]}], overwrite=False)  
    table=table.set_table_styles([{'selector': 'td','props': [('color', 'black'),('border-left', '2px solid #FFFFFF'),('border-right', '2px solid #FFFFFF'),('text-align', 'center'),('font-size', '14px')]}], overwrite=False)  
    table=table.set_table_styles([{'selector': 'th','props': [('color', 'black'),('border', '2px solid #FFFFFF'),('font-size', '14px')]}], overwrite=False)  
    table=table.set_properties(**{'text-align': 'right','width':'100px'})
    annee=int(df['annee_soins'].unique())
    st.dataframe(table)
    
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
    table = table.applymap(lambda x: '{:.1%}'.format(x))


    table=pd.concat([table2[['Famille acte']],table],axis=1).rename(columns={'Famille acte':str(annee)+' vs '+str(annee-1)})
    tableAvantMiseEnforme=table.copy()

    table=table.style.set_table_styles({str(annee)+' vs '+str(annee-1): [{'selector': '','props': [('background-color', '#2C67AF'),('text-align', 'left'),('color', 'white'),('font-size', '14px')]}]}).hide(axis='index')
    table=table.set_table_styles({max(table.index): [{'selector': '','props': [('background-color', '#173A64'),('color', 'white'),('font-weight', 'bold'),('font-size', '14px')]}]}, axis=1, overwrite=False)  
    table=table.set_table_styles({max(table.index): [{'selector': '','props': [('border', '2px solid #FFFFFF'),('font-size', '14px')]}]}, axis=1, overwrite=False)  
    table=table.set_table_styles([{'selector': 'th:not(.index_name)','props': [('background-color', '#173A64'),('color', 'white'),('text-align', 'center'),('font-size', '14px')]}], overwrite=False)  
    table=table.set_table_styles([{'selector': 'td','props': [('color', 'black'),('border-left', '2px solid #FFFFFF'),('border-right', '2px solid #FFFFFF'),('text-align', 'center'),('font-size', '14px')]}], overwrite=False)  
    table=table.set_table_styles([{'selector': 'th','props': [('color', 'black'),('border', '2px solid #FFFFFF'),('font-size', '14px')]}], overwrite=False)  
    table=table.set_properties(**{'text-align': 'right','width':'100px'})

    st.dataframe(table)
    try:
        dfi.export(table, Emplacement_stockage+"/"+str(annee)+'_vs_'+str(annee-1)+'_'+'_tableConso.jpg',dpi=200,table_conversion=backend)
    except:
        print("erreur de la librairie dfi")
    return tableAvantMiseEnforme


def format_table_Sousfamille(table,annee):
    table=table.set_table_styles([
    {'selector': 'tr:nth-child(even)','props': [('background-color', '#ffffff'),('color', 'black')]},
    {'selector': 'tr:nth-child(odd)','props': [('background-color', '#cccccc'),('color', 'black')]}], overwrite=False)

    # 3 colonnes : index, RC, T%
    table=table.set_table_styles({str(annee): [{'selector': '','props': [('background-color', '#2C67AF'),('text-align', 'left'),('color', 'white'),('font-size', '14px')]}]}, overwrite=False).hide(axis='index')
    table=table.set_table_styles({'Remboursement complémentaire': [{'selector': '','props': [('background-color', '#2B3885'),('text-align', 'center'),('color', 'white'),('font-size', '14px')]}]}, overwrite=False)
    table=table.set_table_styles({'Taux de couverture': [{'selector': '','props': [('background-color', '#662064'),('text-align', 'center'),('color', 'white'),('font-size', '14px')]}]}, overwrite=False)

    # Dernière ligne
    table=table.set_table_styles({max(table.index): [{'selector': '','props': [('background-color', '#173A64'),('color', 'white'),('font-weight', 'bold'),('font-size', '14px')]}]}, axis=1, overwrite=False)  
    table=table.set_table_styles({max(table.index): [{'selector': '','props': [('border', '1px solid #FFFFFF'),('font-size', '14px')]}]}, axis=1, overwrite=False) 

    # Première ligne
    table=table.set_table_styles([{'selector': 'th:not(.index_name)','props': [('background-color', '#173A64'),('color', 'white'),('text-align', 'center'),('font-size', '14px')]}], overwrite=False) 
    table=table.set_table_styles([{'selector': 'td','props': [('color', 'black'),('border-left', '1px solid #FFFFFF'),('border-right', '1px solid #FFFFFF'),('text-align', 'center'),('font-size', '14px')]}], overwrite=False)  
    table=table.set_table_styles([{'selector': 'th','props': [('color', 'black'),('border', '1px solid #FFFFFF'),('font-size', '14px')]}], overwrite=False) 

    # largeur colonne
    table=table.set_properties(**{'text-align': 'right','width':'100px'})

    return table

def TableConso_par_sous_familles(df,Emplacement_stockage,ID,mesure,Variable_bouclée,backend):

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
    table['Remboursement complémentaire moyen']=round(table['RC']/table['nb_acte'],2)
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

    TT = pd.DataFrame([[dfassureur[ID].nunique(),dfassureur['frais_reels'].sum(),dfassureur['rbt_ss'].sum(),dfassureur['RC'].sum(),dfassureur['RàC'].sum(),dfassureur['nb_acte'].sum(),(dfassureur['RC'].sum()/dfassureur['nb_acte'].sum())]], columns=['Nombre consommants','frais_reels','Remboursement sécurité sociale','Remboursement complémentaire','RàC','Nombre actes','Remboursement complémentaire moyen'], index=['Total']).reset_index().rename(columns={'index':mesure,'frais_reels':'Frais réels','RàC':'Reste à charge'})

    TT=TT[[mesure,'Nombre actes','Frais réels','Remboursement complémentaire','Remboursement sécurité sociale','Reste à charge','Nombre consommants','Remboursement complémentaire moyen']]
    table=pd.concat([table,TT],axis=0)
    table['Taux de couverture']=(table['Remboursement complémentaire']+table['Remboursement sécurité sociale'])/table['Frais réels']
    table = table[[mesure,'Nombre consommants','Nombre actes','Remboursement complémentaire','Reste à charge','Taux de couverture','Remboursement complémentaire moyen']].reset_index(drop=True).fillna(0) # 'Frais réels','Remboursement sécurité sociale'
    tableAvantMiseEnforme=table.copy()

    table[['Nombre consommants','Nombre actes','Remboursement complémentaire','Reste à charge']]=table[['Nombre consommants','Nombre actes','Remboursement complémentaire','Reste à charge']].map(formatM) # 'Frais réels','Remboursement sécurité sociale'
    table=table.rename(columns={mesure:str(annee)})   
    table['Remboursement complémentaire moyen']=table['Remboursement complémentaire moyen'].apply(lambda x: '{:.2f} €'.format(x))
    table=table.style.format({'Taux de couverture': "{:.1%}"})

    table=format_table_Sousfamille(table,annee)
    st.dataframe(table)
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
    res = res.applymap(
        lambda x: '{:.1%}'.format(x) if (0.00001 < abs(x) < 1000 or x == 0) else ""
    )
    table = res.reset_index().rename(columns={mesure:str(annee)+' vs '+str(annee-1)})

    tableAvantMiseEnforme=table.copy()
    table=table.style.set_table_styles({str(annee)+' vs '+str(annee-1): [{'selector': '','props': [('background-color', '#2C67AF'),('text-align', 'left'),('color', 'white'),('font-size', '14px')]}]}).hide(axis='index')
    table=format_table_Sousfamille(table,annee)

    st.dataframe(table)
    
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
    pt.loc['100% santé', ('RAC', slice(None))] = 0

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
    ordre=['100% santé','Maîtrisés','Libre','Total']
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
            return ['background-color:#F7DFE2; color:#173A64; font-weight:bold'] * len(row)
        else:
            return ['background-color:#FFFFFF; color:#173A64; font-weight:bold'] * len(row)

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
                    ('background-color', '#0070C0'),
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
    pt=pd.pivot_table(df[(df['annee_soins']>=2024) & (df['annee_soins']==df['annee_paiement']) & (df['mois_paiement']<=9) & (df['mois_soins']<=9) & (df['sous_famille'].isin(['Verres', 'Monture']))], 
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


HEADER_BG = "#17375E"
SUBHEADER_BG = "#4F81BD"
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
    elif df['famille_acte_aops'].iloc[0] not in ['Optique', 'Dentaire']:
        st.write("Le tableau 100% santé n'est disponible que pour les familles d'actes Optique et Dentaire.")
        return None
    elif df['famille_acte_aops'].iloc[0]=='Dentaire':
        pt = table_dentaire(df)
        table=style_pivot_dentaire(pt)
        st.dataframe(table)
        try:
            dfi.export(table, Emplacement_stockage+"/"+'table_100_sante_dentaire.jpg',dpi=100,table_conversion=backend)
        except:
            print("erreur de la librairie dfi")
    elif df['famille_acte_aops'].iloc[0]=='Optique':
        pt = table_optique(df)
        table = apply_pivot_style_optique(pt)
        st.dataframe(table)
        try:
            dfi.export(table, Emplacement_stockage+"/"+'table_100_sante_optique.jpg',dpi=100,table_conversion=backend)
        except:
            print("erreur de la librairie dfi")



