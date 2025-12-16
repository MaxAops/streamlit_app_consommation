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
        if i in df['Famille acte'].unique():
            Famille_acte_sorted.append(i)
    return Famille_acte_sorted

def TableConso(df,Emplacement_stockage,ID,Assureur):

    #Ordre ligne
    ### kk
    sort_by=Famille_acte_sorted(df)
    if 'Divers' in sort_by:
        sort_by.remove('Divers')
    
    annee=int(df['annee_soins'].unique())
    # Dernière modif pour gérer les làl avec une ligne base une ligne option (problème pour le RàC total)
    dfassureur= pd.pivot_table(df,values=['FR','R_SS','RC'],index=[ID,'Famille acte'],aggfunc='sum').reset_index()
    dfassureur['FR']=np.where(np.abs(dfassureur['FR'])<np.abs(dfassureur['R_SS']+dfassureur['RC']),(dfassureur['R_SS']+dfassureur['RC'])*np.sign(dfassureur['RC']),dfassureur['FR'])
    dfassureur['RàC']=dfassureur['FR']-dfassureur['R_SS']-dfassureur['RC']
    ##################################


    table=pd.pivot_table(dfassureur, values=['FR','RàC','RC','R_SS'], index=['Famille acte'], aggfunc=np.sum).reset_index()
    table=table[table['Famille acte']!='Divers']
    table.index=table['Famille acte']
    table = table.reindex(sort_by).drop(columns='Famille acte')
    table.reset_index()


    table2=pd.pivot_table(dfassureur, values=[ID], index=['Famille acte'], aggfunc=pd.Series.nunique).reset_index()
    table=pd.merge(table,table2,on='Famille acte')
    table['FR']=np.where(table['RàC']<0,table['R_SS']+table['RC'],table['FR'])
    table['RàC']=table['FR']-table['RC']-table['R_SS']
    table.rename(columns={ID: "Nombre consommants",'FR':'Frais réels','RàC':'Reste à charge','R_SS':'Remboursement sécurité sociale','RC':'Remboursement complémentaire'},inplace=True)


    TT = pd.DataFrame([[dfassureur[ID].nunique(),dfassureur['FR'].sum(),dfassureur['R_SS'].sum(),dfassureur['RC'].sum(),dfassureur['RàC'].sum()]], columns=['Nombre consommants','FR','Remboursement sécurité sociale','Remboursement complémentaire','RàC'], index=['Total']).reset_index().rename(columns={'index':'Famille acte','FR':'Frais réels','RàC':'Reste à charge'})

    TT=TT[['Famille acte','Frais réels','Remboursement complémentaire','Remboursement sécurité sociale','Reste à charge','Nombre consommants']]
    table=pd.concat([table,TT],axis=0)
    table['Taux de couverture']=(table['Remboursement complémentaire']+table['Remboursement sécurité sociale'])/table['Frais réels']
    table = table[['Famille acte','Nombre consommants','Frais réels','Remboursement sécurité sociale','Remboursement complémentaire','Reste à charge','Taux de couverture']].reset_index(drop=True)
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
        dfi.export(table, Emplacement_stockage+"/"+str(annee)+str(Assureur)+'_tableConso.jpg',dpi=200,table_conversion='matplotlib')#chrome
    except:
        print("erreur de la librairie dfi")  
        
    return tableAvantMiseEnforme



def table_N_vs_NMoins1(table1,table2,annee,Emplacement_stockage,Assureur):

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
        dfi.export(table, Emplacement_stockage+"/"+str(annee)+'_vs_'+str(annee-1)+'_'+str(Assureur)+'_tableConso.jpg',dpi=200,table_conversion='matplotlib')
    except:
        print("erreur de la librairie dfi")
    return tableAvantMiseEnforme

""" # Ancienne version avec FR et R_SS
def TableConso_par_sous_familles(df,Emplacement_stockage,ID,mesure,Variable_bouclée):

    st.write(f"{Variable_bouclée} : {len(df[df[mesure].isna()])} lignes n'ont pas de sous famille renseignées. Soit {formatM(df[df[mesure].isna()]['RC'].sum())}€ de remboursement complémentaire")
    
    sort_by=['Dentaire','Optique','Hospitalisation','Consultations et visites',
             'Soins courants','Pharmacie'] #,'Divers'

    annee=int(df['annee_soins'].unique())
    # Dernière modif pour gérer les làl avec une ligne base une ligne option (problème pour le RàC total)
    dfassureur= pd.pivot_table(df,values=['FR','R_SS','RC','nombre_acte'],index=[ID,mesure],aggfunc='sum').reset_index()
    dfassureur['FR']=np.where(np.abs(dfassureur['FR'])<np.abs(dfassureur['R_SS']+dfassureur['RC']),(dfassureur['R_SS']+dfassureur['RC'])*np.sign(dfassureur['RC']),dfassureur['FR'])
    dfassureur['RàC']=dfassureur['FR']-dfassureur['R_SS']-dfassureur['RC']
    ##################################


    table=pd.pivot_table(dfassureur, values=['FR','RàC','RC','R_SS','nombre_acte'], index=[mesure], aggfunc=np.sum).reset_index()
    table['Remboursement complémentaire moyen']=round(table['RC']/table['nombre_acte'],2)
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


    table2=pd.pivot_table(dfassureur, values=[ID], index=[mesure], aggfunc=lambda x: len(x.unique())).reset_index()
    table=pd.merge(table,table2,on=mesure)
    # Correction si RàC <0
    table['FR']=np.where(table['RàC']<0,table['R_SS']+table['RC'],table['FR'])
    table['RàC']=table['FR']-table['RC']-table['R_SS']
    table.rename(columns={ID: "Nombre consommants",'FR':'Frais réels','RàC':'Reste à charge','R_SS':'Remboursement sécurité sociale','RC':'Remboursement complémentaire','nombre_acte':'Nombre actes'},inplace=True)

    TT = pd.DataFrame([[dfassureur[ID].nunique(),dfassureur['FR'].sum(),dfassureur['R_SS'].sum(),dfassureur['RC'].sum(),dfassureur['RàC'].sum(),dfassureur['nombre_acte'].sum(),(dfassureur['RC'].sum()/dfassureur['nombre_acte'].sum())]], columns=['Nombre consommants','FR','Remboursement sécurité sociale','Remboursement complémentaire','RàC','Nombre actes','Remboursement complémentaire moyen'], index=['Total']).reset_index().rename(columns={'index':mesure,'FR':'Frais réels','RàC':'Reste à charge'})

    TT=TT[[mesure,'Nombre actes','Frais réels','Remboursement complémentaire','Remboursement sécurité sociale','Reste à charge','Nombre consommants','Remboursement complémentaire moyen']]
    table=pd.concat([table,TT],axis=0)
    table['Taux de couverture']=(table['Remboursement complémentaire']+table['Remboursement sécurité sociale'])/table['Frais réels']
    table = table[[mesure,'Nombre consommants','Nombre actes','Frais réels','Remboursement sécurité sociale','Remboursement complémentaire','Reste à charge','Taux de couverture','Remboursement complémentaire moyen']].reset_index(drop=True).fillna(0)
    tableAvantMiseEnforme=table.copy()

    table[['Nombre consommants','Nombre actes','Frais réels','Remboursement sécurité sociale','Remboursement complémentaire','Reste à charge']]=table[['Nombre consommants','Nombre actes','Frais réels','Remboursement sécurité sociale','Remboursement complémentaire','Reste à charge']].applymap(formatM)
    table=table.rename(columns={mesure:str(annee)})   
    table['Remboursement complémentaire moyen']=table['Remboursement complémentaire moyen'].apply(lambda x: '{:.2f} €'.format(x))

    table=table.style.format({'Taux de couverture': "{:.1%}"})

# Mise en forme
    #Couleur une ligne sur 2
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

    st.dataframe(table)
    try:
        dfi.export(table, Emplacement_stockage+'table_détails'+str(mesure)+'_'+str(Variable_bouclée)+'_'+str(annee)+'.jpg',dpi=100,table_conversion='chrome')
    except:
        print("erreur de la librairie dfi")
    return tableAvantMiseEnforme
"""

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

def TableConso_par_sous_familles(df,Emplacement_stockage,ID,mesure,Variable_bouclée):

    st.write(f"{Variable_bouclée} : {len(df[df[mesure].isna()])} lignes n'ont pas de sous famille renseignées. Soit {formatM(df[df[mesure].isna()]['RC'].sum())}€ de remboursement complémentaire")
    
    sort_by=['Dentaire','Optique','Hospitalisation','Consultations et visites',
             'Soins courants','Pharmacie'] #,'Divers'

    annee=int(df['annee_soins'].unique())
    # Dernière modif pour gérer les làl avec une ligne base une ligne option (problème pour le RàC total)
    dfassureur= pd.pivot_table(df,values=['FR','R_SS','RC','nombre_acte'],index=[ID,mesure],aggfunc='sum').reset_index()
    dfassureur['FR']=np.where(np.abs(dfassureur['FR'])<np.abs(dfassureur['R_SS']+dfassureur['RC']),(dfassureur['R_SS']+dfassureur['RC'])*np.sign(dfassureur['RC']),dfassureur['FR'])
    dfassureur['RàC']=dfassureur['FR']-dfassureur['R_SS']-dfassureur['RC']
    ##################################


    table=pd.pivot_table(dfassureur, values=['FR','RàC','RC','R_SS','nombre_acte'], index=[mesure], aggfunc=np.sum).reset_index()
    table['Remboursement complémentaire moyen']=round(table['RC']/table['nombre_acte'],2)
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


    table2=pd.pivot_table(dfassureur, values=[ID], index=[mesure], aggfunc=lambda x: len(x.unique())).reset_index()
    table=pd.merge(table,table2,on=mesure)
    # Correction si RàC <0
    table['FR']=np.where(table['RàC']<0,table['R_SS']+table['RC'],table['FR'])
    table['RàC']=table['FR']-table['RC']-table['R_SS']
    table.rename(columns={ID: "Nombre consommants",'FR':'Frais réels','RàC':'Reste à charge','R_SS':'Remboursement sécurité sociale','RC':'Remboursement complémentaire','nombre_acte':'Nombre actes'},inplace=True)

    TT = pd.DataFrame([[dfassureur[ID].nunique(),dfassureur['FR'].sum(),dfassureur['R_SS'].sum(),dfassureur['RC'].sum(),dfassureur['RàC'].sum(),dfassureur['nombre_acte'].sum(),(dfassureur['RC'].sum()/dfassureur['nombre_acte'].sum())]], columns=['Nombre consommants','FR','Remboursement sécurité sociale','Remboursement complémentaire','RàC','Nombre actes','Remboursement complémentaire moyen'], index=['Total']).reset_index().rename(columns={'index':mesure,'FR':'Frais réels','RàC':'Reste à charge'})

    TT=TT[[mesure,'Nombre actes','Frais réels','Remboursement complémentaire','Remboursement sécurité sociale','Reste à charge','Nombre consommants','Remboursement complémentaire moyen']]
    table=pd.concat([table,TT],axis=0)
    table['Taux de couverture']=(table['Remboursement complémentaire']+table['Remboursement sécurité sociale'])/table['Frais réels']
    table = table[[mesure,'Nombre consommants','Nombre actes','Remboursement complémentaire','Reste à charge','Taux de couverture','Remboursement complémentaire moyen']].reset_index(drop=True).fillna(0) # 'Frais réels','Remboursement sécurité sociale'
    tableAvantMiseEnforme=table.copy()

    table[['Nombre consommants','Nombre actes','Remboursement complémentaire','Reste à charge']]=table[['Nombre consommants','Nombre actes','Remboursement complémentaire','Reste à charge']].applymap(formatM) # 'Frais réels','Remboursement sécurité sociale'
    table=table.rename(columns={mesure:str(annee)})   
    table['Remboursement complémentaire moyen']=table['Remboursement complémentaire moyen'].apply(lambda x: '{:.2f} €'.format(x))

    table=table.style.format({'Taux de couverture': "{:.1%}"})

    table=format_table_Sousfamille(table,annee)
    st.dataframe(table)
    try:
        dfi.export(table, Emplacement_stockage+"/"+'table_détails_'+str(mesure)+'_'+str(Variable_bouclée)+'_'+str(annee)+'.jpg',dpi=100,table_conversion='chrome')
    except:
        print("erreur de la librairie dfi")
    return tableAvantMiseEnforme


def comparaison_sf_n_n_1(tn,tn_1,Emplacement_stockage,annee,mesure,Variable_bouclée):
    # Définir les index
    tn_idx = tn.set_index('Sous famille')
    tn_1_idx = tn_1.set_index('Sous famille')

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
        dfi.export(table, Emplacement_stockage+"/"+'table_détails_sf_n_n_1_'+str(mesure)+'_'+str(Variable_bouclée)+'_'+str(annee)+"_vs_"+str(annee-1)+'.jpg',dpi=100,table_conversion='chrome')
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
