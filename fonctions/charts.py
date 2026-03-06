import pandas as pd
import numpy as np
import streamlit as st
import re
from adjustText import adjust_text
import matplotlib.pyplot as plt # librairie visualisation de données (permet de faire des graphiques)
import seaborn as sns # librairie visualisation de données (permet de faire des beaux graphiques) https://seaborn.pydata.org/
import matplotlib.ticker as ticker # permet de modifier les axes des graphiques (légendes, labels, format..)
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter # permet de changer le format des données
import matplotlib.patches as mpatches
import circlify
import textwrap
from matplotlib.ticker import FuncFormatter
import os


import sys
sys.path.append(r"C:\Users\maxime.genet\Desktop\T\Mission R&D\Application santé\fonctions")
from fonctions.workOnData import formatM
from fonctions.workOnData import GetTypeBénéf
from fonctions.workOnData import Famille_acte_sorted
from fonctions.workOnData import format_string_with_linebreak
from fonctions.workOnData import format_value
from fonctions.workOnData import optimal_bins
from fonctions.workOnData import calculer_bins_labels_equilibres


# Légendes des futurs axes
labels=['Janv','Fev','Mars','Avr','Mai','Juin','Juil','Août','Sept','Oct','Nov','Déc']
labels_nbr=[1,2,3,4,5,6,7,8,9,10,11,12]


# Définir vos couleurs hexadécimales
colors = ["#4295CE", "#2C67AF", "#2B3885", "#2A0C53", "#662064", "#9B406D",'#D86173','#F56C26','#EE9744','#EE9780']

# Créer une palette Seaborn personnalisée
palette=sns.color_palette(colors)
#'Appareillage','Maternité'
#'Frais médicaux de ville','Consultations et visites'

color_map = {
    'Hospitalisation':palette[2],
    'Soins courants':palette[3],
    'Consultations et visites':palette[4],
    'Pharmacie':palette[5],
    'Optique':palette[0],
    'Dentaire':palette[1],
    'Divers':palette[6]}


def DispersionChart_year(df,Var,Famille,annee,qualitéGraphique,Emplacement_stockage,ID):
    dfannée=df[(df['annee_soins']==annee) & (df['annee_paiement']==annee)]
    TitleFamille=re.sub(r'[0-9]+. ', '', Famille)
    if Var=='RàC':
        IdGconsommateurs=pd.pivot_table(dfannée, values=['frais_reels','RC','rbt_ss'], index=[ID], 
                        aggfunc='sum', fill_value=0).reset_index()
        IdGconsommateurs['RàC']=IdGconsommateurs['frais_reels']-IdGconsommateurs['rbt_ss']-IdGconsommateurs['RC']
        IdGconsommateurs=IdGconsommateurs.sort_values(by=Var,ascending=False)
    else:
        IdGconsommateurs=pd.pivot_table(dfannée, values=[Var], index=[ID], 
                        aggfunc={Var : np.sum}, fill_value=0).sort_values(by=Var,ascending=False).reset_index()
    
    IdGconsommateurs=IdGconsommateurs[IdGconsommateurs[Var]>0]
    dataframe=IdGconsommateurs.copy()
    if dataframe.empty:
        return
    else:
        #Création d'un groupe par age 
        dataframe["Tranche de montant"] = 0
        for i in dataframe.index:
            if dataframe[Var][i]>1000:
                dataframe["Tranche de montant"][i] = "9"
            if dataframe[Var][i]<=1000:
                dataframe["Tranche de montant"][i] = "8"
            if dataframe[Var][i]<=600:
                dataframe["Tranche de montant"][i] = "7"
            if dataframe[Var][i]<=400:
                dataframe["Tranche de montant"][i] = "6"
            if dataframe[Var][i]<=300:
                dataframe["Tranche de montant"][i] = "5"
            if dataframe[Var][i]<=200:
                dataframe["Tranche de montant"][i] = "4"
            if dataframe[Var][i]<=150:
                dataframe["Tranche de montant"][i] = "3"
            if dataframe[Var][i]<=100:
                dataframe["Tranche de montant"][i] = "2"
            if dataframe[Var][i]<=50:
                dataframe["Tranche de montant"][i] = "1"
        dataframe["Tranche de montant"]=dataframe["Tranche de montant"].astype(int)
        intervalsTable=pd.DataFrame({'Tranche de montant':[1,2,3,4,5,6,7,8,9],'intervalsNames':["< 50 €","50-100 €","100-150 €","150-200 €","200-300 €","300-400 €","400-600 €","600-1000 €",">1000 €"]})
        table=pd.pivot_table(dataframe, values=[ID,Var], index=['Tranche de montant'],aggfunc= {ID:lambda x: len(x.unique()),Var:np.sum}).reset_index()
        table=table.merge(intervalsTable, on='Tranche de montant', how="left")
        
        fig = plt.figure(figsize=(15, 10))
        sns.set( style = "white", font_scale=1.5) 
        ax=sns.barplot(x="Tranche de montant", y=Var, data=table, palette='Blues')
        ax.set(xlabel ="",ylabel="")
        ax.yaxis.set_major_formatter(ticker.EngFormatter('€'))
        plt.xticks(table.index, table['intervalsNames'],rotation ='horizontal',fontsize=15)
        plt.yticks(fontsize=15)
        
        ax2=ax.twinx()
        ax2.plot(table.index, table[ID],color='darkorange',linewidth=3)
        ax2.set(xlabel ="",ylabel="")
        tkw = dict(size=15)
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Définir des valeurs entières sur l'axe des ordonnées de ax2
        ax2.tick_params(axis='y', labelsize=15)
        for bar in ax.patches:
            a=bar.get_height()
            value=int(table[table[Var]==a][ID][0:1])
            ax.annotate(formatM(value),(bar.get_x() + bar.get_width() / 2,
                                bar.get_height()), ha='center', va='bottom',
                            size=15, xytext=(0, 8),
                            textcoords='offset points')
        if Var=='RC':
            title='Dispersion des remboursements complémentaires en '+TitleFamille+' - '+str(annee)
            V='remboursement complémentaire'
        else:
            title='Dispersion des restes à charge en '+TitleFamille+' - '+str(annee)
            V='Reste à charge'
        plt.title(title, fontsize=18,fontweight='bold',pad=20)
        ax.legend([Patch(facecolor='b'),Patch(facecolor='tab:orange'),Patch(facecolor='w')],['Montant total de '+ V +' : '+str(formatM(dataframe[Var].sum()))+ ' euros','Nombre ' + GetTypeBénéf(ID)+' : '+str(formatM(dataframe[ID].nunique())), 'Montant de '+ V + ' moyen : '+str(formatM(dataframe[Var].sum()/dataframe[ID].nunique()))+' euros'], loc='upper center', fontsize=15, 
            fancybox=True, framealpha=0.7,bbox_to_anchor=(0.5, -0.05))


        plt.savefig(Emplacement_stockage+"/"+title+'.jpg',bbox_inches='tight',dpi=qualitéGraphique)
        #plt.show()

        st.pyplot(fig)


def PlotVentilationCouts(df_data, annee,qualitéGraphique,Emplacement_stockage,ID):    
    # Dernière modif pour gérer les làl avec une ligne base une ligne option (problème pour le RàC total)
    dfassureur= pd.pivot_table(df_data,values=['frais_reels','rbt_ss','RC'],index=[ID,'famille_acte_aops'],aggfunc='sum').reset_index()
    dfassureur=dfassureur[dfassureur['RC']>0]
    dfassureur['RàC']=dfassureur['frais_reels']-dfassureur['rbt_ss']-dfassureur['RC']
    ##################################

    Effectif=pd.pivot_table(dfassureur,values=[ID],index=['famille_acte_aops'], aggfunc=lambda x: len(x.unique()))
    Effectif=Effectif.reindex(Famille_acte_sorted(df_data))

    total_row = pd.DataFrame({ID: [dfassureur[ID].nunique()]}, index=['Total'])
    Effectif = pd.concat([Effectif,total_row], ignore_index=False)
    table=pd.pivot_table(dfassureur,values=['rbt_ss','RC','RàC'],index=['famille_acte_aops'], aggfunc=np.sum).round(2)#.round(2) permet d'arrondire les valeurs de la table à 2 chiffres après la virgule
    table=table.reindex(Famille_acte_sorted(df_data))
    total_row = pd.DataFrame({'rbt_ss': [dfassureur['rbt_ss'].sum()], 'RC': [dfassureur['RC'].sum()], 'RàC': [dfassureur['RàC'].sum()]}, index=['Total'])
    table = pd.concat([table,total_row], ignore_index=False)
    table['rbt_ss']=table['rbt_ss']/Effectif[ID]
    table['RC']=table['RC']/Effectif[ID]
    table['RàC']=table['RàC']/Effectif[ID]
    table['total']=table['rbt_ss']+table['RC']+table['RàC']
    table[table < 0] = 0
    table=table[['rbt_ss','RC','RàC','total']]
    stacked_data = table.drop(columns=['total']).apply(lambda x: x*100/sum(x), axis=1).round(2)
    stacked_data.rename(columns={'rbt_ss':'Remboursement Sécurité Sociale','RC':'Remboursement complémentaire','RàC':'Reste à charge'}, inplace=True)
    sns.set(rc={"figure.figsize":(12, 6)}) # taille graphique
    sns.set_style("whitegrid")
    ax=stacked_data.plot(kind='bar', stacked=True,color=[palette[4],palette[5], palette[6]]) # coeur du graphique 
    plt.legend(loc='lower center', borderaxespad=-7,ncol=len(stacked_data.columns)) # legendes dans le cadre, position, nombre de colonnes, taille
    title='Répartition des dépenses de santé et coût moyen  '+ str(annee)+ '  par famille d\'actes'
    plt.title(title+'\n', fontsize=21,fontname="Calibri",fontweight="bold") 
    # Taille texte axe x, y 
    #plt.yticks(fontsize=16)
    #plt.xticks(fontsize=16)
    plt.xticks(rotation= 10) # inclinaison texte (degrés)

    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}%')) # Format texte legende axe y
    ax.set(xlabel ="",ylabel="") # titre axes x, y

    rect = mpatches.Rectangle((0, 0), 1, 1, fc="#2A0C53", alpha=1)

    # Partie du code permettant de positionner les valeurs moyennes et le total au sommet
    columns=table.columns
    index=table.index
    i=0
    j=0
    for p in ax.patches: 
        # moyenne
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        if table[columns[j]][index[i]]!=0:
            ax.text(x+width/2, 
                    y+height/2, 
                    formatM(table[columns[j]][index[i]])+"€",
                    #"{:,.2f}€".format(table[columns[j]][index[i]]), # format text
                    horizontalalignment='center', # position par rapport à la position visée
                    verticalalignment='center', # position par rapport à la position visée
                color='white',fontweight='bold') # couleur, taille et style texte
        if (j==2): 
            # total
            ax.annotate(formatM(table['total'][i])+"€",
                #"{:.2f} €".format(table['total'][i], '.4f'),
                        (p.get_x() + p.get_width() / 2,
                            100), ha='center', va='center',
                         xytext=(0, 14),color='white',
                        textcoords='offset points',fontweight='bold',
                       # bbox ={'facecolor':'gold','alpha':0.7, 'pad':2})
            bbox=dict(boxstyle="square,pad=0.3", fc=rect.get_facecolor(), alpha=rect.get_alpha()))
        if i==len(table)-1:
            i=0
            j=j+1
        else:
            i=i+1
            j=j
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.grid(True, linestyle='--', alpha=0.3,color='grey')

    plt.savefig(Emplacement_stockage+"/"+title+'.jpg',bbox_inches='tight',dpi=qualitéGraphique)

    st.pyplot(plt)


def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal
def distributionFamilleActes(df,annee,Emplacement_stockage,qualitéGraphique):
    
    pv = pd.pivot_table(df[df['annee_soins']==annee], index='famille_acte_aops', values='RC', aggfunc='sum').reset_index()
    pv['taux'] = pv['RC'] / pv['RC'].sum()

    pal_vi = get_color(palette, len(pv))

    rcs = pv.sort_values(by='RC',ascending=False)
    # Compute circle positions using circlify
    circles = circlify.circlify(rcs['RC'].tolist(), show_enclosure=False, target_enclosure=circlify.Circle(x=0, y=0))
    circles.reverse()
    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.axis('off')
    lim = max(max(abs(circle.x) + circle.r, abs(circle.y) + circle.r) for circle in circles)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    # Print circles with labels including percentage
    for circle, label, rc, color in zip(circles, rcs['famille_acte_aops'], rcs['RC'], pal_vi):
        x, y, r = circle
        ax.add_patch(plt.Circle((x, y), r, alpha=0.9, color=color))
        # Calculate percentage
        percentage = rc / rcs['RC'].sum() * 100

        # Concatenate percentage to label
        label_with_percentage = f'{format_string_with_linebreak(label)}\n \n{format_value(rc)}€ \n \n {percentage:.0f}%'

        if r < 0.05:
            fontsize=6
        elif r < 0.1:
            fontsize=7
        elif r < 0.15:
            fontsize=8
        elif r < 0.2:
            fontsize=9
        elif r < 0.25:
            fontsize=10
        else:
            fontsize=11


        plt.annotate(label_with_percentage, (x, y), size=fontsize, va='center', ha='center', weight='bold',color='white')
    # Set title
    title=f"Distribution des actes {annee}"
    plt.title(title,weight='bold')
    plt.savefig(f"{Emplacement_stockage}/_{title}_.jpg",bbox_inches='tight',dpi=qualitéGraphique)

    st.pyplot(plt)


def Evo_Cons_Moyenne(df, qualitéGraphique, Emplacement_stockage, ID):
    df['annee_soins'] = df['annee_soins'].fillna(0).astype(int)
    nb_survenances = df['annee_soins'].nunique()

    if nb_survenances < 2:
        print("Pas assez de survenances pour calculer une évolution.")
        return

    # -----------------------------
    # Construction des tables RC et Effectifs
    # -----------------------------
    table = pd.pivot_table(
        df, values='RC', index=['famille_acte_aops'],
        columns='annee_soins', aggfunc=np.sum, fill_value=0
    ).reindex(Famille_acte_sorted(df))

    tableEff = pd.pivot_table(
        df, values=ID, index='famille_acte_aops',
        columns='annee_soins', aggfunc=pd.Series.nunique
    ).reindex(Famille_acte_sorted(df))

    # On restreint aux dernières survenances
    if nb_survenances >= 3:
        table = table[table.columns[-3:]]
        tableEff = tableEff[tableEff.columns[-3:]]
    else:  # si 2 survenances
        table = table[table.columns[-2:]]
        tableEff = tableEff[tableEff.columns[-2:]]

    # Ajout ligne Total
    df_tot = pd.DataFrame(
        pd.pivot_table(df[df['annee_soins'] >= table.columns.min()],
                       values='RC', columns='annee_soins', aggfunc='sum').values,
        columns=table.columns, index=['Total']
    )
    table = pd.concat([table, df_tot])

    df_eff_tot = pd.DataFrame(
        pd.pivot_table(df[df['annee_soins'] >= table.columns.min()],
                       values=ID, columns='annee_soins', aggfunc=pd.Series.nunique).values,
        columns=tableEff.columns, index=['Total']
    )
    tableEff = pd.concat([tableEff, df_eff_tot])

    # -----------------------------
    # Calcul de la consommation moyenne
    # -----------------------------
    for col in table.columns:
        table[col] = table[col] / tableEff[col]

    # -----------------------------
    # Calcul des évolutions
    # -----------------------------
    evol_cols = []
    if nb_survenances >= 3:
        col0, col1, col2 = table.columns
        table[f"{col1}/{col0}"] = ((table[col1] - table[col0]) / table[col0]) * 100
        table[f"{col2}/{col0}"] = ((table[col2] - table[col0]) / table[col0]) * 100
        table[f"{col2}/{col1}"] = ((table[col2] - table[col1]) / table[col1]) * 100
        evol_cols = [f"{col1}/{col0}", f"{col2}/{col0}", f"{col2}/{col1}"]
    else:  # seulement 2 survenances
        col0, col1 = table.columns
        table[f"{col1}/{col0}"] = ((table[col1] - table[col0]) / table[col0]) * 100
        evol_cols = [f"{col1}/{col0}"]

    # -----------------------------
    # Nettoyage Divers
    # -----------------------------
    if 'Divers' in table.index:
        table = table.drop(index='Divers')

    # -----------------------------
    # Graphique
    # -----------------------------
    sns.set(rc={"figure.figsize": (20, 8)})
    sns.set(style="whitegrid")
    ax = table[evol_cols].plot.bar(stacked=False, color=palette[:len(evol_cols)])

    ax.legend(title='Survenances')
    ax.set_xticklabels(table.index)
    ax.set(xlabel="", ylabel="")
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14, rotation=00)

    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}%'))

    ymin, ymax = ax.get_ylim()
    padding = (ymax - ymin) * 0.2
    ax.set_ylim(ymin - padding, ymax + padding)

    # Placement des labels %
    for bar in ax.patches:
        val = bar.get_height()
        if val >= 0:
            ax.annotate(f"+{val:.0f}%", (bar.get_x() + bar.get_width() / 2,
                                         val + padding * 0.25), ha='center',
                        size=12, xytext=(0, 8),
                        textcoords='offset points', rotation=0)
        else:
            ax.annotate(f"{val:.0f}%", (bar.get_x() + bar.get_width() / 2,
                                        val - padding * 0.35), ha='center',
                        size=12, xytext=(0, 8),
                        textcoords='offset points', rotation=0)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.grid(True, linestyle='--', alpha=0.3, color='grey')

    survenance = sorted(df['annee_soins'].unique())
    title = f"Evolution de la consommation moyenne par consommant des survenances {survenance[0]} à {survenance[-1]}"
    plt.title(title + '\n', fontsize=20)
    plt.legend(loc='lower center', borderaxespad=-7, fontsize=16, ncol=len(evol_cols))

    plt.savefig(Emplacement_stockage+"/" + title + '.jpg', bbox_inches='tight', dpi=qualitéGraphique)
    st.pyplot(plt)


def EVO_Consommateurs(df,qualitéGraphique,Emplacement_stockage,ID):
    df['annee_soins']=df['annee_soins'].fillna(0).astype(int)
    fig = plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax=sns.lineplot(data=pd.pivot_table(df,values=[ID], index=['annee_soins','mois_soins'],aggfunc=lambda x: len(x.unique())).reset_index(), x="mois_soins", y=ID, hue="annee_soins",hue_order = sorted(df['annee_soins'].unique(),reverse=True),palette=palette[::3][0:len(df['annee_soins'].unique())],linewidth = 3)
    plt.xticks(np.unique(df['mois_soins']).tolist(), labels[int(df['mois_soins'].min())-1:int(df['mois_soins'].max())])
    ax.legend(loc='best', ncol=1)
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    #ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    #plt.savefig('images/linesplot.jpg', bbox_inches='tight', dpi=150)
    title='Evolution mensuelle du nombre de consommants par survenance'
    plt.title(title,fontname="Calibri",fontweight="bold", fontsize=20)
    ax.set_xlabel('')
    ax.set_ylabel('Nombre de consommants') 

    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.grid(True, linestyle='--', alpha=0.3, color='grey')   

    if df['annee_soins'].nunique()>1: # commentaire en bas de graphique
        value=round((df[df['annee_soins']==sorted(df['annee_soins'].unique())[-1]][ID].nunique()/df[df['annee_soins']==sorted(df['annee_soins'].unique())[-2]][ID].nunique()-1)*100,2)
        commentaire= 'En '+str(sorted(df['annee_soins'].unique())[-1])+', le nombre de consommants a évolué de '+str(value)+'% par rapport à '+ str(sorted(df['annee_soins'].unique())[-2])

        ax = plt.gca()
        ax.set_ylim(bottom=ax.get_ylim()[0] - 0.2*(ax.get_ylim()[1] - ax.get_ylim()[0]))
        plt.text(0.5, -0.15, commentaire, fontsize=14, ha='center', va='center', transform=ax.transAxes,
                bbox=dict(facecolor='orange', edgecolor='black', boxstyle='round', linewidth=1, pad=.5),
                weight='bold')


    plt.savefig(Emplacement_stockage+"/"+title+'.jpg',bbox_inches='tight',dpi=qualitéGraphique)

    st.pyplot(fig)

def EVO_Remboursement_moy(df,var,qualitéGraphique,Emplacement_stockage,ID):
    df['annee_soins']=df['annee_soins'].fillna(0).astype(int)
    table=pd.pivot_table(df,values=[ID,var], index=['annee_soins','mois_soins'],aggfunc={ID:lambda x: len(x.unique()),var:np.sum}).reset_index()
    table['Moyenne remboursement OAR par consommant']=table[var]/table[ID]

    fig = plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax=sns.lineplot(data=table, x="mois_soins", y="Moyenne remboursement OAR par consommant", hue="annee_soins",hue_order = sorted(df['annee_soins'].unique(),reverse=True),palette=palette[::3][0:len(df['annee_soins'].unique())],linewidth = 3)
    plt.xticks(np.unique(df['mois_soins']).tolist(), labels[int(df['mois_soins'].min())-1:int(df['mois_soins'].max())])
    ax.legend(loc='best', ncol=1)
    #ax.yaxis.set_major_formatter(ticker.EngFormatter())
    ax.yaxis.set_major_formatter(ticker.EngFormatter('€'))
    if var=="RC":
        name="remboursement complémentaire"
        ax.set_ylabel("Remboursement complémentaire moyen")
    elif var=='RàC':
        name='RàC'
        ax.set_ylabel("RàC moyen")
    elif var=='rbt_ss':
        name='remboursement sécurité sociale'
        ax.set_ylabel("Remboursement sécurité sociale moyen")
    title='Evolution mensuelle du '+name +' moyen par consommant, par survenance'
    plt.title(title,fontname="Calibri",fontweight="bold", fontsize=20)

    ax.set_xlabel('')

    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.grid(True, linestyle='--', alpha=0.3, color='grey')
    
    Moy_annee1=df[df['annee_soins']==sorted(df['annee_soins'].unique())[-1]][var].sum()/df[df['annee_soins']==sorted(df['annee_soins'].unique())[-1]][ID].nunique()
    Moy_annee2=df[df['annee_soins']==sorted(df['annee_soins'].unique())[-2]][var].sum()/df[df['annee_soins']==sorted(df['annee_soins'].unique())[-2]][ID].nunique()
    
    if df['annee_soins'].nunique()>1: # commentaire en bas de graphique

        Moy_annee1=df[df['annee_soins']==sorted(df['annee_soins'].unique())[-1]][var].sum()/df[df['annee_soins']==sorted(df['annee_soins'].unique())[-1]][ID].nunique()
        Moy_annee2=df[df['annee_soins']==sorted(df['annee_soins'].unique())[-2]][var].sum()/df[df['annee_soins']==sorted(df['annee_soins'].unique())[-2]][ID].nunique()
    
        value=round(((Moy_annee1/Moy_annee2)-1)*100,2)
        commentaire= 'En '+str(sorted(df['annee_soins'].unique())[-1])+', la consommation moyenne par bénéficiaire a évolué de '+str(value)+'% par rapport à '+ str(sorted(df['annee_soins'].unique())[-2])

        ax = plt.gca()
        ax.set_ylim(bottom=ax.get_ylim()[0] - 0.2*(ax.get_ylim()[1] - ax.get_ylim()[0]))
        plt.text(0.5, -0.15, commentaire, fontsize=14, ha='center', va='center', transform=ax.transAxes,
                bbox=dict(facecolor='orange', edgecolor='black', boxstyle='round', linewidth=1, pad=.5),
                weight='bold')


    plt.savefig(Emplacement_stockage+"/"+title+'.jpg',bbox_inches='tight',dpi=qualitéGraphique)

    st.pyplot(fig)

def Evo_RC(df,qualitéGraphique,Emplacement_stockage):

    # Ensure the 'Famille_acte_sorted' function is defined elsewhere in your code
    ordre = Famille_acte_sorted(df)  # Ensure this function is defined
    if 'Divers' in ordre:
        ordre.remove('Divers')
    # Fill NaN values in 'annee_soins' with 0 and convert to int using .loc
    df['annee_soins'] = df['annee_soins'].fillna(0).astype(int) 

    # Create a pivot table with the sum of 'RC' per 'Famille acte' and 'annee_soins'
    # Use 'sum' as a string to avoid the FutureWarning
    table = pd.pivot_table(
        df,
        values='RC',
        index=['famille_acte_aops', 'annee_soins'],
        aggfunc='sum',  # Use 'sum' as a string
        fill_value=0
    ).reset_index()

    # Calculate the percentage of 'RC' for each 'Famille acte' within each 'annee_soins'
    # Use groupby and transform to calculate the sum and avoid loop
    table['% sur l\'année T'] = table.groupby('annee_soins')['RC'].transform(lambda x: x / x.sum())
    table=table[table['famille_acte_aops']!='Divers']
    
    fig = plt.figure(figsize=(12, 6))
    sns.set( style = "whitegrid" )   # apparence du font du graphique
    ax=sns.barplot(x="RC", y="famille_acte_aops", hue='annee_soins', data=table, palette=palette[0:len(table['annee_soins'].unique())][::-1],order=ordre) # graphique construit à partir de la table précédente
    if df['annee_soins'].nunique()==1:
        title='Evolution du remboursement complémentaire pour les principaux postes et poids dans la survenances '+str(int(table['annee_soins'].unique()))

    else:
        title='Evolution du remboursement complémentaire pour les principaux postes et poids dans les exercices de survenances de '+str(table['annee_soins'].min()) + ' à ' + str(table['annee_soins'].max())
    
    plt.title(title,fontname="Calibri",fontweight="bold") # titre, taille, style police, gras/italique/normale
    handles, labels = plt.gca().get_legend_handles_labels()
    order=list(range(len(handles)))[::-1]
    plt.legend([handles[i] for i in order], [labels[i] for i in order])
    #ax.legend(bbox_to_anchor=(1, 1),ncol = 1,fontsize=25) # position cadre, nombre de colonne, taille police
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,}€')) # format légende x
    ax.xaxis.set_major_formatter(ticker.EngFormatter('€'))

    ax.set(xlabel ="",ylabel="") # titre axes x,y

    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.grid(True, linestyle='--', alpha=0.3, color='grey')


    # Légendes bars
    i=0
    for bar in ax.patches:
        a=bar.get_width()
        try:
            value=table.loc[table['RC'] == a, '% sur l\'année T'].values[0] * 100
            ax.annotate("{:.1f}%".format(value),
                            (bar.get_width(), bar.get_y()+bar.get_height()/2), ha='left', 
                            va='top', 
                            xytext=(1, 5),
                            textcoords='offset points')
        except:
            print(bar)
    i=i+1
    plt.savefig(Emplacement_stockage+"/"+title+'.jpg',bbox_inches='tight',dpi=qualitéGraphique)

    st.pyplot(fig)


def EVO_Montant(df,var,qualitéGraphique,Emplacement_stockage):
    df['annee_soins']=df['annee_soins'].fillna(0).astype(int)
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(12, 6))
    ax=sns.lineplot(data=df[['mois_soins','annee_soins',var]].groupby(['annee_soins','mois_soins']).sum().reset_index(), x="mois_soins", y=var, hue="annee_soins",hue_order = sorted(df['annee_soins'].unique(),reverse=True),palette=palette[::3][0:len(df['annee_soins'].unique())],linewidth = 3)
    #plt.xticks(np.unique(df['mois_soins']).tolist(), labels=list(range(df['mois_soins'].min()-1,df['mois_soins'].max())))
    plt.xticks(np.unique(df['mois_soins']).tolist(), labels[int(df['mois_soins'].min())-1:int(df['mois_soins'].max())])
    #ax.set(ylim = (20000,500000))
    ax.legend(loc='best', ncol=1)
    ax.yaxis.set_major_formatter(ticker.EngFormatter('€'))
    #ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,}€'))
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    # Commentaire 

    if var=="RC":
        name="remboursement complémentaire"
        ax.set_ylabel("Remboursement complémentaire")
    elif var=='RàC':
        name='RàC'
        ax.set_ylabel("Reste à charge")
    elif var=='rbt_base':
        name='remboursement sécurité sociale'
        ax.set_ylabel("Remboursement sécurité sociale")
    else:
        name=var
    title='Evolution mensuelle du '+name +' par survenance'
    plt.title(title, fontsize=20,fontname="Calibri",fontweight="bold")
    ax.set_xlabel('')

    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.grid(True, linestyle='--', alpha=0.3, color='grey')

    if df['annee_soins'].nunique()>1: # commentaire en bas de graphique
        value=round((df[df['annee_soins']==sorted(df['annee_soins'].unique())[-1]][var].sum()/df[df['annee_soins']==sorted(df['annee_soins'].unique())[-2]][var].sum()-1)*100,2)
        commentaire= 'En '+str(sorted(df['annee_soins'].unique())[-1])+', le '+name+' a évolué de '+str(value)+'% par rapport à '+ str(sorted(df['annee_soins'].unique())[-2])
        ax = plt.gca()
        ax.set_ylim(bottom=ax.get_ylim()[0] - 0.2*(ax.get_ylim()[1] - ax.get_ylim()[0]))
        plt.text(0.5, -0.15, commentaire, fontsize=14, ha='center', va='center', transform=ax.transAxes,
                bbox=dict(facecolor='orange', edgecolor='black', boxstyle='round', linewidth=1, pad=.5),
                weight='bold')
    


    plt.savefig(Emplacement_stockage+"/"+title+'.jpg',bbox_inches='tight',dpi=qualitéGraphique)

    st.pyplot(fig)


def Panier_plot(d, ID, PanierVar, titre, qualitéGraphique, Emplacement_stockage):

    
    # =========================
    # Préparation des données
    # =========================
    d[PanierVar] = d[PanierVar].str.title()
    #d[PanierVar] = d[PanierVar].replace({'maîtrisés':'Maîtrisés','libre':'Libre'})
    
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4))

    # =========================
    # Fonction auto-fontsize
    # =========================
    def auto_fontsize(fig, base=2.2, dpi_ref=100, dpi=None, factor=1.0):
        h = fig.get_size_inches()[1]
        fs = h * base
        if dpi:
            fs *= dpi / dpi_ref
        return fs * factor

    base_fontsize = auto_fontsize(fig, dpi=qualitéGraphique)

    # =========================
    # Table pivot
    # =========================
    table = pd.pivot_table(
        d,
        values=[ID, 'nb_acte', 'RC'],
        index=['annee_soins', PanierVar],
        aggfunc={ID: 'nunique', 'nb_acte': 'sum', 'RC': 'sum'}
    ).reset_index()

    table = table.rename(columns={'annee_soins': 'Année de survenance'})
    table['RC moyen'] = round(table['RC'] / table['nb_acte'], 2)

    # =========================
    # Palette dynamique
    # =========================
    unique_vals = table[PanierVar].nunique()
    pal = palette[::3][:unique_vals]

    # =========================
    # Scatter plot
    # =========================
    sns.scatterplot(
        data=table,
        x="RC moyen",
        y="nb_acte",
        hue=PanierVar,
        palette=pal,
        size="RC",
        sizes=(1000, 10000),
        legend=False,
        ax=ax
    )

    # =========================
    # Axes
    # =========================
    ax.yaxis.set_major_formatter(ticker.EngFormatter(''))
    ax.xaxis.set_major_formatter(ticker.EngFormatter('€'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.tick_params(labelsize=base_fontsize* 1.2)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.set_xlim(x_min - (x_max - x_min) * 0.25, x_max + (x_max - x_min) * 0.25)
    ax.set_ylim(y_min - (y_max - y_min) * 0.25, y_max + (y_max - y_min) * 0.25)

    # =========================
    # Taille des labels (densité)
    # =========================
    n_points = len(table)
    density_factor = max(0.7, min(1.0, 8 / n_points))
    label_fontsize = base_fontsize * density_factor

    # =========================
    # Labels des bulles
    # =========================
    text_positions = []

    for i in range(len(table)):
        x = table["RC moyen"].iloc[i]
        y = table["nb_acte"].iloc[i]

        text = (
            f"{table['Année de survenance'][i]}\n"
            f"{table[PanierVar][i]}\n"
            f"RC : {formatM(table['RC'][i])} €"
        )

        label = ax.text(
            x, y, text,
            ha='center',
            va='center',
            fontsize=label_fontsize,
            fontweight='bold',
            bbox=dict(
                boxstyle='round',
                fc='white',
                ec='black',
                alpha=0.7
            )
        )

        text_positions.append(label)

    # =========================
    # Ajustement des chevauchements
    # =========================
    fig.canvas.draw()

    adjust_text(
        text_positions,
        ax=ax,
        expand=(1.05, 1.2),
        arrowprops=dict(arrowstyle="-", lw=0),
        force_points=(0.3, 0.5),
        force_text=(0.5, 0.5),
        only_move={'points': 'y', 'text': 'xy'}
    )

    # =========================
    # Style final
    # =========================
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(True, linestyle='--', alpha=0.3, color='grey')

    ax.set_xlabel(
        'Remboursement complémentaire moyen',
        fontsize=base_fontsize* 1.2
    )

    ax.set_ylabel(
        "Nombre d'actes",
        fontsize=base_fontsize* 1.2
    )

    ax.set_title(
        f"100% santé - {titre}",
        fontsize=base_fontsize * 1.5,
        fontname="Calibri"
    )

    plt.tight_layout()

    # =========================
    # Sauvegarde
    # =========================
    plt.savefig(
        f"{Emplacement_stockage}/{titre}.jpg",
        dpi=qualitéGraphique,
        bbox_inches='tight'
    )

    st.pyplot(fig)


def Panier_plot_ventilation(d, ID, PanierVar, titre, qualitéGraphique, Emplacement_stockage):
    
    d[PanierVar] = d[PanierVar].str.title()
    # Table initiale
    table1 = pd.pivot_table(
        d, values=[ID, 'nb_acte', 'RC'], 
        index=['annee_soins', PanierVar,'sous_famille'], 
        aggfunc={ID: pd.Series.nunique, 'nb_acte': 'sum', 'RC': 'sum'}
    ).reset_index()

    table= pd.pivot_table(
            d, values=[ID, 'nb_acte', 'RC'], 
            index=['annee_soins', PanierVar], 
            aggfunc={ID: pd.Series.nunique, 'nb_acte': 'sum', 'RC': 'sum'}
        ).reset_index()
    
    res=pd.merge(table1[['annee_soins',PanierVar,'RC','sous_famille']],table[['annee_soins',PanierVar,'RC']],on=['annee_soins',PanierVar],how='left')
    res['taux'] = (res['RC_x'] / res['RC_y'].replace(0, np.nan) * 100).round(2)
    res['taux'] = res['taux'].fillna(0)
    res[["annee_soins", PanierVar,"sous_famille","taux"]]

    ### Table des contenant les montants d'utilisation de sf par panier
    df_m = res[["annee_soins", 	PanierVar,"sous_famille","RC_x"]]
    # Pivoter le DataFrame pour faciliter le traçage
    pivot_df_m = df_m.pivot(index=[PanierVar,'annee_soins'], columns=['sous_famille'], values='RC_x').fillna(0)

    ### Table des contenant les pourcentage d'utilisation de sf par panier
    df = res[["annee_soins", 	PanierVar,"sous_famille","taux"]]
    # Pivoter le DataFrame pour faciliter le traçage
    pivot_df = df.pivot(index=[PanierVar,'annee_soins'], columns=['sous_famille'], values='taux').fillna(0)

    if (PanierVar=='sante_100') and ('Dentaire' in d['famille_acte_aops'].unique()):
        ordre_categorie = ['100% Santé', 'Maîtrisés', 'Libre']
        # Convertir le niveau 'categorie' en catégorie ordonnée
        # ----- pivot_df (taux)
        pivot_df.index = pd.MultiIndex.from_arrays([
            pd.Categorical(pivot_df.index.get_level_values('sante_100'), categories=ordre_categorie, ordered=True),
            pivot_df.index.get_level_values('annee_soins')
        ], names=pivot_df.index.names)
        
        pivot_df = pivot_df.sort_index()


        # ----- pivot_df_m (montants)
        pivot_df_m.index = pd.MultiIndex.from_arrays([
            pd.Categorical(
                pivot_df_m.index.get_level_values(PanierVar),
                categories=ordre_categorie,
                ordered=True
            ),
            pivot_df_m.index.get_level_values('annee_soins')
        ], names=pivot_df_m.index.names)
    
        pivot_df_m = pivot_df_m.sort_index()

    # Tracer un graphique à barres empilées
    sns.set_style("whitegrid")
    ax = pivot_df.plot.bar(stacked=True, figsize=(10, 3), color=palette[:len(pd.unique(pivot_df.index.get_level_values(PanierVar)))])

    # Personnalisation de l'axe des abscisses et de l'axe des ordonnées
    ax.set_xlabel('')  # Titre de l'axe des abscisses
    ax.set_ylabel('Taux', fontweight='bold')  # Titre de l'axe des ordonnées
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}%'))  # Format texte légende axe Y

    # Automatisation des labels de l'axe des abscisses
    # Extraire les années à partir de l'index
    new_labels = [str(x[1]) for x in pivot_df.index]
    ax.set_xticklabels(new_labels, rotation=0, fontweight='normal')

    # Automatisation des annotations
    # Extraire les sous-familles de l'index pour les annotations
    sous_familles  = list(pd.unique(pivot_df.index.get_level_values(PanierVar)))

    paniers = pivot_df.index.get_level_values(PanierVar).unique()

    for panier in paniers:
        
        # positions des barres correspondant à ce panier
        pos = [i for i, x in enumerate(pivot_df.index.get_level_values(PanierVar)) if x == panier]
        
        # centre du groupe
        center = sum(pos) / len(pos)
        
        ax.text(
            center,
            -20,              # position verticale sous l'axe
            panier,
            ha='center',
            va='top',
            fontsize=10,
            fontweight='normal'
        )

    col=0
    line=0
    for p in ax.patches: 
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        if pivot_df_m.iloc[line,col]!=0 and pivot_df.iloc[line,col]>=8:
            ax.text(x+width/2, 
                    y+height/2, 
                    f"{formatM(pivot_df_m.iloc[line,col])}€", # format text
                    horizontalalignment='center', # position par rapport à la position visée
                    verticalalignment='center', # position par rapport à la position visée
                color='white',fontweight='bold',fontsize=9) # couleur, taille et style texte
        if line==len(pivot_df)-1:
            line=0
            col=col+1
        else:
            line=line+1
            col=col
        
    for spine in plt.gca().spines.values():
            spine.set_visible(False)

    plt.grid(True, linestyle='--', alpha=0.3,color='grey')

    # Ajouter un titre et personnalisation de la légende
    plt.title(f"Taux par panier et sous famille", fontweight='bold')
    plt.legend(loc='lower center', borderaxespad=-5, ncol=3)  # Ajuster la légende

    # Assurer un espacement correct
    plt.tight_layout()

    # Sauvegarde et affichage
    plt.savefig(Emplacement_stockage+"/" + titre + 'Ventilation_coûts_.jpg', bbox_inches='tight', dpi=qualitéGraphique)
    st.pyplot(plt)


def Sous_famille_comparaison_montants(data, var,qualitéGraphique, Emplacement_stockage):

    df = data[(data[var] > 0)].groupby(['annee_soins', var]).size().reset_index(name='count')

    bins,labels=optimal_bins(data[data[var]>0][var], min_bin_size=0.05, max_bin_size=0.2, initial_bins=20)

    max_val=max(bins)
    min_val=min(bins)

    # Création de la colonne de tranches (y compris pour les valeurs aberrantes)
    df['tranche montant'] = pd.cut(df[var], bins=bins, labels=labels, right=False)

    # Ajouter une catégorie pour les valeurs extrèmes
    if len(df[df[var]<max_val])>0: 
        df['tranche montant'] = df['tranche montant'].cat.add_categories([f">{int(max_val)}"])
        df.loc[df[var] > max_val, 'tranche montant'] = f">{int(max_val)}"
    if len(df[df[var]<min_val])>0:
        df['tranche montant'] = df['tranche montant'].cat.add_categories([f"{int(min_val)}<"])
        df.loc[df[var] < min_val, 'tranche montant'] = f"{int(min_val)}<"


    # Calculer le nombre d'occurrences pour chaque tranche et année
    df_grouped = df.groupby(['tranche montant', 'annee_soins'])['count'].sum().reset_index()

    df_grouped=df_grouped.rename(columns={"annee_soins":'Année de survenance'})

    df_grouped=df_grouped[df_grouped['count']!=0]
    # Créer le graphique
    fig=plt.figure(figsize=(10, 5))
    ax=sns.barplot(df_grouped, x="tranche montant", y="count", hue="Année de survenance",palette=palette[::3][0:len(df_grouped['Année de survenance'].unique())],hue_order = sorted(df_grouped['Année de survenance'].unique(),reverse=True))

    ax.yaxis.set_major_formatter(ticker.EngFormatter(''))

    plt.xlabel(f"Tranches de montants en € - {var}")
    plt.ylabel('Nombre d\'occurrences')
    titre=f"Histogramme des montants par tranche - {data['sous_famille'].unique()[0]} - {var}"
    plt.title(titre)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Sauvegarde et affichage
    plt.savefig(Emplacement_stockage+"/" + titre + '.jpg', bbox_inches='tight', dpi=qualitéGraphique)
    st.pyplot(fig)


def etude_composante_dépense(data, variable_prix, sf,Emplacement_stockage,qualitéGraphique, layout='vertical'):
    data['annee_soins'] = data['annee_soins'].astype(int).astype(str)

    if 'nb_acte' not in data.columns:
        print('Variable "nb_acte" manquante')
        return
    else:
        # Agrégations
        inter = pd.pivot_table(
            data,
            values=[variable_prix, 'nb_acte'],
            index=['id_beneficiaire', 'annee_soins'],
            aggfunc='sum'
        ).reset_index()

        inter_bis = pd.pivot_table(
            inter,
            values=[variable_prix, 'nb_acte'],
            index='annee_soins',
            aggfunc='mean'
        )

        inter_ = pd.pivot_table(
            inter,
            values=[variable_prix, 'nb_acte'],
            index='annee_soins',
            aggfunc='sum'
        )
        inter_['prix_actes'] = inter_[variable_prix] / inter_['nb_acte']
        df_graph = pd.concat([inter_bis, inter_[['prix_actes']]], axis=1).reset_index()

        # Nom variable
        if variable_prix == 'RC':
            nom_var = 'remboursement complémentaire'
        elif variable_prix == 'rbt_ss':
            nom_var = 'remboursement sécurité sociale'
        elif variable_prix == 'frais_reels':
            nom_var = 'Frais réels'
        else:
            nom_var = variable_prix


        texte_intro = f"{sf} - Évolution des composantes de la dépense -"
        texte_intro_coupé = "\n".join(textwrap.wrap(texte_intro, width=60 if layout == 'vertical' else 300))
        # Mise en gras via LaTeX (avec espace visible)
        nom_var_affiche = nom_var.replace(" ", r"\ ")

        # Titre final à afficher
        if layout == 'vertical':
            titre_coupé = texte_intro_coupé + "\n" + r"$\bf{" + nom_var_affiche + "}$"
        else:
            titre_coupé = texte_intro_coupé + r" $\bf{" + nom_var_affiche + "}$"

        nom_fichier = f"{sf} - Evolution des composantes de la dépense - {nom_var}"

        # Choix du layout
        if layout == 'horizontal':
            fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
        else:
            fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

        # Liste des titres, données, couleurs et styles
        infos = [
            ("Coût moyen / personne (€)", variable_prix, '#2B3885', 'o', '-'),
            ("Coût moyen / acte (€)", 'prix_actes', '#D86173', 's', '-'),
            ("Nombre d'actes moyen / personne", 'nb_acte', '#EE9744', '^', '-')
        ]

        for ax, (titre_graph, col, couleur, marker, style) in zip(axes, infos):
            ax.plot(df_graph['annee_soins'], df_graph[col], marker=marker, color=couleur, linestyle=style)
            ax.set_title(titre_graph)
            ax.set_facecolor('white')
            ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
            ax.tick_params(labelsize=11)# 👈 taille des ticks
            ax.set_ylabel("Montant (€)" if "€" in titre_graph else "Unités")
            for spine in ax.spines.values():  # 👈 supprime les cadres
                spine.set_visible(False)

        axes[-1].set_xlabel("Survenance")

        fig.suptitle(titre_coupé)

        plt.figtext(
            0.5, 0.01 if layout == "vertical" else 0.03,
            "Lecture : chaque graphique représente une composante différente de la dépense.",
            wrap=True, horizontalalignment='center', fontsize=9, color='gray'
        )

        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        # Sauvegarde et affichage
        plt.savefig(os.path.join(Emplacement_stockage+"/", nom_fichier + '.jpg'), bbox_inches='tight', dpi=qualitéGraphique)
        st.pyplot(fig)


def dispertion_chart_comparaison(df,var_montant,element_titre,qualitéGraphique, Emplacement_stockage):

    
    table=pd.pivot_table(df,values=[var_montant,'nb_acte'],index=['id_beneficiaire','annee_soins'],aggfunc='sum').reset_index()
    table=table[table[var_montant]>0]
    
    bins, labels=calculer_bins_labels_equilibres(table, var_montant, min_pct=0.05, max_pct=0.3, max_bins=20, multiple=5)
    table['Tranche_montant'] = pd.cut(table[var_montant], bins=bins, labels=labels, right=False)
    
    t=pd.pivot_table(table,values=[var_montant,'nb_acte','id_beneficiaire'],index=['Tranche_montant','annee_soins'],observed=False,aggfunc={var_montant:'sum','nb_acte':'sum','id_beneficiaire':pd.Series.nunique}).reset_index()
    t[var_montant]=t[var_montant].fillna(0)
    t['nb_acte']=t['nb_acte'].fillna(0)
    t['id_beneficiaire']=t['id_beneficiaire'].fillna(0)
    
    # Créer le graphique à barres
    fig, ax = plt.subplots(figsize=(10, 6))

    # Utiliser seaborn pour un meilleur aspect (facultatif)
    #sns.set_theme()
    sns.set( style = "darkgrid")    
    # Tracer le graphique à barres

    if table['annee_soins'].nunique()<=2:
        color_palette=[palette[1], palette[5]]
    else:
        color_palette=palette[0:table['annee_soins'].nunique()]

    bar_plot = sns.barplot(x='Tranche_montant', y=var_montant, hue='annee_soins', data=t, ax=ax,palette=color_palette)#[0:len(liste_annee_soins)])

    # Ajouter les annotations id_bénéf au-dessus de chaque barre
    for p in bar_plot.patches:
        val_rc = p.get_height()
        match = t.loc[t[var_montant] == val_rc, 'id_beneficiaire']
        
        if (not match.empty) and (val_rc>0):
            id_benef = match.values[0]
            ax.annotate(
                f'{formatM(id_benef)}',
                (p.get_x() + p.get_width() / 2., val_rc + t[var_montant].mean() / 20),
                ha='center', va='bottom',
                fontweight='bold', color='black',rotation=90,
            )
        else:
            print(f"[INFO] Aucun match RC={val_rc} trouvé dans la table.")

    # Ajouter des étiquettes et un titre
    ax.set_ylabel('Remboursement complémentaire')
    ax.set_xlabel('Tranches de montants')
    
    if len(element_titre) ==1:
        title=f"{element_titre[0]} : Dispersion des remboursements complémentaires et \n $\mathbf{{nombre\ de\ consommants }}$ par survenance\n "
    else:
        title=f"{', '.join(element_titre)} :\nDispersion des remboursements complémentaires et \n $\mathbf{{nombre\ de\ consommants}}$ par survenance"

    ax.set_title(title, pad=30)
    plt.xticks(rotation=45)

    plt.gca().set_facecolor('white')
    plt.grid(True, axis='y', linestyle='--', alpha=0.3, color='grey')

    # Afficher la légende
    ax.legend(title='Survenances', loc='best')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: formatM(x)))
    # Afficher le graphique
    # Ajout du cadre avec les montants de RC par année de soins

    plt.savefig(Emplacement_stockage+"/" +"dispertion_conso_"+ ''.join(element_titre) + '.jpg', bbox_inches='tight', dpi=qualitéGraphique)
    st.pyplot(fig)

