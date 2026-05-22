

import os
import re
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap as _tw
import circlify
import streamlit as st
from adjustText import adjust_text

from fonctions.workOnData import (
    formatM, GetTypeBénéf, Famille_acte_sorted,
    format_string_with_linebreak, format_value,
    optimal_bins, calculer_bins_labels_equilibres,
)

HEX = ["#4295CE","#2C67AF","#2B3885","#2A0C53",
       "#662064","#9B406D","#D86173","#F56C26",
       "#EE9744","#EE9780"]
PALETTE   = sns.color_palette(HEX)
COLOR_MAP = {
    'Hospitalisation':HEX[2],'Soins courants':HEX[3],
    'Consultations et visites':HEX[4],'Pharmacie':HEX[5],
    'Optique':HEX[0],'Dentaire':HEX[1],'Divers':HEX[6]}

BG         = "#FFFFFF"
BG_AX      = "#FFFFFF"
STRIP_DARK = "#1A2440"
STRIP_LIGHT= "#EEF1F9"
C_TITLE    = "#FFFFFF"
C_SUBTITLE = "#3A4A72"
C_ANNOT    = "#2B3885"
C_GRID     = "#EEF1F9"
C_ZERO     = "#A0AAC0"
C_KPIBG    = "#FFFFFF"
C_KPIBORDER= "#2C67AF"
C_COMMENT  = "#1A2440"
C_POS      = "#2B9E6E"
C_NEG      = "#D85050"
FONT       = "Arial"

mpl.rcParams.update({
    "font.family":"Arial",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.spines.left":False,"axes.spines.bottom":False,
    "axes.grid":True,"grid.color":C_GRID,
    "grid.linewidth":0.77,"grid.linestyle":"--",
    "axes.facecolor":BG_AX,"figure.facecolor":BG,
    "xtick.color":"#666E88","ytick.color":"#666E88",
    "legend.labelcolor": "#173A64","legend.fontsize":11,
    "xtick.labelsize":11,"ytick.labelsize":11,
})
MOIS=['Janv','Fév','Mars','Avr','Mai','Juin',
      'Juil','Août','Sept','Oct','Nov','Déc']

# ── helpers ──────────────────────────────────────────────────────────────────

def _fmt_euro(x, _=None):
    if abs(x) >= 1_000_000:
        v = x / 1_000_000
        s = f"{v:.0f}" if v == int(v) else f"{v:.1f}".replace(".", ",")
        return s + " M€"
    if abs(x) >= 1_000:
        v = x / 1_000
        s = f"{v:.0f}" if v == int(v) else f"{v:.1f}".replace(".", ",")
        return s + " k€"
    return f"{x:.0f} €"

def _fmt_k(x, _=None):
    if abs(x) >= 1_000:
        v = x / 1_000
        s = f"{v:.0f}" if v == int(v) else f"{v:.1f}".replace(".", ",")
        return s + " k"
    return f"{x:.0f}"

def _sign(v):
    return f"+{v:.1f} %" if v>=0 else f"{v:.1f} %"

def _color_sign(v):
    return C_POS if v>=0 else C_NEG

def _apply_ax_theme(ax):
    ax.set_facecolor(BG_AX)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(True,linestyle="--",linewidth=0.7,alpha=1,color=C_GRID)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10.5,colors="#666E88",length=0)

def _strip_title(fig, title, subtitle=None, strip_h=0.055, sub_h=0.032, size=13):

    
 
    fig.add_axes([0, 1-strip_h, 1, strip_h], zorder=10)
    ax_t = fig.axes[-1]
    ax_t.set_facecolor(BG)
    ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)
    for sp in ax_t.spines.values(): sp.set_visible(False)
    ax_t.set_xticks([]); ax_t.set_yticks([])
    ax_t.text(0.018, 0.5, title, color=STRIP_DARK, fontsize=size+2,
              fontweight="bold", va="center", transform=ax_t.transAxes)
 
    if subtitle:
        n_chars = len(subtitle)
        if n_chars < 80:
            fs_sub = size
        elif n_chars < 120:
            fs_sub = size - 1
        else:
            fs_sub = size - 2
 
        if n_chars > 110:
            subtitle = "\n".join(_tw.wrap(subtitle, width=110))
            sub_h = 0.065
 
        top_sub = 1 - strip_h
        fig.add_axes([0, top_sub-sub_h, 1, sub_h], zorder=9)
        ax_s = fig.axes[-1]
        ax_s.set_facecolor("none")   # transparent
        ax_s.patch.set_alpha(0)
        ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1)
        for sp in ax_s.spines.values(): sp.set_visible(False)
        ax_s.set_xticks([]); ax_s.set_yticks([])
        ax_s.text(0.018, 0.5, subtitle, color=C_SUBTITLE, fontsize=fs_sub,
                  va="center", transform=ax_s.transAxes, linespacing=1.4)
        return strip_h + sub_h
    return strip_h

def _kpi_row(fig,kpis,bottom=0.0,height=0.13):
    n=len(kpis);w=1/n
    for i,(lbl,val,delta) in enumerate(kpis):
        ax=fig.add_axes([i*w+0.005,bottom+0.005,w-0.010,height-0.010],zorder=8)
        ax.set_facecolor(C_KPIBG)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.axvline(0,color=C_KPIBORDER,linewidth=3)
        ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.set_xticks([]);ax.set_yticks([])
        ax.text(0.08,0.72,lbl,color=C_SUBTITLE,fontsize=8.5,
                va="center",transform=ax.transAxes)
        ax.text(0.08,0.28,val,color=STRIP_DARK,fontsize=13,
                fontweight="bold",va="center",transform=ax.transAxes)
        if delta is not None:
            col=_color_sign(float(str(delta).replace("+","").replace(" %","")))
            ax.text(0.08,0.10,str(delta),color=col,fontsize=9,
                    fontweight="bold",va="center",transform=ax.transAxes)

def _comment_banner(fig,text,bottom=0.0,height=0.055):
    ax=fig.add_axes([0,bottom,1,height],zorder=7)
    ax.set_facecolor(STRIP_LIGHT)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]);ax.set_yticks([])
    ax.text(0.012,0.5,f"▶  {text}",color=C_COMMENT,fontsize=9.5,
            va="center",style="italic",transform=ax.transAxes)

def _finalize(fig, path, dpi):
    """
    Sauvegarde optimisée qualité / poids.

    Stratégie :
    - On plafonne le dpi à 150 (au-delà le gain visuel est nul en PPT,
      mais le poids explose).
    - On cible une largeur pixel maxi de 2 400 px (≈ largeur diapo Full HD ×2).
      Si la figure est trop grande on recalcule le dpi à la baisse.
    - On sauvegarde en PNG plutôt qu'en JPEG :
        * PNG est sans perte → textes nets, pas d'artefacts de compression.
        * Avec optimize=True + niveau 6 il reste léger (≈ 150–400 Ko).
    - bbox_inches="tight" + pad_inches=0.15 pour ne pas rogner les bandes
      de titre tout en évitant les marges inutiles.
    """
    fig.patch.set_facecolor(BG)

    # ── Calcul du dpi effectif ──────────────────────────────────────────
    MAX_DPI   = 200          # plafond absolu
    MAX_WIDTH = 3200         # largeur cible en pixels

    w_inch = fig.get_size_inches()[0]
    dpi_eff = min(dpi, MAX_DPI)
    if w_inch * dpi_eff > MAX_WIDTH:
        dpi_eff = int(MAX_WIDTH / w_inch)

    # ── Extension PNG ───────────────────────────────────────────────────
    png_path = os.path.splitext(path)[0] + ".png"

    plt.savefig(
        png_path,
        dpi=dpi_eff,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor=BG,
        edgecolor="none",
        format="png",
        metadata={"Software": ""},   # réduit légèrement le poids
    )
    st.pyplot(fig,use_container_width=True)
    plt.close(fig)


def _make_fig(w=14, h=7):
    
    return plt.figure(figsize=(w, h), facecolor=BG)

def _add_main_ax(fig,rect):
    ax=fig.add_axes(rect)
    ax.set_facecolor(BG_AX)
    return ax

# ── 1. DISPERSION PAR TRANCHE ─────────────────────────────────────────────────

def DispersionChart_year(df,Var,Famille,annee,qualitéGraphique,Emplacement_stockage,ID):
    dfan=df[(df["annee_soins"]==annee)&(df["annee_paiement"]==annee)]
    TitleFamille=re.sub(r"[0-9]+\. ","",Famille)
    if Var=="RàC":
        piv=pd.pivot_table(dfan,values=["frais_reels","RC","rbt_ss"],
                           index=[ID],aggfunc="sum",fill_value=0).reset_index()
        piv["RàC"]=piv["frais_reels"]-piv["rbt_ss"]-piv["RC"]
    else:
        piv=pd.pivot_table(dfan,values=[Var],index=[ID],
                           aggfunc={Var:np.sum},fill_value=0).reset_index()
    piv=piv[piv[Var]>0]
    if piv.empty: return
    edges=[0,50,100,150,200,300,400,600,1000,np.inf]
    noms=["< 50 €","50–100","100–150","150–200",
          "200–300","300–400","400–600","600–1k","> 1 000 €"]
    piv["Tranche"]=pd.cut(piv[Var],bins=edges,labels=noms,right=False)
    t=(piv.groupby("Tranche",observed=False)
        .agg(Montant=(Var,"sum"),N=(ID,"nunique")).reset_index())
    fig=_make_fig(15,8)
    V="remboursements complémentaires" if Var=="RC" else "restes à charge"
    title_h=_strip_title(fig,f"Dispersion des {V}",
                         subtitle=f"{TitleFamille}  ·  Survenance {annee}")
    kpi_h=0.13;xlabel_h=0.06;comm_h=0.06;bottom=kpi_h+comm_h+xlabel_h
    ax1=_add_main_ax(fig,[0.06,bottom+0.01,0.88,1-title_h-0.02-bottom])
    x=np.arange(len(t))
    bars=ax1.bar(x,t["Montant"],color=HEX[1],width=0.55,
                 edgecolor=BG,linewidth=1.2,zorder=3,alpha=0.92)
    for bar in bars:
        ax1.bar(bar.get_x(),bar.get_height(),width=bar.get_width(),
                bottom=0,color=HEX[0],alpha=0.18,zorder=4)
    _apply_ax_theme(ax1)
    ax1.yaxis.set_major_formatter(FuncFormatter(_fmt_euro))
    ax1.set_xticks(x);ax1.set_xticklabels(t["Tranche"],fontsize=10.5)
    ax1.axhline(0,color=C_ZERO,linewidth=0.8)
    for bar,nb in zip(bars,t["N"]):
        if nb>0:
            ax1.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()*1.015,
                     f"{int(nb):,}".replace(","," "),
                     ha="center",va="bottom",fontsize=10,
                     fontweight="bold",color=C_ANNOT)
    ax2=ax1.twinx()
    ax2.plot(x,t["N"],color=HEX[7],linewidth=2.5,marker="o",markersize=7,
             zorder=5,markerfacecolor="white",markeredgewidth=2,
             markeredgecolor=HEX[7])
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.tick_params(axis="y",labelsize=10,colors=HEX[7])
    for sp in ax2.spines.values(): sp.set_visible(False)
    moy=piv[Var].sum()/piv[ID].nunique()
    _kpi_row(fig,[("Montant total",f"{formatM(piv[Var].sum())} €",None),
                  (f"Nb {GetTypeBénéf(ID)}",f"{int(piv[ID].nunique()):,}".replace(",", " "),None),
                  ("Moyenne",f"{formatM(moy)} €",None)],
             bottom=comm_h,height=kpi_h)
    _comment_banner(fig,
        f"Tranche la plus représentée : "
        f"{t.loc[t['Montant'].idxmax(),'Tranche']}  "
        f"({formatM(t['Montant'].max())} €)",
        bottom=0,height=comm_h)
    title=f"Dispersion {V} {TitleFamille} {annee}"
    _finalize(fig,f"{Emplacement_stockage}/{title}.jpg",qualitéGraphique)

# ── 2. VENTILATION COÛTS ──────────────────────────────────────────────────────

def PlotVentilationCouts(df_data,annee,qualitéGraphique,Emplacement_stockage,ID):
    dfp=pd.pivot_table(df_data,values=["frais_reels","rbt_ss","RC"],
                       index=[ID,"famille_acte_aops"],aggfunc="sum").reset_index()
    dfp= dfp[dfp['famille_acte_aops']!='Divers']
    dfp=dfp[dfp["RC"]>0]
    dfp["RàC"]=dfp["frais_reels"]-dfp["rbt_ss"]-dfp["RC"]
    eff=(pd.pivot_table(dfp,values=[ID],index=["famille_acte_aops"],
                        aggfunc=lambda x:len(x.unique()))
         .reindex(Famille_acte_sorted(df_data)))
    eff=pd.concat([eff,pd.DataFrame({ID:[dfp[ID].nunique()]},index=["Total"])])
    t=(pd.pivot_table(dfp,values=["rbt_ss","RC","RàC"],
                      index=["famille_acte_aops"],aggfunc=np.sum)
       .round(2).reindex(Famille_acte_sorted(dfp)))
    t=pd.concat([t,pd.DataFrame({"rbt_ss":[dfp["rbt_ss"].sum()],
                                  "RC":[dfp["RC"].sum()],
                                  "RàC":[dfp["RàC"].sum()]},index=["Total"])])
    for col in ["rbt_ss","RC","RàC"]: t[col]=t[col]/eff[ID]
    t["total"]=t[["rbt_ss","RC","RàC"]].sum(axis=1);t[t<0]=0
    stk=(t[["rbt_ss","RC","RàC"]]
         .apply(lambda x:x*100/x.sum(),axis=1)
         .rename(columns={"rbt_ss":"Sécurité Sociale",
                           "RC":"Remboursement Complémentaire","RàC":"Reste à Charge"}))
    
    date_obs = ""
    if "date_paiement" in df_data.columns:
        d = pd.to_datetime(df_data["date_paiement"], errors="coerce").max()
        if pd.notna(d):
            date_obs = f"  ·  Observée au {d.strftime('%d/%m/%Y')}"
 
    fig = _make_fig(14, 8)
    title_h = _strip_title(
        fig,
        f"Survenance {annee}{date_obs}"
    )
    kpi_h=0.13   # hauteur KPI boxes
    xlabel_h=0.02  # espace réservé pour les labels axe X (rotation 20°, textes longs)
    bottom=kpi_h+xlabel_h
    ax=_add_main_ax(fig,[0.06,bottom,0.90,1-title_h-0.02-bottom])
    stk.plot(kind="bar",stacked=True,ax=ax,
             color=[HEX[4],HEX[5],HEX[6]],edgecolor=BG,
             linewidth=0.8,zorder=3,width=0.55)
    _apply_ax_theme(ax)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}%"))

    for i, label in enumerate(ax.get_xticklabels()):
        if i % 2 == 1:
            label.set_y(label.get_position()[1] - 0.05)

    
    ax.set_xticklabels(stk.index,rotation=00,ha="center",fontsize=15.5)
    ax.tick_params(axis='x',pad=4)
    ax.set(xlabel="",ylabel="")
    ax.tick_params(axis='y', colors="#173A64",labelsize=13)
    ax.tick_params(axis='x', colors="#173A64")
    idx_list=list(t.index);col_names=["rbt_ss","RC","RàC"];n_idx=len(idx_list)
    for pi,p in enumerate(ax.patches):
        col_i=pi//n_idx;row_i=pi%n_idx
        val=t[col_names[col_i]].iloc[row_i]
        pct=stk.iloc[row_i,col_i]
        if val>0 and pct>=6:
            ax.text(p.get_x()+p.get_width()/2,p.get_y()+p.get_height()/2,
                    f"{formatM(val)} €",ha="center",va="center",
                    color="white",fontweight="bold",fontsize=16.5)
        if col_i==2 and val>0:
            ax.annotate(f"{formatM(t['total'].iloc[row_i])} €",
                xy=(p.get_x()+p.get_width()/2,101),
                xytext=(0,5),textcoords="offset points",
                ha="center",color="white",fontsize=16.5,fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",fc=STRIP_DARK,ec="none",alpha=0.88))
    ax.legend(loc="lower center",bbox_to_anchor=(0.5,-0.22),ncol=3,
              fontsize=15,frameon=False)
    title=f"Ventilation coûts {annee}"
    _finalize(fig,f"{Emplacement_stockage}/{title}.jpg",qualitéGraphique)

# ── 3. BULLES FAMILLES ────────────────────────────────────────────────────────
def get_color(name,number):
    return list(sns.color_palette(palette=name,n_colors=number).as_hex())

def distributionFamilleActes(df,annee,Emplacement_stockage,qualitéGraphique):
    pv=(pd.pivot_table(df[df["annee_soins"]==annee],index="famille_acte_aops",
                       values="RC",aggfunc="sum").reset_index())
    pv["taux"]=pv["RC"]/pv["RC"].sum()
    rcs=pv.sort_values("RC",ascending=False)

    rcs_plot = rcs[rcs["famille_acte_aops"] != "Divers"]
    circles=circlify.circlify(rcs_plot["RC"].tolist(),show_enclosure=False,
                               target_enclosure=circlify.Circle(x=0,y=0))
    circles.reverse()
    fig,ax=plt.subplots(figsize=(9,9),facecolor=STRIP_DARK)
    ax.set_facecolor(STRIP_DARK);ax.axis("off")
    lim=max(max(abs(c.x)+c.r,abs(c.y)+c.r) for c in circles)
    ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim)

    for circle,label,rc,color in zip(circles,rcs["famille_acte_aops"],rcs["RC"],[HEX[0]]+HEX[2:5]+[HEX[6]] +[HEX[8]]+[HEX[7]]):
        x,y,r=circle
        ax.add_patch(plt.Circle((x,y),r*1.06,alpha=0.15,color="white",zorder=1))
        ax.add_patch(plt.Circle((x,y),r,color=color,alpha=0.95,zorder=2,edgecolor="black",linewidth=1.2))
        pct=rc/rcs["RC"].sum()*100
        fs=max(7,min(13,int(r*52)))
        ax.text(x,y,
                f"{format_string_with_linebreak(label)}\n{format_value(rc)} €\n{pct:.0f} %",
                fontsize=fs,va="center",ha="center",fontweight="bold",
                color="white",zorder=3,
                path_effects=[pe.withStroke(linewidth=2,foreground=color)])
    ax.set_title(f"Distribution des familles d'actes  ·  {annee}",
                 fontsize=15,fontweight="bold",color="white",pad=0)
    fig.patch.set_facecolor(STRIP_DARK)
    title=f"Distribution actes {annee}"
    plt.title(title,color=STRIP_DARK, weight='bold',fontsize=13 )
    _finalize(fig,f"{Emplacement_stockage}/_{title}_.jpg",qualitéGraphique)



def distributionFamilleActes_pyramide(df, annee, Emplacement_stockage, qualitéGraphique):
    """Pyramide inversée : plus gros RC en haut, décroissant vers le bas."""
    pv = (pd.pivot_table(df[df["annee_soins"] == annee],
                          index="famille_acte_aops",
                          values="RC", aggfunc="sum").reset_index())
    rcs = pv[pv["famille_acte_aops"] != "Divers"].sort_values("RC", ascending=False)
    total = rcs["RC"].sum()

    colors = ([HEX[0]] + HEX[2:5] + [HEX[6]] + [HEX[8]])[:len(rcs)]
    MAX_W, MIN_W, ROW_H = 0.85, 0.15, 0.13
    n = len(rcs)

    fig = _make_fig(13, 7)
    title_h = _strip_title(fig,
        f"Distribution des familles d'actes · {annee}")
    ax = _add_main_ax(fig, [0.05, 0.05, 0.90, 1 - title_h - 0.05])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, (_, row) in enumerate(rcs.iterrows()):
        rc    = row["RC"]
        label = row["famille_acte_aops"]
        pct   = rc / total * 100
        color = colors[i]

        w = MIN_W + (rc / rcs["RC"].max()) * (MAX_W - MIN_W)
        x = (1 - w) / 2
        y = 1 - (i + 1) * ROW_H - 0.02

        rect = mpatches.FancyBboxPatch(
            (x, y), w, ROW_H * 0.88,
            boxstyle="round,pad=0.005",
            facecolor=color, edgecolor="white",
            linewidth=1.5, zorder=3)
        ax.add_patch(rect)

        cx, cy = x + w / 2, y + ROW_H * 0.44
        fs_label = max(14, min(14, int(w * 20)))
        ax.text(cx - w * 0.18, cy, label,
                ha="center", va="center",
                fontsize=fs_label, fontweight="bold",
                color="white", zorder=4)
        ax.text(cx + w * 0.22, cy,
                f"{formatM(rc)} €   {pct:.0f} %",
                ha="center", va="center",
                fontsize=fs_label - 1, color="white",
                alpha=0.92, zorder=4)

    title = f"Distribution actes pyramide {annee}"
    _finalize(fig, f"{Emplacement_stockage}/_{title}_.png", qualitéGraphique)


def distributionFamilleActes_donut(df, annee, Emplacement_stockage, qualitéGraphique):
    """Donut avec légende détaillée à droite — barre de progression par famille."""
    pv = (pd.pivot_table(df[df["annee_soins"] == annee],
                          index="famille_acte_aops",
                          values="RC", aggfunc="sum").reset_index())
    rcs = pv[pv["famille_acte_aops"] != "Divers"].sort_values("RC", ascending=False)
    total = rcs["RC"].sum()

    labels = rcs["famille_acte_aops"].tolist()
    values = rcs["RC"].tolist()
    colors = ([HEX[0]] + HEX[2:5] + [HEX[6]] + [HEX[8]])[:len(labels)]

    fig = _make_fig(14, 7)
    title_h = _strip_title(fig,
        f"Distribution des familles d'actes · {annee}")

    # ── Donut (gauche) ────────────────────────────────────────────────────────
    ax_d = fig.add_axes([0.02, 0.08, 0.44, 1 - title_h - 0.10])
    ax_d.set_facecolor(BG)

    wedges, _ = ax_d.pie(
        values, colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2.5),
        counterclock=False)

    # Centre du donut
    ax_d.text(0, 0.08, formatM(total) + " €",
              ha="center", va="center",
              fontsize=16, fontweight="bold", color=STRIP_DARK)
    ax_d.text(0, -0.12, "Total RC",
              ha="center", va="center",
              fontsize=10, color=C_SUBTITLE)

    # ── Légende détaillée (droite) ────────────────────────────────────────────
    ax_l = fig.add_axes([0.50, 0.08, 0.48, 1 - title_h - 0.10])
    ax_l.set_facecolor(BG)
    ax_l.axis("off")
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1)

    n = len(labels)
    row_h = 0.88 / n
    max_rc = max(values)
    BAR_MAX = 0.30

    for i, (label, rc, color) in enumerate(zip(labels, values, colors)):
        pct = rc / total * 100
        y   = 0.94 - i * row_h
        cy  = y - row_h * 0.35

        # Barre d'accent colorée
        ax_l.add_patch(mpatches.FancyBboxPatch(
            (0.0, cy - 0.025), 0.012, row_h * 0.72,
            boxstyle="round,pad=0.002",
            facecolor=color, linewidth=0, zorder=3))

        # Nom famille
        ax_l.text(0.04, cy + 0.03, label,
                  fontsize=15.5, fontweight="bold",
                  color=STRIP_DARK, va="center")

        # Montant
        ax_l.text(0.04, cy - 0.018, f"{formatM(rc)} €",
                  fontsize=14.5, color=C_SUBTITLE, va="center")

        # Pourcentage
        ax_l.text(0.62, cy, f"{pct:.0f} %",
                  fontsize=15, fontweight="bold",
                  color=STRIP_DARK, va="center", ha="right")

        # Mini barre de progression
        bar_w = (rc / max_rc) * BAR_MAX
        ax_l.add_patch(mpatches.FancyBboxPatch(
            (0.64, cy - 0.008), BAR_MAX, 0.016,
            boxstyle="round,pad=0.003",
            facecolor=C_GRID, linewidth=0, zorder=2))
        ax_l.add_patch(mpatches.FancyBboxPatch(
            (0.64, cy - 0.008), bar_w, 0.016,
            boxstyle="round,pad=0.003",
            facecolor=color, linewidth=0, zorder=3))

    title = f"Distribution actes donut {annee}"
    _finalize(fig, f"{Emplacement_stockage}/_{title}_.png", qualitéGraphique)

#### EvoRC par famille d'actes (barplot horizontal)

def Evo_RC(df,cat_assure,qualitéGraphique,Emplacement_stockage):

    df = df.copy()
    
    if not isinstance(cat_assure, list):
        cat_assure = [cat_assure]

    
    df = df[df["cat_assure"].isin(cat_assure)]


    ordre=Famille_acte_sorted(df)
    if "Divers" in ordre: ordre.remove("Divers")
    df=df.copy();df["annee_soins"]=df["annee_soins"].fillna(0).astype(int)
    t=(pd.pivot_table(df,values="RC",index=["famille_acte_aops","annee_soins"],
                      aggfunc="sum",fill_value=0).reset_index())
    t["pct"]=t.groupby("annee_soins")["RC"].transform(lambda x:x/x.sum())
    t=t[t["famille_acte_aops"]!="Divers"]
    annees=sorted(t["annee_soins"].unique());n_annees=len(annees)
    bar_h=0.65/n_annees
    fig=_make_fig(20,max(9,len(ordre)*0.9+3))
    ax=_add_main_ax(fig,[0.24,0.08,0.68,1-0.06])
    y_pos={fam:i for i,fam in enumerate(ordre[::-1])}
    offsets=np.linspace(-(n_annees-1)/2,(n_annees-1)/2,n_annees)*bar_h
    palette = [HEX[i] for i in [0,1, 4]] + [c for i, c in enumerate(HEX) if i not in [0, 1, 4]]

    for j,annee in enumerate(annees):
        sub=t[t["annee_soins"]==annee];col=palette[j%len(palette)]
        for _,row in sub.iterrows():
            fam=row["famille_acte_aops"]
            if fam not in y_pos: continue
            y=y_pos[fam]+offsets[j]
            ax.barh(y,row["RC"],height=bar_h*0.88,color=col,alpha=0.88,
                    edgecolor=BG,linewidth=0.8,zorder=3,
                    label=str(annee) if fam==ordre[0] else "")
            ax.text(row["RC"]*1.01,y,f"{row['pct']*100:.1f} %",
                    va="center",fontsize=20,color=col,fontweight="bold")
    _apply_ax_theme(ax)
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()),fontsize=17, color = "#173A64")
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_euro))
    ax.set_xlim(0, t["RC"].max() * 1.2)
    ax.tick_params(axis='x', colors="#173A64", labelsize=17)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.grid(True, linestyle='--', alpha=0.3, color='grey')
    handles,labels=ax.get_legend_handles_labels()
    seen={}
    for h,l in zip(reversed(handles),reversed(labels)):
        if l not in seen: seen[l]=h
    ax.legend(seen.values(),seen.keys(),loc="upper right",fontsize=15,frameon=False)
    if df["annee_soins"].nunique()==1:
        title=f"RC par famille {annees[0]}"
    else:
        title=f"RC par famille {annees[0]}-{annees[-1]}"
    _finalize(fig,f"{Emplacement_stockage}/{title}{cat_assure}.jpg",qualitéGraphique)


# ── 4. ÉVOLUTION CONSO MOYENNE ───────────────────────────────────────────────

def Evo_Cons_Moyenne(df, cat_assure, qualitéGraphique, Emplacement_stockage, ID):
    df = df.copy()

    if not isinstance(cat_assure, list):
        cat_assure = [cat_assure]

    df = df[df["cat_assure"].isin(cat_assure)]

    
    df = df.copy()
    df["annee_soins"] = df["annee_soins"].fillna(0).astype(int)
    nb_surv = df["annee_soins"].nunique()
    if nb_surv < 2:
        st.warning("Pas assez de survenances."); return

    # ── Calcul tables RC et effectifs ────────────────────────────────────────
    table = (pd.pivot_table(df, values="RC", index=["famille_acte_aops"],
                             columns="annee_soins", aggfunc=np.sum, fill_value=0)
             .reindex(Famille_acte_sorted(df)))
    tableEff = (pd.pivot_table(df, values=ID, index="famille_acte_aops",
                                columns="annee_soins", aggfunc=pd.Series.nunique)
                .reindex(Famille_acte_sorted(df)))
    n = min(nb_surv, 3)
    table    = table[table.columns[-n:]]
    tableEff = tableEff[tableEff.columns[-n:]]
    sub = df[df["annee_soins"] >= table.columns.min()]
    row_rc  = pd.DataFrame({c: [sub[sub["annee_soins"]==c]["RC"].sum()]   for c in table.columns},    index=["Total"])
    row_eff = pd.DataFrame({c: [sub[sub["annee_soins"]==c][ID].nunique()] for c in tableEff.columns}, index=["Total"])
    table    = pd.concat([table,    row_rc])
    tableEff = pd.concat([tableEff, row_eff])
    for col in table.columns:
        table[col] = table[col] / tableEff[col]

    # ── Calcul évolutions ────────────────────────────────────────────────────
    cols = list(table.columns); evol_cols = []
    if nb_surv >= 3:
        c0, c1, c2 = cols[0], cols[1], cols[2]
        table[f"{c1}/{c0}"] = (table[c1]-table[c0]) / table[c0] * 100
        table[f"{c2}/{c0}"] = (table[c2]-table[c0]) / table[c0] * 100
        table[f"{c2}/{c1}"] = (table[c2]-table[c1]) / table[c1] * 100
        evol_cols = [f"{c1}/{c0}", f"{c2}/{c0}", f"{c2}/{c1}"]
    else:
        c0, c1 = cols[0], cols[1]
        table[f"{c1}/{c0}"] = (table[c1]-table[c0]) / table[c0] * 100
        evol_cols = [f"{c1}/{c0}"]

    if "Divers" in table.index:
        table = table.drop(index="Divers")

    # ── surv_all et date_max — après n est connu ──────────────────────────────

    fig     = _make_fig(16, 8)
    comm_h  = 0.06
    ax      = _add_main_ax(fig, [0.06, comm_h+0.01, 0.90, 1-0.04-comm_h])

    # ── CAS >= 3 SURVENANCES : ancienne structure, couleurs palette ──────────
    if nb_surv >= 3:
        x       = np.arange(len(table))
        n_evol  = len(evol_cols)
        bar_w   = 0.65 / n_evol

        for ei, col in enumerate(evol_cols):
            off   = (ei - (n_evol-1)/2) * bar_w
            vals  = table[col].values
            palette = [HEX[i] for i in [0,1, 4]] + [c for i, c in enumerate(HEX) if i not in [0, 1, 4]]
            color = palette[ei % len(palette)]
            ax.bar(x + off, vals, width=bar_w*0.92, color=color,
                   alpha=0.88, edgecolor=BG, linewidth=0.8, zorder=3)
            ymin_ax, ymax_ax = ax.get_ylim()
            pad = (ymax_ax - ymin_ax) * 0.2 if ymax_ax != ymin_ax else 1
            for xi, v in enumerate(vals):
                
                sign = "+" if v >= 0 else ""
                ax.annotate(f"{sign}{v:.0f}%",
                    xy=(x[xi] + off, v + (pad*0.15 if v >= 0 else -pad*0.25)),
                    ha="center", va="bottom" if v >= 0 else "top",
                    size=15.5, xytext=(0, 8), textcoords="offset points",
                    rotation=0, color=color,fontweight="bold")

        ax.axhline(0, color=C_ZERO, linewidth=1.2, zorder=2)
        ax.set_xticks(x)
        for i, label in enumerate(ax.get_xticklabels()):

            if i % 2 == 1:
                label.set_y(label.get_position()[1] - 0.025)

        ax.set_xticklabels(table.index, fontsize=30, rotation=0, ha="center")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}%"))
        _apply_ax_theme(ax)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.3, color='grey')
        ymin_ax, ymax_ax = ax.get_ylim()
        pad = (ymax_ax - ymin_ax) * 0.2
        ax.set_ylim(ymin_ax - pad, ymax_ax + pad)
        ax.tick_params(axis='y', colors="#173A64", labelsize = 15)
        ax.tick_params(axis='x', colors="#173A64", labelsize= 15)
        ax.legend(
            handles=[mpatches.Patch(color=palette[i], label=col)
                     for i, col in enumerate(evol_cols)],
            loc="lower center", bbox_to_anchor=(0.5, -0.18),
            ncol=n_evol, fontsize=13.5 ,frameon=False)

    # ── CAS 2 SURVENANCES : vert/rouge, sans cubes ───────────────────────────
    else:
        bar_w   = 0.22
        n_evol  = len(evol_cols)
        x       = np.arange(len(table))
        offsets = np.linspace(-(n_evol-1)/2, (n_evol-1)/2, n_evol) * bar_w

        for i, (col, off) in enumerate(zip(evol_cols, offsets)):
            vals = table[col].values
            for xi, v in enumerate(vals):
                ax.bar(xi+off, v, width=bar_w*0.92, color=_color_sign(v),
                       alpha=0.80, edgecolor=BG, linewidth=0.8, zorder=3)
                sign = "+" if v >= 0 else ""
                ax.text(xi+off, v+(0.4 if v >= 0 else -0.7),
                        f"{sign}{v:.1f}%",
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=13.5, fontweight="bold", color=_color_sign(v))

        ax.axhline(0, color=C_ZERO, linewidth=1.2, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(table.index, rotation=0, ha="right", fontsize=15)
        for i, label in enumerate(ax.get_xticklabels()):
            if i % 2 == 1:
                label.set_y(label.get_position()[1] - 0.05)

        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}%"))
        _apply_ax_theme(ax)
        ax.tick_params(axis='y', colors="#173A64",labelsize=15)
        ax.tick_params(axis='x', colors="#173A64",labelsize=15)
        pad_pct = (ax.get_ylim()[1]-ax.get_ylim()[0]) * 0.15
        ax.set_ylim(ax.get_ylim()[0]-pad_pct, ax.get_ylim()[1]+pad_pct)

    _finalize(fig, f"{Emplacement_stockage}/{"evo_cons"}{cat_assure}.jpg", qualitéGraphique)

# ── 5. ÉVOLUTION NB CONSOMMATEURS ────────────────────────────────────────────

def EVO_Consommateurs(df,cat_assure,qualitéGraphique,Emplacement_stockage,ID):

    df=df.copy();df["annee_soins"]=df["annee_soins"].fillna(0).astype(int)

    df = df[df["cat_assure"].isin(cat_assure)]

    annees=sorted(df["annee_soins"].unique(),reverse=True)
    pivot=(pd.pivot_table(df,values=[ID],index=["annee_soins","mois_soins"],
                          aggfunc=lambda x:len(x.unique())).reset_index())
    mois_min=int(df["mois_soins"].min());mois_max=int(df["mois_soins"].max())
    delta=None;delta_txt=None
    if len(annees)>1:
        a1,a0=sorted(annees)[-1],sorted(annees)[-2]
        delta=round((df[df["annee_soins"]==a1][ID].nunique()/
                     df[df["annee_soins"]==a0][ID].nunique()-1)*100,2)
        delta_txt=_sign(delta)
    fig=_make_fig(14,7)
    title_h=_strip_title(fig,f"Évolution mensuelle du nombre de consommants")

    kpi_h=0.13;xlabel_h=0.07;comm_h=0.06 if delta_txt else 0;bottom=kpi_h+comm_h+xlabel_h
    ax=_add_main_ax(fig,[0.07,bottom+0.01,0.88,1-title_h-0.03-bottom])
    for annee,col in zip(annees,[HEX[4]] + [HEX[1]] +[HEX[0]][:len(annees)]):
        sub=pivot[pivot["annee_soins"]==annee]
        ax.plot(sub["mois_soins"],sub[ID],color=col,linewidth=2.8,zorder=3,
                marker="o",markersize=6,markerfacecolor="white",
                markeredgewidth=2,markeredgecolor=col,label=str(annee))
        if annee==annees[0]:
            ax.fill_between(sub["mois_soins"],sub[ID],alpha=0.08,color=col,zorder=2)
    _apply_ax_theme(ax)
    ax.set_xticks(range(mois_min,mois_max+1))
    ax.set_xticklabels(MOIS[mois_min-1:mois_max],fontsize=10.5)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_k))
    ax.tick_params(axis='y', colors="#173A64",labelsize=15)
    ax.tick_params(axis='x', colors="#173A64",labelsize=15)
    ax.set_ylabel("Consommants" ,color="#173A64", fontsize=11)

    leg = ax.legend(title="Survenance",fontsize=12,title_fontsize=12,frameon=False,loc="center right")
    leg.get_title().set_color("#173A64")
    _finalize(fig,f"{Emplacement_stockage}/Évolution nb consommants.jpg",qualitéGraphique)

# ── 6. ÉVOLUTION REMBOURSEMENT MOYEN ─────────────────────────────────────────

def EVO_Remboursement_moy(df,var,qualitéGraphique,Emplacement_stockage,ID):
    df=df.copy();df["annee_soins"]=df["annee_soins"].fillna(0).astype(int)
    annees=sorted(df["annee_soins"].unique(),reverse=True)
    t=(pd.pivot_table(df,values=[ID,var],index=["annee_soins","mois_soins"],
                      aggfunc={ID:lambda x:len(x.unique()),var:np.sum})
       .reset_index())
    t["Moy"]=t[var]/t[ID]
    noms={"RC":"remboursement complémentaire","RàC":"reste à charge",
          "rbt_ss":"remboursement Sécurité Sociale"}
    name=noms.get(var,var)
    delta=None;delta_txt=None
    if len(annees)>1:
        a1,a0=sorted(annees)[-1],sorted(annees)[-2]
        m1=df[df["annee_soins"]==a1][var].sum()/df[df["annee_soins"]==a1][ID].nunique()
        m0=df[df["annee_soins"]==a0][var].sum()/df[df["annee_soins"]==a0][ID].nunique()
        delta=round((m1/m0-1)*100,2);delta_txt=_sign(delta)
    mois_min=int(df["mois_soins"].min());mois_max=int(df["mois_soins"].max())
    fig=_make_fig(14,7)
    title_h=_strip_title(fig,f"Évolution mensuelle du {name} moyen par consommant")

    kpi_h=0.13;xlabel_h=0.07;comm_h=0.06 if delta_txt else 0;bottom=kpi_h+comm_h+xlabel_h
    ax=_add_main_ax(fig,[0.07,bottom+0.01,0.88,1-title_h-0.03-bottom])
    for annee,col in zip(annees,[HEX[4]] + [HEX[1]] +[HEX[0]][:len(annees)]):
        sub=t[t["annee_soins"]==annee]
        ax.plot(sub["mois_soins"],sub["Moy"],color=col,linewidth=2.8,zorder=3,
                marker="o",markersize=6,markerfacecolor="white",
                markeredgewidth=2,markeredgecolor=col,label=str(annee))
        if annee==annees[0]:
            ax.fill_between(sub["mois_soins"],sub["Moy"],alpha=0.08,color=col,zorder=2)
    _apply_ax_theme(ax)
    ax.set_xticks(range(mois_min,mois_max+1))
    ax.set_xticklabels(MOIS[mois_min-1:mois_max],fontsize=11.5)
    ax.tick_params(axis='y', colors="#173A64",labelsize=15)
    ax.tick_params(axis='x', colors="#173A64",labelsize=15)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_euro))
    leg = ax.legend(title="Survenance",fontsize=11.5,title_fontsize=11.5,frameon=False,loc="upper right")
    leg.get_title().set_color("#173A64")

    title=f"Évolution {name} moyen"
    _finalize(fig,f"{Emplacement_stockage}/{title}.jpg",qualitéGraphique)




# ── 8. MONTANT MENSUEL ────────────────────────────────────────────────────────

def EVO_Montant(df,var,qualitéGraphique,Emplacement_stockage):
    df=df.copy();df["annee_soins"]=df["annee_soins"].fillna(0).astype(int)
    annees=sorted(df["annee_soins"].unique(),reverse=True)
    monthly=df.groupby(["annee_soins","mois_soins"])[var].sum().reset_index()
    noms={"RC":"remboursement complémentaire","RàC":"reste à charge",
          "rbt_base":"remboursement Sécurité Sociale"}
    name=noms.get(var,var)
    mois_min=int(df["mois_soins"].min());mois_max=int(df["mois_soins"].max())
    delta=None;delta_txt=None
    if len(annees)>1:
        a1,a0=sorted(annees)[-1],sorted(annees)[-2]
        delta=round((df[df["annee_soins"]==a1][var].sum()/
                     df[df["annee_soins"]==a0][var].sum()-1)*100,2)
        delta_txt=_sign(delta)
    fig=_make_fig(14,7)
    title_h=_strip_title(fig,f"Évolution mensuelle du {name}")

    kpi_h=0.13;xlabel_h=0.07;comm_h=0.06 if delta_txt else 0;bottom=kpi_h+comm_h+xlabel_h
    ax=_add_main_ax(fig,[0.07,bottom+0.01,0.88,1-title_h-0.03-bottom])
    for annee,col in zip(annees,[HEX[4]] + [HEX[1]] +[HEX[0]] [:len(annees)]):
        sub=monthly[monthly["annee_soins"]==annee]
        ax.plot(sub["mois_soins"],sub[var],color=col,linewidth=2.8,zorder=3,
                marker="o",markersize=6,markerfacecolor="white",
                markeredgewidth=2,markeredgecolor=col,label=str(annee))
        if annee==annees[0]:
            ax.fill_between(sub["mois_soins"],sub[var],alpha=0.08,color=col,zorder=2)
    _apply_ax_theme(ax)
    ax.set_xticks(range(mois_min,mois_max+1))
    ax.set_xticklabels(MOIS[mois_min-1:mois_max],fontsize=10.5)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_euro))
    ax.tick_params(axis='y', colors="#173A64",labelsize=15)
    ax.tick_params(axis='x', colors="#173A64",labelsize=15)
    leg = ax.legend(title="Survenance",fontsize=12,title_fontsize=12,  frameon=False,loc="best")
    leg.get_title().set_color("#173A64")
    title=f"Évolution mensuelle {name}"
    _finalize(fig,f"{Emplacement_stockage}/{title}.jpg",qualitéGraphique)

# ── 9. SCATTER PANIER ─────────────────────────────────────────────────────────

def Panier_plot(d,ID,PanierVar,titre,qualitéGraphique,Emplacement_stockage):
    d=d.copy();d[PanierVar]=d[PanierVar].str.title()
    t=(pd.pivot_table(d,values=[ID,"nb_acte","RC"],
                      index=["annee_soins",PanierVar],
                      aggfunc={ID:"nunique","nb_acte":"sum","RC":"sum"})
       .reset_index().rename(columns={"annee_soins":"Année"}))
    t["RC moyen"]=(t["RC"]/t["nb_acte"]).round(2)
    pal=HEX[::3][:t[PanierVar].nunique()]
    fig=_make_fig(8,4)
    title_h=_strip_title(fig,f"100 % Santé  ·  {titre}",size = 7, sub_h = 0.090)
    ax=_add_main_ax(fig,[0.09,0.06,0.86,1-title_h-0.12])
    base_fs=fig.get_size_inches()[1]*2.0
    sns.scatterplot(data=t,x="RC moyen",y="nb_acte",hue=PanierVar,palette=pal,
                    size="RC",sizes=(600,7000),legend=False,ax=ax,alpha=0.82,
                    edgecolors="white",linewidth=2.0,zorder=4)
    _apply_ax_theme(ax)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_euro))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_k))
    ax.set_xlabel("Remboursement complémentaire moyen / acte",color = "#173A64",fontsize=11)
    ax.set_ylabel("Nombre d'actes",color = "#173A64",fontsize=11)
    x_min,x_max=ax.get_xlim();y_min,y_max=ax.get_ylim()
    ax.set_xlim(x_min-(x_max-x_min)*0.25,x_max+(x_max-x_min)*0.25)
    ax.set_ylim(y_min-(y_max-y_min)*0.25,y_max+(y_max-y_min)*0.25)
    ax.tick_params(axis='y', colors="#173A64",labelsize=13.5)
    ax.tick_params(axis='x', colors="#173A64",labelsize=13.5)
    lfs=base_fs*max(0.65,min(1.0,7/len(t)))
    texts=[]
    for _,row in t.iterrows():
        txt=ax.text(row["RC moyen"],row["nb_acte"],
                    f"{row['Année']}\n{row[PanierVar]}\n{formatM(row['RC'])} €",
                    ha="center",va="center",fontsize=lfs,fontweight="bold",zorder=5,
                    bbox=dict(boxstyle="round,pad=0.50",fc="white",
                              ec=HEX[1],alpha=0.85,linewidth=0.9))
        texts.append(txt)
    fig.canvas.draw()
    adjust_text(texts,ax=ax,expand=(1.05,1.2),
                arrowprops=dict(arrowstyle="-",lw=0),
                force_points=(0.3,0.5),force_text=(0.5,0.5),
                only_move={"points":"y","text":"xy"})
    _finalize(fig,f"{Emplacement_stockage}/{titre}.jpg",qualitéGraphique)

# ── 10. PANIER VENTILATION ────────────────────────────────────────────────────

def Panier_plot_ventilation(d,ID,PanierVar,titre,qualitéGraphique,Emplacement_stockage):
    d=d.copy();d[PanierVar]=d[PanierVar].str.title()
    t1=pd.pivot_table(d,values=[ID,"nb_acte","RC"],
                      index=["annee_soins",PanierVar,"sous_famille"],
                      aggfunc={ID:pd.Series.nunique,"nb_acte":"sum","RC":"sum"}).reset_index()
    t=pd.pivot_table(d,values=[ID,"nb_acte","RC"],
                     index=["annee_soins",PanierVar],
                     aggfunc={ID:pd.Series.nunique,"nb_acte":"sum","RC":"sum"}).reset_index()
    res=pd.merge(t1[["annee_soins",PanierVar,"RC","sous_famille"]],
                 t[["annee_soins",PanierVar,"RC"]],on=["annee_soins",PanierVar],how="left")
    res["taux"]=(res["RC_x"]/res["RC_y"].replace(0,np.nan)*100).round(2).fillna(0)
    piv_m=res[["annee_soins",PanierVar,"sous_famille","RC_x"]].pivot(
        index=[PanierVar,"annee_soins"],columns=["sous_famille"],values="RC_x").fillna(0)
    piv_t=res[["annee_soins",PanierVar,"sous_famille","taux"]].pivot(
        index=[PanierVar,"annee_soins"],columns=["sous_famille"],values="taux").fillna(0)
    if PanierVar=="sante_100" and "Dentaire" in d["famille_acte_aops"].unique():
        ordre=["100% Santé","Maîtrisés","Libre"]
        for piv in (piv_t,piv_m):
            piv.index=pd.MultiIndex.from_arrays([
                pd.Categorical(piv.index.get_level_values(PanierVar),
                               categories=ordre,ordered=True),
                piv.index.get_level_values("annee_soins")],names=piv.index.names)
            piv.sort_index(inplace=True)
    fig=_make_fig(11,4)
    title_h=_strip_title(fig,"Taux d'utilisation par panier et sous-famille", size=8,sub_h=0.090)

    
    ax=_add_main_ax(fig,[0.06,0.14,0.90,1-title_h-0.12])
    piv_t.plot.bar(stacked=True,ax=ax,color=HEX[3:3+len(piv_t.columns)-1]+[HEX[0]] ,
                   edgecolor=BG,linewidth=0.8,zorder=3,width=0.55)
    _apply_ax_theme(ax)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}%"))
    ax.tick_params(axis='y', colors="#173A64",labelsize=14)
    ax.tick_params(axis='x', colors="#173A64",labelsize=14)
    ax.set_xticklabels([str(x[1]) for x in piv_t.index],rotation=0,fontsize=12.5, zorder=1)
    ax.set(xlabel="",ylabel="")
    for panier in piv_t.index.get_level_values(PanierVar).unique():
        pos=[i for i,x in enumerate(piv_t.index.get_level_values(PanierVar)) if x==panier]
        ax.text(sum(pos)/len(pos),-11,panier,ha="center",va="top",
                fontsize=11.5,color=STRIP_DARK,fontweight="bold")
    for row_i,col_i in np.ndindex(piv_t.shape):
        vm=piv_m.iloc[row_i,col_i];vp=piv_t.iloc[row_i,col_i]
        if vm!=0 and vp>=8:
            p=ax.patches[col_i*len(piv_t)+row_i]
            ax.text(p.get_x()+p.get_width()/2,p.get_y()+p.get_height()/2,
                    f"{_fmt_euro(vm)}",ha="center",va="center",
                    color=BG,fontweight="bold",fontsize=13.5)
                    # , path_effects=[pe.withStroke(linewidth=2, foreground=STRIP_DARK)],fontweight="bold",fontsize=10.5)
            leg = ax.legend(loc="lower center",bbox_to_anchor=(0.5,-0.26),
            ncol=min(4,len(piv_t.columns)),fontsize=12,frameon=False)
            leg.get_title().set_color("#173A64")
    
    _finalize(fig,f"{Emplacement_stockage}/{titre}Ventilation_coûts_.jpg",qualitéGraphique)

# ── 11. HISTOGRAMME TRANCHES ──────────────────────────────────────────────────

def Sous_famille_comparaison_montants(data,var,qualitéGraphique,Emplacement_stockage):
    df=(data[data[var]>0].groupby(["annee_soins",var]).size().reset_index(name="count"))
    bins,lbl=optimal_bins(data[data[var]>0][var],min_bin_size=0.05,max_bin_size=0.2,initial_bins=20)
    df["tranche"]=pd.cut(df[var],bins=bins,labels=lbl,right=False)
    grouped=(df.groupby(["tranche","annee_soins"])["count"].sum()
             .reset_index().rename(columns={"annee_soins":"Année"}))
    grouped=grouped[grouped["count"]!=0]
    n_annees=grouped["Année"].nunique()
    sf=data["sous_famille"].unique()[0]
    fig=_make_fig(12,6)
    title_h=_strip_title(fig,f"Distribution des montants par tranche  ·  {sf}",
                         subtitle=f"Variable : {var}  ·  Nombre d'occurrences")
    ax=_add_main_ax(fig,[0.07,0.06,0.88,1-title_h-0.04])
    sns.barplot(grouped,x="tranche",y="count",hue="Année",
                palette=HEX[::3][:n_annees],ax=ax,
                hue_order=sorted(grouped["Année"].unique(),reverse=True),
                edgecolor=BG,linewidth=0.8,zorder=3)
    _apply_ax_theme(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_k))
    ax.set_xlabel("Tranches de montants (€)",fontsize=11)
    ax.set_ylabel("Occurrences",fontsize=11)
    ax.tick_params(axis='y', colors="#173A64")
    ax.tick_params(axis='x', colors="#173A64")
    plt.xticks(rotation=38,ha="right",fontsize=9.5)
    ax.legend(title="Survenance",fontsize=10.5,frameon=False,loc="best")
    title=f"Distribution montants {sf} {var}"
    _finalize(fig,f"{Emplacement_stockage}/{title}.jpg",qualitéGraphique)

# ── 12. COMPOSANTES DE LA DÉPENSE ────────────────────────────────────────────

def etude_composante_dépense(data,variable_prix,sf,Emplacement_stockage,
                              qualitéGraphique,layout="vertical"):
    data=data.copy();data["annee_soins"]=data["annee_soins"].astype(int).astype(str)
    if "nb_acte" not in data.columns: st.error('"nb_acte" manquant.'); return
    inter=pd.pivot_table(data,values=[variable_prix,"nb_acte"],
                         index=["id_beneficiaire","annee_soins"],aggfunc="sum").reset_index()
    inter_bis=pd.pivot_table(inter,values=[variable_prix,"nb_acte"],
                              index="annee_soins",aggfunc="mean")
    inter_=pd.pivot_table(inter,values=[variable_prix,"nb_acte"],
                          index="annee_soins",aggfunc="sum")
    inter_["prix_actes"]=inter_[variable_prix]/inter_["nb_acte"]
    df_graph=pd.concat([inter_bis,inter_[["prix_actes"]]],axis=1).reset_index()
    noms={"RC":"remboursement complémentaire","rbt_ss":"remboursement Sécurité Sociale",
          "frais_reels":"frais réels"}
    nom_var=noms.get(variable_prix,variable_prix)
    nom_fichier=f"{sf} - Composantes dépense - {nom_var}"
    infos=[("Coût moyen / personne",variable_prix,HEX[2],"o"),
           ("Coût moyen / acte","prix_actes",HEX[6],"s"),
           ("Nb d'actes / personne","nb_acte",HEX[8],"^")]
    if layout=="horizontal":
        fig,axes=plt.subplots(1,3,figsize=(16,5),facecolor=BG)
        fig.subplots_adjust(top=0.82,bottom=0.12,left=0.06,right=0.97,wspace=0.35)
    else:
        fig,axes=plt.subplots(3,1,figsize=(7,10),facecolor=BG,sharex=True)
        fig.subplots_adjust(top=0.88,bottom=0.07,left=0.14,right=0.92,hspace=0.4)
    for ax, (titre_ax, col, col_color, marker) in zip(axes, infos):
        ax.plot(df_graph["annee_soins"], df_graph[col], marker=marker, color=col_color,
            linewidth=2.5, markersize=8, zorder=3, markerfacecolor="white",
            markeredgewidth=2.2, markeredgecolor=col_color)
        for _, row in df_graph.iterrows():
            ax.annotate(f"{row[col]:,.0f}".replace(",", " "),
                xy=(row["annee_soins"], row[col]),
                xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=9, color=col_color, fontweight="bold")
                
        ax.set_title(titre_ax, fontsize=11, color=STRIP_DARK, fontweight="bold", pad=8)
        ax.set_facecolor(BG_AX)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6, color=C_GRID)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(labelsize=10, colors="#666E88", length=0)


    y_max_01 = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    marge = y_max_01 * 0.15

    y_max_02 = max(axes[1].get_ylim()[1], axes[1].get_ylim()[1])
    marge2 = y_max_02 * 0.15

    axes[0].set_ylim(0, y_max_01 + marge)
    axes[1].set_ylim(0, y_max_02 + marge2)

# Le 3ème garde sa propre échelle, juste floor à 0
    y_max_2 = axes[2].get_ylim()[1]
    axes[2].set_ylim(0, y_max_2 * 1.15)
    
    
    fig.suptitle(f"{sf}  ·  Évolution des composantes  ·  {nom_var}",
                 fontsize=13,fontweight="bold",color=STRIP_DARK,
                 y=0.97 if layout=="horizontal" else 0.99)
    for ax in axes:
        ax.tick_params(axis='y', colors="#173A64")
        ax.tick_params(axis='x', colors="#173A64")

    fig.text(0.5,0.005,
             "Lecture : chaque graphique représente une composante indépendante de la dépense.",
             ha="center",fontsize=8.5,color="#173A64",style="italic")
    _finalize(fig,os.path.join(Emplacement_stockage,nom_fichier+".jpg"),qualitéGraphique)

# ── 13. DISPERSION COMPARAISON ────────────────────────────────────────────────

def dispertion_chart_comparaison(df,var_montant,element_titre,
                                  qualitéGraphique,Emplacement_stockage):
    table=(pd.pivot_table(df,values=[var_montant,"nb_acte"],
                          index=["id_beneficiaire","annee_soins"],
                          aggfunc="sum").reset_index())
    table=table[table[var_montant]>0]
    bins,lbl=calculer_bins_labels_equilibres(table,var_montant,
                                              min_pct=0.05,max_pct=0.3,
                                              max_bins=20,multiple=5)
    table["Tranche"]=pd.cut(table[var_montant],bins=bins,labels=lbl,right=False)
    t=(pd.pivot_table(table,values=[var_montant,"nb_acte","id_beneficiaire"],
                      index=["Tranche","annee_soins"],observed=False,
                      aggfunc={var_montant:"sum","nb_acte":"sum",
                               "id_beneficiaire":pd.Series.nunique})
       .reset_index())
    # Remplissage sélectif (évite l'erreur sur colonnes catégorielles)
    for col in [var_montant,"nb_acte","id_beneficiaire"]:
        t[col]=t[col].fillna(0)
    t.rename(columns={"annee_soins":"Survenance"},inplace=True)
    n_annees=t["Survenance"].nunique()

    bar_colors=[HEX[1],HEX[4]] if n_annees<=2 else HEX[:n_annees]
    main_title=(f"{element_titre[0]}  ·  Répartition des consommants selon le remboursement complémentaire")
    
    fig=_make_fig(13,7)

    title_h=_strip_title(fig,main_title)

    ax=_add_main_ax(fig,[0.06,0.10,0.90,1-title_h-0.04-0.10])
    sns.barplot(x="Tranche",y=var_montant,hue="Survenance",data=t,ax=ax,
                palette=bar_colors,
                hue_order=sorted(t["Survenance"].unique()),
                edgecolor=BG,linewidth=0.8,zorder=3)
    _apply_ax_theme(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_euro))
    ax.set_xlabel("Remboursement Complémentaire" ,color="#173A64", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(axis='y', colors="#173A64",labelsize=13)
    ax.tick_params(axis='x', colors="#173A64",labelsize=13)
    plt.xticks(rotation=0,color = "#173A64",ha="center",fontsize=12.5)
    moy_rc=t[var_montant].replace(0,np.nan).mean()
    for i, p in enumerate(ax.patches):

        val_rc = p.get_height()
        match = t.loc[t[var_montant] == val_rc, "id_beneficiaire"]
        color = p.get_facecolor()
    

        if not match.empty and val_rc > 0:

            nb = match.values[0]

            ax.annotate(
                f"{formatM(nb)}",
                xy=(p.get_x() + p.get_width()/2., val_rc ),
                xytext=(0, 4),          # ← juste 4 pixels au-dessus de la barre
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                color= color,
                fontsize=17,
                rotation=23
        )
    ax.legend(loc="best",fontsize=12.5,frameon=False,reverse = True)
    title=f"Dispersion RC {'_'.join(element_titre)}"
    _finalize(fig,f"{Emplacement_stockage}/dispersion_conso_{''.join(element_titre)}.jpg",
              qualitéGraphique)



### 14 proportion des prest pae categorie 


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
    ax1.set_xticklabels(years, fontsize=15)
    ax1.yaxis.set_major_formatter(FuncFormatter(_fmt_euro))
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='y', colors="#173A64",labelsize=15)
    ax1.tick_params(axis='x', colors="#173A64",labelsize=15)
    ax1.tick_params(axis="x", length=0.0)

    # Annotations montants sur barres
    for bar, val in zip(bars, rc_porta.values):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() * 1.012,
                 _fmt_euro(val),
                 ha="center", va="bottom",
                 fontsize=18, fontweight="bold",zorder = 10, color=HEX[2])

    # ── Courbe % (axe droit) ──────────────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.plot(x, prop_porta.values * 100,
             color=HEX[8], linewidth=2.5, marker="o", markersize=7,
             zorder=1, markerfacecolor="white",
             markeredgewidth=2, markeredgecolor=HEX[8])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f} %"))
    ax2.tick_params(axis="y", labelsize=15, colors="#173A64", length=0)
    for sp in ax2.spines.values(): sp.set_visible(False)
    ax2.set_ylim(bottom=0,
                 top=max(prop_porta.values) * 100 * 1.35)
    ax2.grid(False)

    # Annotations % au-dessus des points
    for xi, pct in zip(x, prop_porta.values):
        ax2.annotate(
            f"{pct*100:.1f} %",
            zorder=2,
            xy=(xi, pct * 100),
            xytext=(0, -30), textcoords="offset points",
            ha="center", fontsize=17, fontweight="bold",
            color=HEX[8],
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="none", alpha=0)
        )

    # ── Légende ───────────────────────────────────────────────────────────────
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    patch = mpatches.Patch(color=HEX[1], label=f"RC")
    line  = mlines.Line2D([], [], color=HEX[8], marker="o",
                          markersize=8, markerfacecolor="white",
                          markeredgewidth=1.5, label="Part dans le total")
    ax1.legend(handles=[patch, line],
               loc="lower center", bbox_to_anchor=(0.5, -0.18),
               ncol=2, fontsize=13.5, frameon=False)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    title_safe = (label_cat.replace(' ', '_')
                            .replace(',', '').replace(':', '')
                            .replace('-', '_'))
    filepath = f"{Emplacement_stockage}/RC_proportion_{title_safe}.jpg"
    _finalize(fig, filepath, 150)