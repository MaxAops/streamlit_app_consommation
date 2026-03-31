"""
charts_effectifs.py  —  Graphiques page Effectifs
Même charte graphique que charts.py.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from fonctions.workOnData import formatM

# ── Design system (identique à charts.py) ────────────────────────────────────
HEX = ["#4295CE","#2C67AF","#2B3885","#2A0C53",
       "#662064","#9B406D","#D86173","#F56C26",
       "#EE9744","#EE9780"]

BG          = "#F7F8FC"
BG_AX       = "#FFFFFF"
STRIP_DARK  = "#1A2440"
STRIP_LIGHT = "#EEF1F9"
C_TITLE     = "#FFFFFF"
C_SUBTITLE  = "#3A4A72"
C_ANNOT     = "#2B3885"
C_GRID      = "#E4E8F0"
C_KPIBG     = "#EEF1F9"
C_KPIBORDER = "#2C67AF"
C_POS       = "#2B9E6E"
C_NEG       = "#D85050"

MOIS = ['Janv','Fév','Mars','Avr','Mai','Juin',
        'Juil','Août','Sept','Oct','Nov','Déc']

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "axes.grid": True, "grid.color": C_GRID,
    "grid.linewidth": 0.7, "grid.linestyle": "--",
    "axes.facecolor": BG_AX, "figure.facecolor": BG,
    "xtick.color": "#666E88", "ytick.color": "#666E88",
    "xtick.labelsize": 11, "ytick.labelsize": 11,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_k(x, _=None):
    return f"{x/1_000:.1f} k" if abs(x) >= 1_000 else f"{x:.0f}"

def _sign(v):
    return f"+{v:.1f} %" if v >= 0 else f"{v:.1f} %"

def _color_sign(v):
    return C_POS if v >= 0 else C_NEG

def _apply_ax_theme(ax):
    ax.set_facecolor(BG_AX)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7, color=C_GRID)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10.5, colors="#666E88", length=0)

def _make_fig(w=14, h=7):
    return plt.figure(figsize=(w, h), facecolor=BG)

def _add_main_ax(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG_AX)
    return ax

def _strip_title(fig, title, subtitle=None, strip_h=0.055, sub_h=0.032):
    fig.add_axes([0, 1-strip_h, 1, strip_h], zorder=10)
    ax_t = fig.axes[-1]
    ax_t.set_facecolor(STRIP_DARK)
    ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)
    for sp in ax_t.spines.values(): sp.set_visible(False)
    ax_t.set_xticks([]); ax_t.set_yticks([])
    ax_t.text(0.018, 0.5, title, color=C_TITLE, fontsize=13,
              fontweight="bold", va="center", transform=ax_t.transAxes)
    if subtitle:
        top_sub = 1 - strip_h
        fig.add_axes([0, top_sub-sub_h, 1, sub_h], zorder=9)
        ax_s = fig.axes[-1]
        ax_s.set_facecolor(STRIP_LIGHT)
        ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1)
        for sp in ax_s.spines.values(): sp.set_visible(False)
        ax_s.set_xticks([]); ax_s.set_yticks([])
        ax_s.text(0.018, 0.5, subtitle, color=C_SUBTITLE, fontsize=10,
                  va="center", transform=ax_s.transAxes)
        return strip_h + sub_h
    return strip_h

def _kpi_row(fig, kpis, bottom=0.0, height=0.13):
    n = len(kpis); w = 1/n
    for i, (lbl, val, delta) in enumerate(kpis):
        ax = fig.add_axes([i*w+0.005, bottom+0.005, w-0.010, height-0.010], zorder=8)
        ax.set_facecolor(C_KPIBG)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.axvline(0, color=C_KPIBORDER, linewidth=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.08, 0.72, lbl, color=C_SUBTITLE, fontsize=8.5,
                va="center", transform=ax.transAxes)
        ax.text(0.08, 0.38, val, color=STRIP_DARK, fontsize=13,
                fontweight="bold", va="center", transform=ax.transAxes)
        if delta is not None:
            col = _color_sign(float(str(delta).replace("+","").replace(" %","")))
            ax.text(0.08, 0.10, str(delta), color=col, fontsize=9,
                    fontweight="bold", va="center", transform=ax.transAxes)

def _finalize(fig, path, dpi):
    MAX_DPI = 200; MAX_WIDTH = 3200
    w_inch = fig.get_size_inches()[0]
    dpi_eff = min(dpi, MAX_DPI)
    if w_inch * dpi_eff > MAX_WIDTH:
        dpi_eff = int(MAX_WIDTH / w_inch)
    fig.patch.set_facecolor(BG)
    png_path = os.path.splitext(path)[0] + ".png"
    plt.savefig(png_path, dpi=dpi_eff, bbox_inches="tight",
                pad_inches=0.20, facecolor=BG, edgecolor="none", format="png")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. KPIs EFFECTIFS ─────────────────────────────────────────────────────────

def kpis_effectifs(df, df_presta=None):
    """
    Affiche les KPIs principaux en haut de page via st.columns.
    df       : DataFrame effectifs
    df_presta: DataFrame prestations (optionnel, pour taux de consommation)
    """
    nb_assures    = df["id_assure"].nunique()
    nb_beneficiaires = df["id_beneficiaire"].nunique()
    ratio         = round(nb_beneficiaires / nb_assures, 2) if nb_assures > 0 else 0

    kpis = [
        ("Assurés",        f"{nb_assures:,}".replace(",", " "),       None),
        ("Bénéficiaires",  f"{nb_beneficiaires:,}".replace(",", " "), None),
        ("Ratio bénéf. / assuré", f"{ratio:.2f}",                     None),
    ]

    if df_presta is not None and "id_beneficiaire" in df_presta.columns:
        nb_conso = df_presta["id_beneficiaire"].nunique()
        taux_conso = round(nb_conso / nb_beneficiaires * 100, 1) if nb_beneficiaires > 0 else 0
        kpis.append(("Taux de consommation", f"{taux_conso} %", None))

    cols = st.columns(len(kpis))
    for col, (lbl, val, _) in zip(cols, kpis):
        with col:
            st.markdown(
                f"<div style='background:{C_KPIBG};border-left:3px solid {C_KPIBORDER};"
                f"border-radius:8px;padding:0.8rem 1rem;'>"
                f"<div style='font-size:11px;color:{C_SUBTITLE};margin-bottom:4px;'>{lbl}</div>"
                f"<div style='font-size:22px;font-weight:700;color:{STRIP_DARK};'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True)


# ── 2. ÉVOLUTION MENSUELLE EFFECTIFS ─────────────────────────────────────────

def evolution_effectifs(df, qualitéGraphique, Emplacement_stockage):
    """
    Courbe d'évolution mensuelle des bénéficiaires actifs.
    Reconstruit les effectifs présents chaque mois depuis
    date_affiliation et date_sortie_beneficiaire.
    """
    df = df.copy()

    # Conversion dates
    df["date_affiliation"] = pd.to_datetime(df["date_affiliation"], errors="coerce")
    df["date_sortie_beneficiaire"] = pd.to_datetime(df["date_sortie_beneficiaire"], errors="coerce")

    # Déterminer la plage de mois à couvrir
    date_min = df["date_affiliation"].min()
    date_max = df["date_sortie_beneficiaire"].dropna().max()
    if pd.isna(date_max):
        date_max = pd.Timestamp.today()

    mois_range = pd.date_range(
        start=date_min.to_period("M").to_timestamp(),
        end=date_max.to_period("M").to_timestamp(),
        freq="MS"
    )

    # Compter les bénéficiaires actifs chaque mois
    counts = []
    for mois in mois_range:
        fin_mois = mois + pd.offsets.MonthEnd(0)
        mask = (
            (df["date_affiliation"] <= fin_mois) &
            (df["date_sortie_beneficiaire"].isna() | (df["date_sortie_beneficiaire"] >= mois))
        )
        counts.append({"mois": mois, "effectif": mask.sum()})

    evo = pd.DataFrame(counts)
    evo["annee"] = evo["mois"].dt.year
    evo["mois_num"] = evo["mois"].dt.month

    annees = sorted(evo["annee"].unique())
    n_annees = len(annees)
    line_colors = HEX[::3][:n_annees]

    fig = _make_fig(14, 6)
    title_h = _strip_title(fig,
        "Évolution mensuelle des effectifs bénéficiaires",
        subtitle="Bénéficiaires actifs reconstitués par mois")
    kpi_h = 0.0
    ax = _add_main_ax(fig, [0.07, 0.10, 0.88, 1 - title_h - 0.04 - 0.10])

    for annee, col in zip(annees, line_colors):
        sub = evo[evo["annee"] == annee]
        ax.plot(sub["mois_num"], sub["effectif"],
                color=col, linewidth=2.8, marker="o", markersize=5,
                markerfacecolor="white", markeredgewidth=2,
                markeredgecolor=col, label=str(annee), zorder=3)
        if annee == annees[-1]:
            ax.fill_between(sub["mois_num"], sub["effectif"],
                            alpha=0.07, color=col, zorder=2)

    _apply_ax_theme(ax)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MOIS, fontsize=10.5)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_k))
    ax.set_ylabel("Bénéficiaires actifs", fontsize=11)
    ax.legend(title="Année", fontsize=10.5, title_fontsize=10.5,
              frameon=False, loc="best")

    _finalize(fig, f"{Emplacement_stockage}/effectifs_evolution_mensuelle.jpg",
              qualitéGraphique)


# ── 3. RÉPARTITION PAR CATÉGORIE ─────────────────────────────────────────────

def repartition_categories(df, qualitéGraphique, Emplacement_stockage):
    """
    Double graphique : répartition par cat_assure + par type_beneficiaire.
    """
    fig = _make_fig(14, 6)
    title_h = _strip_title(fig,
        "Répartition des effectifs par catégorie",
        subtitle="Catégorie assurés et type de bénéficiaire")
    ax1 = _add_main_ax(fig, [0.05, 0.14, 0.42, 1 - title_h - 0.04 - 0.14])
    ax2 = _add_main_ax(fig, [0.55, 0.14, 0.42, 1 - title_h - 0.04 - 0.14])

    # ── Graphique 1 : cat_assure ─────────────────────────────────────────────
    if "cat_assure" in df.columns:
        cat = (df.groupby("cat_assure")["id_beneficiaire"]
               .nunique().sort_values(ascending=True).reset_index())
        cat.columns = ["Catégorie", "Effectif"]

        bars = ax1.barh(cat["Catégorie"], cat["Effectif"],
                        color=HEX[1], edgecolor=BG, linewidth=0.8,
                        alpha=0.88, zorder=3)
        _apply_ax_theme(ax1)
        ax1.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
        ax1.set_title("Par catégorie assurés", fontsize=11,
                      color=STRIP_DARK, fontweight="bold", pad=8)
        ax1.grid(axis="x"); ax1.grid(axis="y", alpha=0)

        for bar in bars:
            val = bar.get_width()
            if val > 0:
                ax1.text(val * 1.02, bar.get_y() + bar.get_height()/2,
                         formatM(val), va="center", fontsize=9,
                         color=C_ANNOT, fontweight="bold")

    # ── Graphique 2 : type_beneficiaire ─────────────────────────────────────
    if "type_beneficiaire" in df.columns:
        typ = (df.groupby("type_beneficiaire")["id_beneficiaire"]
               .nunique().sort_values(ascending=True).reset_index())
        typ.columns = ["Type", "Effectif"]

        bars2 = ax2.barh(typ["Type"], typ["Effectif"],
                         color=HEX[4], edgecolor=BG, linewidth=0.8,
                         alpha=0.88, zorder=3)
        _apply_ax_theme(ax2)
        ax2.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
        ax2.set_title("Par type de bénéficiaire", fontsize=11,
                      color=STRIP_DARK, fontweight="bold", pad=8)
        ax2.grid(axis="x"); ax2.grid(axis="y", alpha=0)

        for bar in bars2:
            val = bar.get_width()
            if val > 0:
                ax2.text(val * 1.02, bar.get_y() + bar.get_height()/2,
                         formatM(val), va="center", fontsize=9,
                         color=C_ANNOT, fontweight="bold")

    _finalize(fig, f"{Emplacement_stockage}/effectifs_categories.jpg",
              qualitéGraphique)


# ── 4. PYRAMIDE DES ÂGES ─────────────────────────────────────────────────────

def pyramide_ages(df, qualitéGraphique, Emplacement_stockage):
    """
    Pyramide des âges classique : hommes à gauche, femmes à droite.
    """
    df = df.copy()

    # Tranches d'âge
    if "tranche d'age" in df.columns:
        col_tranche = "tranche d'age"
    elif "Age" in df.columns:
        bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                50, 55, 60, 65, 70, 75, 80, 200]
        labels = ["0–4","5–9","10–14","15–19","20–24","25–29",
                  "30–34","35–39","40–44","45–49","50–54",
                  "55–59","60–64","65–69","70–74","75–79","80+"]
        df["tranche_age"] = pd.cut(df["Age"], bins=bins,
                                    labels=labels, right=False)
        col_tranche = "tranche_age"
    else:
        st.warning("Colonne d'âge introuvable.")
        return

    # Normalisation genre
    if "genre_beneficiaire" not in df.columns:
        st.warning("Colonne genre_beneficiaire introuvable.")
        return

    df["genre_norm"] = df["genre_beneficiaire"].str.upper().str.strip()
    hommes = df[df["genre_norm"].isin(["M","H","HOMME","MASCULIN"])]
    femmes = df[df["genre_norm"].isin(["F","FEMME","FÉMININ","FEMININ"])]

    tranches = sorted(df[col_tranche].dropna().unique().tolist())

    h_counts = (hommes.groupby(col_tranche)["id_beneficiaire"]
                .nunique().reindex(tranches, fill_value=0))
    f_counts = (femmes.groupby(col_tranche)["id_beneficiaire"]
                .nunique().reindex(tranches, fill_value=0))

    fig = _make_fig(13, 7)
    title_h = _strip_title(fig,
        "Pyramide des âges",
        subtitle=f"Hommes · {int(len(hommes)):,}".replace(",", " ") +
                 f"    Femmes · {int(len(femmes)):,}".replace(",", " "))
    ax = _add_main_ax(fig, [0.12, 0.08, 0.78, 1 - title_h - 0.04 - 0.08])

    y = np.arange(len(tranches))
    ax.barh(y, -h_counts.values, color=HEX[1], alpha=0.88,
            edgecolor=BG, linewidth=0.8, zorder=3, label="Hommes")
    ax.barh(y, f_counts.values, color=HEX[4], alpha=0.88,
            edgecolor=BG, linewidth=0.8, zorder=3, label="Femmes")

    _apply_ax_theme(ax)
    ax.set_yticks(y)
    ax.set_yticklabels(tranches, fontsize=10)
    ax.axvline(0, color=STRIP_DARK, linewidth=1, zorder=4)
    ax.grid(axis="x"); ax.grid(axis="y", alpha=0)

    # Formateur axe X symétrique (valeurs positives des deux côtés)
    max_val = max(h_counts.max(), f_counts.max())
    ticks = np.linspace(-max_val, max_val, 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_k(abs(t)) for t in ticks], fontsize=10)

    ax.legend(fontsize=11, frameon=False, loc="lower right")

    _finalize(fig, f"{Emplacement_stockage}/effectifs_pyramide_ages.jpg",
              qualitéGraphique)


# ── 5. TAUX DE CONSOMMATION ───────────────────────────────────────────────────

def taux_consommation(df_eff, df_presta, qualitéGraphique, Emplacement_stockage):
    """
    Taux de consommation par famille d'actes :
    nb bénéficiaires consommants / nb bénéficiaires présents.
    """
    if df_presta is None or df_presta.empty:
        st.warning("Données de prestations manquantes pour calculer le taux de consommation.")
        return

    nb_total = df_eff["id_beneficiaire"].nunique()
    if nb_total == 0:
        st.warning("Aucun bénéficiaire dans les effectifs.")
        return

    conso = (df_presta.groupby("famille_acte_aops")["id_beneficiaire"]
             .nunique().reset_index())
    conso.columns = ["Famille", "Nb consommants"]
    conso["Taux (%)"] = (conso["Nb consommants"] / nb_total * 100).round(1)
    conso = conso.sort_values("Taux (%)", ascending=True)

    fig = _make_fig(13, 6)
    title_h = _strip_title(fig,
        "Taux de consommation par famille d'actes",
        subtitle=f"Nb bénéficiaires consommants / effectifs totaux  ·  Base : {formatM(nb_total)} bénéficiaires")
    ax = _add_main_ax(fig, [0.22, 0.10, 0.68, 1 - title_h - 0.04 - 0.10])

    colors_bar = [HEX[i % len(HEX)] for i in range(len(conso))]
    bars = ax.barh(conso["Famille"], conso["Taux (%)"],
                   color=colors_bar, edgecolor=BG,
                   linewidth=0.8, alpha=0.88, zorder=3)
    _apply_ax_theme(ax)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}%"))
    ax.set_xlim(0, min(100, conso["Taux (%)"].max() * 1.25))
    ax.grid(axis="x"); ax.grid(axis="y", alpha=0)
    ax.tick_params(axis="y", labelsize=11)

    # Annotations
    for bar, (_, row) in zip(bars, conso.iterrows()):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f"{row['Taux (%)']:.1f} %  ({formatM(row['Nb consommants'])})",
                va="center", fontsize=9.5,
                color=C_ANNOT, fontweight="bold")

    _finalize(fig, f"{Emplacement_stockage}/effectifs_taux_consommation.jpg",
              qualitéGraphique)
