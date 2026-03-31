import streamlit as st
import zipfile
import io
from pathlib import Path
import shutil 



try:
    from pages_AppliAnalyseSanté_V2_1.page_tableConso import tableConso
    from pages_AppliAnalyseSanté_V2_1.page_dispersion_couts_an import Dispersion
    from pages_AppliAnalyseSanté_V2_1.page_analyse_generale import Analyse_generale
    from pages_AppliAnalyseSanté_V2_1.page_comparaison_survenances import comparaison_survenances
    from pages_AppliAnalyseSanté_V2_1.page_cadencement_PSAP import cadencement_PSAP
    from pages_AppliAnalyseSanté_V2_1.page_100p100Santé import _100p100Santé
    from pages_AppliAnalyseSanté_V2_1.page_etude_sous_famille import etude_sous_famille
    from pages_AppliAnalyseSanté_V2_1.page_etude_prix import etude_prix
    from pages_AppliAnalyseSanté_V2_1.page_effectifs import page_effectifs
except ImportError as e:
    st.error(f"Erreur lors de l'importation des modules : {e}")
    st.stop()

st.set_page_config(page_title="Application Consommation Santé", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
if "donnees"           not in st.session_state: st.session_state["donnees"]           = None
if "repertoire_images" not in st.session_state: st.session_state["repertoire_images"] = None
if "Qualité images"    not in st.session_state: st.session_state["Qualité images"]    = 120
if "page"              not in st.session_state: st.session_state["page"]              = "Accueil"
if "galerie_selection" not in st.session_state: st.session_state["galerie_selection"] = set()

PAGES = [
    "Accueil", "Tables consommations",
    "Dispersion des coûts", "Analyse générale","Effectifs",
    "Comparaison entre survenances", "Etude sous famille",
    "Etude prix", "100% santé", "Cadencements & PSAP",
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
current_index = PAGES.index(st.session_state["page"])
page = st.sidebar.selectbox("Navigation", PAGES, index=current_index)
st.session_state["page"] = page


from fonctions.workOnData import pad_column_with_zeros
from fonctions.workOnData import load_csv


def charger_donnees():
    fichier = st.file_uploader("Charger un fichier CSV", type=["csv"])
    if fichier is not None:
        try:
            df = load_csv(fichier)
            df["id_beneficiaire"] = pad_column_with_zeros(df["id_beneficiaire"])
            df["id_assure"] = pad_column_with_zeros(df["id_assure"])

            st.session_state["donnees"] = df
            st.success("Données chargées avec succès !")

            export_dir = init_export_dir()
            st.session_state["repertoire_images"] = str(export_dir)

        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")


def init_export_dir():
    # Définir le répertoire d'export par défaut
    export_dir = Path(__file__).resolve().parents[1] / "exports" / "images"

    # Si le dossier existe déjà, supprimer son contenu
    if export_dir.exists():
        shutil.rmtree(export_dir)

    # Recréer le dossier vide
    export_dir.mkdir(parents=True, exist_ok=True)


    return export_dir




# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS GALERIE
# ══════════════════════════════════════════════════════════════════════════════

def _get_images():
    rep = st.session_state.get("repertoire_images")
    if not rep:
        return []
    return sorted(
        [p for p in Path(rep).glob("*")
         if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

def _build_zip(paths):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=p.name)
    buf.seek(0)
    return buf

def _galerie():
    images = _get_images()

    st.markdown("---")
    st.markdown("### Galerie des images générées")

    if not images:
        st.info("Aucune image générée pour l'instant. Lancez une analyse pour voir vos exports ici.")
        return

    # ── Barre d'outils ────────────────────────────────────────────────────────
    col_info, col_all, col_none, col_dl = st.columns([2, 1, 1, 2])

    with col_info:
        n_sel = len(st.session_state["galerie_selection"])
        st.markdown(
            f"<span style='font-size:13px;color:#4A5568;'>"
            f"<b style='color:#1A2440;'>{len(images)}</b> image(s) · "
            f"<b style='color:#2C67AF;'>{n_sel}</b> sélectionnée(s)</span>",
            unsafe_allow_html=True)

    with col_all:
        if st.button("✅ Tout sélectionner", use_container_width=True, key="sel_all"):
            st.session_state["galerie_selection"] = {p.name for p in images}
            for p in images:
                st.session_state[f"chk_{p.name}"] = True
            st.rerun()

    with col_none:
        if st.button("✖ Désélectionner", use_container_width=True, key="sel_none"):
            st.session_state["galerie_selection"] = set()
            for p in images:
                st.session_state[f"chk_{p.name}"] = False
            st.rerun()

    with col_dl:
        selected_paths = [p for p in images
                          if p.name in st.session_state["galerie_selection"]]
        n_sel = len(selected_paths)
        if n_sel == 0:
            st.button("⬇ Télécharger (0 sélectionnée)",
                      disabled=True, use_container_width=True, key="dl_disabled")
        else:
            zip_buf = _build_zip(selected_paths)
            st.download_button(
                label=f"⬇ Télécharger ({n_sel} image{'s' if n_sel > 1 else ''})",
                data=zip_buf,
                file_name="selection_images.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_zip")

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    # ── Grille 3 colonnes ─────────────────────────────────────────────────────
    N_COLS = 3
    rows = [images[i:i+N_COLS] for i in range(0, len(images), N_COLS)]

    for row in rows:
        cols = st.columns(N_COLS)
        for col, img_path in zip(cols, row):
            with col:
                is_selected  = img_path.name in st.session_state["galerie_selection"]
                border_color = "#2C67AF" if is_selected else "#E2E8F0"
                border_width = "2px"     if is_selected else "0.5px"

                st.markdown(
                    f"<div style='border:{border_width} solid {border_color};"
                    f"border-radius:10px;overflow:hidden;margin-bottom:4px;'>",
                    unsafe_allow_html=True)
                st.image(str(img_path), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                size_kb = img_path.stat().st_size // 1024
                st.markdown(
                    f"<div style='font-size:11.5px;font-weight:600;color:#1A2440;"
                    f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                    f"margin-bottom:1px;' title='{img_path.name}'>{img_path.stem}</div>"
                    f"<div style='font-size:10px;color:#888;margin-bottom:4px;'>"
                    f"PNG · {size_kb} Ko</div>",
                    unsafe_allow_html=True)

                checked = st.checkbox(
                    "Sélectionner",
                    value=is_selected,
                    key=f"chk_{img_path.name}")

                if checked and img_path.name not in st.session_state["galerie_selection"]:
                    st.session_state["galerie_selection"].add(img_path.name)
                    st.rerun()
                elif not checked and img_path.name in st.session_state["galerie_selection"]:
                    st.session_state["galerie_selection"].discard(img_path.name)
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state["page"] == "Accueil":

    st.markdown("""
<style>
@keyframes fadeUp  {from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideIn {from{opacity:0;transform:translateX(-20px)}to{opacity:1;transform:translateX(0)}}
@keyframes pulse   {0%,100%{opacity:1}50%{opacity:.55}}
@keyframes countUp {from{opacity:0;transform:scale(.85)}to{opacity:1;transform:scale(1)}}
@keyframes wave    {0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
.hero{padding:2.5rem 0 1.5rem;text-align:center}
.hero-tag{display:inline-flex;gap:0;font-size:16px;letter-spacing:.12em;text-transform:uppercase;background:#EEF1F9;padding:10px 22px;border-radius:28px;margin-bottom:1rem;animation:fadeUp 2s ease both}
.hero-tag span{display:inline-block;color:#2C67AF;animation:wave 2s ease-in-out infinite;}
.hero-tag span:nth-child(1){animation-delay:0s}
.hero-tag span:nth-child(2){animation-delay:.07s}
.hero-tag span:nth-child(3){animation-delay:.14s}
.hero-tag span:nth-child(4){animation-delay:.21s}
.hero-tag span:nth-child(5){animation-delay:.28s}
.hero-tag span:nth-child(6){animation-delay:.35s}
.hero-tag span:nth-child(7){animation-delay:.42s}
.hero-tag span:nth-child(8){animation-delay:.49s}
.hero-tag span:nth-child(9){animation-delay:.56s}
.hero-tag span:nth-child(10){animation-delay:.63s}
.hero-tag span:nth-child(11){animation-delay:.70s}
.hero-tag span:nth-child(12){animation-delay:.77s}
.hero-tag span:nth-child(13){animation-delay:.84s}
.hero-tag span:nth-child(14){animation-delay:.91s}
.hero-tag span:nth-child(15){animation-delay:.98s}
.hero-tag span:nth-child(16){animation-delay:1.05s}
.hero-tag span:nth-child(17){animation-delay:1.12s}
.hero-tag span:nth-child(18){animation-delay:1.19s}
.hero-tag span:nth-child(19){animation-delay:1.26s}
.hero-tag span:nth-child(20){animation-delay:1.33s}
.hero-tag span:nth-child(21){animation-delay:1.40s}
.hero-tag span:nth-child(22){animation-delay:1.47s}
.hero-tag span:nth-child(23){animation-delay:1.54s}
.hero-tag span:nth-child(24){animation-delay:1.61s}
.hero-tag span:nth-child(25){animation-delay:1.68s}
.hero-title span{color:#2C67AF}
.hero-sub{font-size:14px;color:#4A5568;animation:fadeUp .6s ease .2s both;margin-bottom:2rem}
.divider{width:48px;height:3px;background:#2C67AF;margin:0 auto 2rem;border-radius:2px;animation:fadeUp .6s ease .3s both}
.kpi-row{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-bottom:2rem}
.kpi{background:#F7F8FC;border:.5px solid #E2E8F0;border-radius:10px;padding:1rem 1.5rem;min-width:130px;text-align:center;border-top:3px solid #2C67AF;animation:countUp .5s ease both}
.kpi:nth-child(2){border-top-color:#2B3885;animation-delay:.1s}
.kpi:nth-child(3){border-top-color:#662064;animation-delay:.2s}
.kpi:nth-child(4){border-top-color:#F56C26;animation-delay:.3s}
.kpi-val{font-size:22px;font-weight:700;color:#1A2440}
.kpi-lbl{font-size:11px;color:#666E88;margin-top:2px}
.nav-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:1rem}
.nav-card{background:white;border:.5px solid #E2E8F0;border-radius:10px;padding:1rem;border-left:3px solid transparent;animation:slideIn .5s ease both}
.nav-card:nth-child(1){animation-delay:.10s;border-left-color:#4295CE}
.nav-card:nth-child(2){animation-delay:.15s;border-left-color:#2C67AF}
.nav-card:nth-child(3){animation-delay:.20s;border-left-color:#2B3885}
.nav-card:nth-child(4){animation-delay:.25s;border-left-color:#662064}
.nav-card:nth-child(5){animation-delay:.30s;border-left-color:#9B406D}
.nav-card:nth-child(6){animation-delay:.35s;border-left-color:#F56C26}
.nav-icon{font-size:20px;margin-bottom:6px}
.nav-title{font-size:12.5px;font-weight:600;color:#1A2440}
.nav-desc{font-size:11px;color:#666E88;margin-top:2px;line-height:1.4}
.status-bar{display:flex;align-items:center;justify-content:center;gap:8px;font-size:12px;color:#4A5568;animation:fadeUp .6s ease .5s both;margin-top:1rem}
.dot{width:8px;height:8px;border-radius:50%;background:#2B9E6E;animation:pulse 2s ease-in-out infinite;display:inline-block}
</style>
<div class="hero">
  <div class="hero-tag"><span class="w00">A</span><span class="w01">p</span><span class="w02">p</span><span class="w03">l</span><span class="w04">i</span><span class="w05">c</span><span class="w06">a</span><span class="w07">t</span><span class="w08">i</span><span class="w09">o</span><span class="w10">n</span><span>&nbsp;</span><span class="w11">A</span><span class="w12">n</span><span class="w13">a</span><span class="w14">l</span><span class="w15">y</span><span class="w16">s</span><span class="w17">e</span><span>&nbsp;</span><span class="w18">S</span><span class="w19">a</span><span class="w20">n</span><span class="w21">t</span><span class="w22">é</span></div>
  <div class="hero-sub">Analyse de consommation ·
  <div class="divider"></div>
</div>
""", unsafe_allow_html=True)



    # Boutons navigation
    nav_items = [
        ("📊 Tables consommations",   "Tables consommations"),
        ("📈 Analyse générale",       "Analyse générale"),
        ("🔍 Dispersion des coûts",   "Dispersion des coûts"),
        ("⚖️ Comparaison survenances", "Comparaison entre survenances"),
        ("🏥 100 % Santé",            "100% santé"),
    ]
    cols = st.columns(3)
    for i, (label, target) in enumerate(nav_items):
        with cols[i % 3]:
            if st.button(label, key=f"nav_{i}", use_container_width=True):
                st.session_state["page"] = target
                st.rerun()

    # Statut
    if st.session_state["donnees"] is None:
        st.markdown("""
<div class="status-bar">
</div>""", unsafe_allow_html=True)
    else:
        n = len(st.session_state["donnees"])
        st.markdown(f"""
<div class="status-bar">
  <span class="dot"></span>
  Données chargées &mdash; <strong style="color:#1A2440;">{n:,} lignes</strong>
</div>""", unsafe_allow_html=True)

    # ── Chargement données intégré ──────────────────────────────────────────
    st.markdown("---")
    with st.expander("📂 Charger un fichier de données",
                     expanded=st.session_state["donnees"] is None):
        charger_donnees()
        if st.session_state["donnees"] is not None:
            st.dataframe(st.session_state["donnees"].head(),
                         use_container_width=True)

    # Galerie
    _galerie()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state["page"] == "Charger les données":
    st.title("Chargement des données")
    charger_donnees()
    if st.session_state["donnees"] is not None:
        st.write("Aperçu des données :")
        st.dataframe(st.session_state["donnees"].head())
        try:
            for date_col in st.session_state["donnees"].columns:
                if "date" in date_col.lower():
                    st.write(f"{date_col} : "
                             f"{st.session_state['donnees'][date_col].min()} "
                             f"— {st.session_state['donnees'][date_col].max()}")
        except Exception as e:
            st.error(f"Erreur lors de l'affichage des dates : {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGES ANALYSES
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state["page"] in [
    "Tables consommations", "Dispersion des coûts", "Analyse générale",
    "Comparaison entre survenances", "100% santé", "Etude sous famille",
    "Etude prix", "Cadencements & PSAP"
]:
    if st.session_state["donnees"] is None:
        st.warning("Veuillez d'abord charger un jeu de données.")
        if st.button("← Aller au chargement des données"):
            st.session_state["page"] = "Charger les données"
            st.rerun()
    else:
        st.title(st.session_state["page"])
        p = st.session_state["page"]
        if   p == "Tables consommations":          tableConso()
        elif p == "Dispersion des coûts":          Dispersion()
        elif p == "Analyse générale":              Analyse_generale()
        elif p == "Comparaison entre survenances": comparaison_survenances()
        elif p == "100% santé":                    _100p100Santé()
        elif p == "Etude sous famille":            etude_sous_famille()
        elif p == "Etude prix":                    etude_prix()
        elif p == "Effectifs":                     page_effectifs()
        elif p == "Cadencements & PSAP":           cadencement_PSAP()
