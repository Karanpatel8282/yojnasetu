"""
app.py  —  YojanaSetu (final UI)
====================================
Premium dark-ui Streamlit app for the Marathi government-scheme
retrieval engine.  Tidy single file — read top to bottom.

Flow
  NAVBAR -> SIDEBAR(profile + language) -> HERO + SEARCH
         -> CATEGORY filter -> STAT cards -> RANKED result cards
         -> Tabs: Recommendations | Explore dataset

The engine (engine.py) does TF-IDF ranking + eligibility; this file is
only the presentation layer.
"""
import html as _html
import pandas as pd
import streamlit as st
from collections import Counter

from engine import load_schemes, build_index, rank, DATA_PATH

st.set_page_config(page_title="YojanaSetu", page_icon="🏛️", layout="wide")

# =====================================================================
# 1. DESIGN SYSTEM  (custom CSS — a premium look the user requested)
# =====================================================================
CSS = """
<style>
:root{--accent:#10b981;--accent2:#8b5cf6;--accent3:#3b82f6;--bg:#0a0e1a;
      --card:rgba(255,255,255,.045);--line:rgba(255,255,255,.09);}
.stApp,[data-testid="stAppViewContainer"]{background:
  radial-gradient(900px 500px at 8% -5%,rgba(139,92,246,.22),transparent 60%),
  radial-gradient(900px 500px at 100% 0%,rgba(16,185,129,.18),transparent 55%),
  radial-gradient(700px 500px at 50% 110%,rgba(59,130,246,.14),transparent 60%),#0a0e1a !important;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stMainBlockContainer"]{padding-top:1.2rem;max-width:1500px;}

.nav{display:flex;align-items:center;gap:16px;padding:.6rem .2rem 1.4rem;
  border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:1.6rem;}
.nav .logo{width:48px;height:48px;flex:none;border-radius:14px;
  background-image:url('data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 46 46%22%3E%3Cdefs%3E%3ClinearGradient id=%22g%22 x1=%220%22 y1=%220%22 x2=%2246%22 y2=%2246%22%3E%3Cstop offset=%220%22 stop-color=%22%2310b981%22/%3E%3Cstop offset=%221%22 stop-color=%22%238b5cf6%22/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width=%2246%22 height=%2246%22 rx=%2214%22 fill=%22url(%23g)%22/%3E%3Cpath d=%22M13 28 L23 16 L33 28 Z%22 fill=%22%23fff%22/%3E%3Crect x=%2217%22 y=%2228%22 width=%2212%22 height=%227%22 fill=%22%23fff%22/%3E%3Crect x=%2221%22 y=%2230%22 width=%224%22 height=%225%22 fill=%22%2310b981%22/%3E%3Ccircle cx=%2234%22 cy=%2212%22 r=%225%22 fill=%22none%22 stroke=%22%23fff%22 stroke-width=%222.6%22 opacity=%220.92%22/%3E%3Cline x1=%2237.6%22 y1=%2215.6%22 x2=%2241%22 y2=%2219%22 stroke=%22%23fff%22 stroke-width=%222.6%22 stroke-linecap=%22round%22 opacity=%220.92%22/%3E%3C/svg%3E');
  background-size:cover;background-position:center;background-repeat:no-repeat;
  box-shadow:0 10px 28px -6px rgba(16,185,129,.55);}
.nav-br{display:flex;flex-direction:column;justify-content:center;line-height:1.15;gap:3px;}
.nav-br h1{font-size:1.4rem;margin:0;font-weight:800;letter-spacing:-.02em;
  background:linear-gradient(90deg,#fff,#c7cbe0);-webkit-background-clip:text;background-clip:text;color:transparent;}
.nav-br p{margin:0;font-size:.76rem;color:#8fa0bf;letter-spacing:.02em;}
.nav .badge{margin-left:auto;font-size:.74rem;font-weight:600;color:var(--accent);
  border:1px solid rgba(16,185,129,.35);background:rgba(16,185,129,.08);
  padding:.45rem .9rem;border-radius:999px;white-space:nowrap;display:flex;align-items:center;gap:6px;}

.hero{text-align:center;padding:.2rem 0 1.6rem;}
.hero h2{font-size:2.15rem;margin:.2rem 0 .4rem;font-weight:800;letter-spacing:-.02em;
  background:linear-gradient(90deg,#fff,#9fb3d9);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero .sub{color:#93a5c7;font-size:.96rem;max-width:720px;line-height:1.55;margin:0 auto;}

[data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.015));
  border:1px solid var(--line)!important;border-radius:18px!important;backdrop-filter:blur(6px);
  transition:transform .15s ease,border-color .2s ease,box-shadow .2s ease;}
[data-testid="stVerticalBlockBorderWrapper"]:hover{transform:translateY(-2px);
  border-color:rgba(16,185,129,.5)!important;box-shadow:0 18px 40px -18px rgba(16,185,129,.35);}

.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:0 0 1.4rem;}
.stats .s{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem;
  display:flex;align-items:center;gap:.85rem;backdrop-filter:blur(6px);}
.stats .s .ic{width:44px;height:44px;border-radius:12px;display:grid;place-items:center;font-size:1.3rem;flex-shrink:0;}
.stats .s .v{font-size:1.4rem;font-weight:800;letter-spacing:-.02em;}
.stats .s .l{font-size:.72rem;color:#8fa0bf;}
@media(max-width:900px){.stats{grid-template-columns:repeat(2,1fr);}}

.rcard .rc-top{display:flex;align-items:center;gap:12px;}
.rcard .rnum{width:30px;height:30px;flex-shrink:0;border-radius:9px;display:grid;place-items:center;
  font-weight:800;font-size:.9rem;color:#0b1220;background:linear-gradient(135deg,var(--accent),#34d399);}
.rcard .rtitle{flex:1;min-width:0;}
.rcard .rname{font-size:.98rem;font-weight:700;line-height:1.35;}
.rcard .rmeta{font-size:.74rem;color:#8fa0bf;margin-top:2px;}
.rcard .rbadge{margin-left:auto;font-size:.72rem;font-weight:700;padding:.3rem .6rem;border-radius:999px;white-space:nowrap;}
.rbadge.on{color:#0f2e22;background:linear-gradient(135deg,#34d399,#10b981);box-shadow:0 4px 14px -4px rgba(16,185,129,.6);}
.rbadge.off{color:#3b0a17;background:linear-gradient(135deg,#fb7185,#f43f5e);box-shadow:0 4px 14px -4px rgba(244,63,94,.5);}
.rcard .rchips{display:flex;gap:6px;flex-wrap:wrap;margin:.7rem 0 .55rem;}
.rchip{font-size:.7rem;font-weight:600;padding:.2rem .6rem;border-radius:999px;background:rgba(139,92,246,.14);color:#c4b5fd;border:1px solid rgba(139,92,246,.3);}
.rmatch{display:flex;align-items:center;gap:.6rem;margin:.35rem 0;}
.rmatch .mlab{font-size:.72rem;color:#8fa0bf;min-width:44px;}
.rmatch .mbar{flex:1;height:7px;border-radius:99px;background:rgba(255,255,255,.08);overflow:hidden;}
.rmatch .mbar i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,var(--accent),var(--accent3));}
.rmatch .mval{font-weight:700;font-size:.82rem;min-width:38px;text-align:right;}
.rwhy{font-size:.78rem;color:#aeb8d0;line-height:1.5;padding-top:.3rem;border-top:1px solid var(--line);}
</style>
"""
st.html(CSS)


def esc(s: str) -> str:
    """HTML-escape a string for safe insertion into our custom markup."""
    return _html.escape(str(s or ""), quote=True)


# =====================================================================
# 2. LOCALISATION  (English / मराठी UI texts)
# =====================================================================
LANG = {
    "en": {
        "profile": "Eligibility profile", "lang": "Language",
        "reset": "Reset profile", "age": "Age", "gender": "Gender",
        "state": "State", "income": "Annual income (₹)", "caste": "Caste",
        "occupation": "Occupation", "residence": "Residence",
        "bpl": "BPL card holder", "disability": "Specially-abled",
        "indexed": "schemes indexed", "reset_btn": "Reset profile",
        "search_ph": "Type here — scholarship, women loan, pension…",
        "quick": "Quick searches:", "cat": "Category filter",
        "top": "Show top results",
        "stat_total": "Total schemes indexed", "stat_eligible": "Eligible in results",
        "stat_top": "Top match score", "stat_avg": "Avg result match",
        "tab_rec": "Recommendations", "tab_exp": "Explore dataset",
        "tagline": "Search & rank Indian government schemes by relevance and your eligibility profile.",
        "empty": "No results — adjust filters / category.",
        "det": "View details", "match": "Match",
        "on": "Eligible · पात्र", "off": "Not eligible · अपात्र",
    },
    "mr": {
        "profile": "पात्रता प्रोफाइल", "lang": "भाषा",
        "reset": "रीसेट प्रोफाइल", "age": "वय", "gender": "लिंग",
        "state": "राज्य", "income": "वार्षिक उत्पन्न (₹)", "caste": "जात",
        "occupation": "व्यवसाय", "residence": "निवास",
        "bpl": "BPL कार्डधारक", "disability": "दिव्यांग",
        "indexed": "योजना", "reset_btn": "↺ रीसेट",
        "search_ph": "टाइप करा — शिष्यवृत्ती, महिला कर्ज, पेन्शन…",
        "quick": "झटपट शोध:", "cat": "वर्ग फिल्टर",
        "top": "निकाल दाखवा",
        "stat_total": "एकूण योजना", "stat_eligible": "पात्र निकाल",
        "stat_top": "सर्वोत्तम जुळणी", "stat_avg": "सरासरी जुळणी",
        "tab_rec": "शिफारस", "tab_exp": "डेटासेट एक्सप्लोर",
        "tagline": "शासकीय योजना शोधा — प्रासंगिकता व पात्रता प्रोफाइलनुसार सर्वात संबंधित योजना सर्वात वर.",
        "empty": "निकाल नाही — फिल्टर / वर्ग बदला.",
        "det": "तपशील पाहा", "match": "जुळणी",
        "on": "पात्र", "off": "अपात्र",
    },
}


# =====================================================================
# 3. ENGINE (cached: schemes + tf-idf index, built once)
# =====================================================================
@st.cache_resource(ttl="1h")
def load_and_build(path=DATA_PATH):
    schemes = load_schemes(path)
    vectorizer, matrix = build_index(schemes)
    return schemes, vectorizer, matrix


schemes, vectorizer, matrix = load_and_build()

# Category visuals (icon + badge colour) for chips
CAT_ICON = {
    "शिक्षण": "🎓", "महिला व बालक": "🧕", "कृषी": "🌾", "ग्रामीण व पर्यावरण": "🌱",
    "सामाजिक कल्याण": "🤝", "आरोग्य": "🏥", "कौशल्य व रोजगार": "💼", "व्यवसाय": "🏢",
    "बँकिंग व विमा": "🏦", "वित्तीय सेवा व विमा": "💳", "खेळ व संस्कृती": "🏆",
    "गृहनिर्माण": "🏠", "प्रवास व पर्यटन": "✈️", "विज्ञान": "🔬", "आयटी व कम्युनिकेशन्स": "💻",
    "वाहतूक व पायाभूत सुविधा": "🚆", "उपयुक्तता व स्वच्छता": "🚰",
    "सार्वजनिक सुरक्षा": "👮", "कायदा व न्याय": "⚖️",
}
CAT_COLOR = {
    "शिक्षण": "blue", "महिला व बालक": "violet", "कृषी": "green", "ग्रामीण व पर्यावरण": "green",
    "सामाजिक कल्याण": "orange", "आरोग्य": "red", "कौशल्य व रोजगार": "blue", "व्यवसाय": "orange",
    "बँकिंग व विमा": "yellow", "वित्तीय सेवा व विमा": "yellow", "खेळ व संस्कृती": "blue",
    "गृहनिर्माण": "orange", "प्रवास व पर्यटन": "green", "विज्ञान": "violet",
    "आयटी व कम्युनिकेशन्स": "violet", "वाहतूक व पायाभूत सुविधा": "blue",
    "उपयुक्तता व स्वच्छता": "green", "सार्वजनिक सुरक्षा": "red", "कायदा व न्याय": "blue",
}


def cat_label(c: str) -> str:
    return f"{CAT_ICON.get(c, '🏷️')} {c}"


# =====================================================================
# 4. SIDEBAR  (language toggle + eligibility profile)
# =====================================================================
lang_sel = st.sidebar.segmented_control(
    LANG["en"]["lang"], ["English", "मराठी"], default="English")
t = LANG["mr"] if lang_sel == "मराठी" else LANG["en"]

st.sidebar.title("👤 " + t["profile"])
age = st.sidebar.number_input(t["age"], 0, 120, 25)
gender = st.sidebar.segmented_control(t["gender"], ["सर्व", "महिला", "पुरुष"], default="सर्व")
state_options = ["सर्व राज्ये", "All states"] + sorted({s["state"] for s in schemes if s.get("state")})
state_label = st.sidebar.selectbox(t["state"], state_options)
income = st.sidebar.number_input(t["income"], 0, 50_000_000, 150_000, step=10_000, format="%d")
caste = st.sidebar.selectbox(t["caste"], ["सर्व", "SC", "ST", "OBC", "EWS"])
occupation = st.sidebar.segmented_control(
    t["occupation"], ["सर्व", "विद्यार्थी", "शेतकरी", "वैयक्तिक", "उद्योग", "ज्येष्ठ नागरिक"],
    default="सर्व")
residence = st.sidebar.segmented_control(t["residence"], ["दोन्ही", "ग्रामीण", "शहरी"], default="दोन्ही")
bpl = st.sidebar.toggle(t["bpl"], value=False)
disability = st.sidebar.toggle(t["disability"], value=False)
if st.sidebar.button(t["reset_btn"], use_container_width=True):
    for k in ["q", "ex", "cat_pick", "topk"]:
        st.session_state.pop(k, None)
    st.rerun()

# =====================================================================
# 5. NAVBAR + HERO + SEARCH
# =====================================================================
st.html(f"""
<div class="nav">
  <div class="logo"></div>
  <div class="nav-br">
    <h1>YojanaSetu</h1>
    <p>शासकीय योजना शिफारस व शोध प्रणाली</p>
  </div>
  <div class="badge">🗂️ {len(schemes)} {t['indexed']} · 37 राज्ये</div>
</div>

<div class="hero">
  <h2>{t['tab_rec']}</h2>
  <p class="sub">{t['tagline']}</p>
</div>
""")


def _apply_example():
    if st.session_state.get("ex"):
        st.session_state["q"] = st.session_state["ex"]


st.text_input("search", key="q", label_visibility="collapsed", type="search", placeholder="🔎 " + t["search_ph"])
st.caption(t["quick"])
st.pills("examples", ["शिष्यवृत्ती", "महिला कर्ज", "शेतकरी कर्जमाफी", "पेन्शन ज्येष्ठ",
                      "गृहनिर्माण", "बेरोजगार भत्ता", "अनुदान शेतकरी"],
         key="ex", on_change=_apply_example, label_visibility="collapsed")

query = (st.session_state.get("q") or "").strip()

# =====================================================================
# 6. PROFILE + CATEGORY FILTER
# =====================================================================
profile = {
    "age": age,
    "gender": {"महिला": "female", "पुरुष": "male"}.get(gender, "all"),
    "state": "all" if state_label in ("सर्व राज्ये", "All states") else state_label,
    "income": income,
    "occupation": "all" if occupation == "सर्व" else occupation,
    "residence": {"ग्रामीण": "rural", "शहरी": "urban"}.get(residence, "both"),
    "bpl": bpl, "disability": disability,
    "caste": "all" if caste == "सर्व" else caste,
}

all_categories = sorted({c for s in schemes for c in s.get("categories", [])})
st.markdown("##### 🗂️ " + t["cat"])
cat_pick = st.pills("cats", [cat_label(c) for c in all_categories],
                    key="cat_pick", selection_mode="multi", label_visibility="collapsed")
selected_cats = {c for c in all_categories if cat_label(c) in (cat_pick or [])}
topk = st.slider(t["top"], 5, 40, 12, 3, key="topk")

# =====================================================================
# 7. RANK + FILTER
# =====================================================================
results = rank(schemes, vectorizer, matrix, query, profile, top_k=topk)
if selected_cats:
    results = [r for r in results if set(r["scheme"].get("categories", [])) & selected_cats]
    results.sort(key=lambda r: r["score"], reverse=True)

n_eligible = sum(r["eligible"] for r in results)
best = max((r["score"] for r in results), default=0.0)
avg = (sum(r["score"] for r in results) / len(results)) * 100 if results else 0


def stat(icon, tint, value, label):
    return (f'<div class="s"><div class="ic" style="background:{tint};color:#0b1220">{icon}</div>'
            f'<div><div class="v">{value}</div><div class="l">{label}</div></div></div>')


st.html('<div class="stats">'
        + stat("🗂️", "linear-gradient(135deg,#34d399,#10b981)", len(schemes), t["stat_total"])
        + stat("✅", "linear-gradient(135deg,#a78bfa,#8b5cf6)", n_eligible, t["stat_eligible"])
        + stat("🏆", "linear-gradient(135deg,#60a5fa,#3b82f6)", f"{best*100:.0f}%", t["stat_top"])
        + stat("📄", "linear-gradient(135deg,#fbbf24,#f59e0b)", f"{avg:.0f}%", t["stat_avg"])
        + '</div>')

# =====================================================================
# 8. RESULTS + EXPLORE
# =====================================================================
tab_rec, tab_exp = st.tabs([":material/search:  " + t["tab_rec"],
                            ":material/bar_chart:  " + t["tab_exp"]])

with tab_rec:
    if not results:
        st.info(t["empty"])
    for i, r in enumerate(results, start=1):
        s = r["scheme"]
        on = r["eligible"]
        chips = " ".join(f'<span class="rchip">{esc(cat_label(c))}</span>'
                         for c in s.get("categories", []))
        pct = min(max(r["score"], 0) * 100, 100)
        reason = esc(" ; ".join(r["reasons"]) if r["reasons"]
                     else ("कोणतेही निर्बंध नाही" if lang_sel == "मराठी" else "no restrictions"))
        badge = t["on"] if on else t["off"]
        badge_cls = "rbadge on" if on else "rbadge off"

        st.html(f"""
        <div class="rcard">
          <div class="rc-top">
            <div class="rnum">{i}</div>
            <div class="rtitle">
              <div class="rname">{esc(s.get('scheme_name',''))}</div>
              <div class="rmeta">{esc(s.get('state',''))} · {esc(s.get('department_or_ministry',''))}</div>
            </div>
            <span class="{badge_cls}">{badge}</span>
          </div>
          <div class="rchips">{chips}</div>
          <div class="rmatch">
            <span class="mlab">{t['match']}</span>
            <div class="mbar"><i style="width:{pct:.0f}%"></i></div>
            <span class="mval">{pct:.0f}%</span>
          </div>
          <div class="rwhy">ℹ️ {reason}</div>
        </div>
        """)
        with st.expander("📖 " + t["det"]):
            st.markdown(f"**Description:** {s.get('description','')}")
            st.markdown(f"**लाभ / Benefits:** {s.get('benefits','')}")
            st.markdown(f"**पात्रता निकष / Criteria:** {s.get('eligibility_criteria','')}")
            st.markdown(f"**अर्ज प्रक्रिया / Application:** {s.get('application_process','')}")
            st.markdown(f"**कागदपत्रे / Documents:** {s.get('documents_required','')}")

with tab_exp:
    _cs = Counter(c for s in schemes for c in s.get("categories", []))
    cat_df = pd.DataFrame([{"Category": cat_label(c), "Schemes": n} for c, n in _cs.most_common()])
    sts = Counter(s.get("state", "—") for s in schemes)
    state_df = pd.DataFrame([{"State": k, "Schemes": v} for k, v in sts.most_common(15)])

    b, c = st.columns(2)
    with b:
        st.subheader("Schemes by category", icon=":material/donut_small:")
        st.bar_chart(cat_df.set_index("Category"))
    with c:
        st.subheader("Top states", icon=":material/public:")
        st.bar_chart(state_df.set_index("State"))

    with st.container(border=True):
        st.subheader("All schemes", icon=":material/table_chart:")
        st.dataframe(pd.DataFrame([{
            "ID": s["scheme_id"], "योजना": s["scheme_name"],
            "राज्य": s.get("state", ""), "वर्ग": ", ".join(s.get("categories", [])),
        } for s in schemes]), hide_index=True, height=320, width="stretch")