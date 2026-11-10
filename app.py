"""Colorectal Cancer Classification - Model Tester & Evaluation Dashboard
Run:  streamlit run app.py
"""
import base64
import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from model_loader import load_model, _torch_load   # noqa: E402
from preprocess import get_transform               # noqa: E402

MODELS_DIR = ROOT / "models"
COLORS = {"Benign": "#22c55e", "Adenocarcinoma": "#ef4444"}
PURPLE = "#7c6cf0"

st.set_page_config(page_title="Colorectal Cancer Classifier", page_icon="🧬", layout="wide")

# ------------------------------------------------------------------ CSS
st.markdown("""
<style>
.stApp{background:radial-gradient(1200px 700px at 70% -10%,#13204a 0%,#0a1020 55%,#080d1a 100%);color:#e6ebff}
#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:1500px}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1226 0%,#080d1c 100%);border-right:1px solid #1b2748}
section[data-testid="stSidebar"] .block-container{padding-top:1.2rem}
.brand{text-align:center;margin-bottom:.8rem}
.brand h3{margin:.5rem 0 .1rem;font-size:1.35rem;line-height:1.2;color:#fff}
.brand p{margin:0;color:#8ea0d0;font-size:.8rem}
section[data-testid="stSidebar"] .stButton button{width:100%;justify-content:flex-start;border-radius:10px;padding:.6rem .9rem;font-weight:500;border:none;box-shadow:none}
section[data-testid="stSidebar"] .stButton button[kind="secondary"],
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-secondary"]{background:transparent;color:#c9d4f5}
section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover,
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-secondary"]:hover{background:#16224a;color:#fff}
section[data-testid="stSidebar"] .stButton button[kind="primary"],
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"]{background:linear-gradient(90deg,#4338ca,#6366f1);color:#fff}
.side-label{color:#e6ebff;font-weight:600;margin:.9rem 0 .25rem;font-size:.95rem}
.side-val{color:#c9d4f5;font-size:.88rem;margin-bottom:.2rem}
.side-foot{color:#6f7fae;font-size:.72rem;text-align:center;margin-top:1.6rem}
hr.sep{border:none;border-top:1px solid #1b2748;margin:.9rem 0}
.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem}
.topbar .left{display:flex;align-items:center;gap:1rem}
.topbar h1{margin:0;font-size:2.1rem;line-height:1.1;color:#fff;font-weight:700;padding:0}
.topbar .subt{color:#9fb0dc;font-size:1rem;margin-top:.15rem}
.badge{display:flex;align-items:center;gap:.55rem;background:#0f1a38;border:1px solid #22325f;border-radius:999px;padding:.5rem 1rem;color:#e6ebff;font-size:.9rem}
.dot{width:12px;height:12px;border-radius:50%}
.info{background:linear-gradient(90deg,#0e1a3d,#0d1633);border:1px solid #2b4a9a;border-radius:12px;padding:.9rem 1.2rem;display:flex;gap:1rem;align-items:center;color:#dfe7ff;font-size:.9rem;line-height:1.5;margin-bottom:1rem}
.info .i{flex:0 0 34px;height:34px;border-radius:50%;background:#3b82f6;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700}
.card{background:linear-gradient(180deg,rgba(20,32,70,.75),rgba(14,22,50,.75));border:1px solid #22325f;border-radius:16px;padding:1.1rem 1.3rem;margin-bottom:1rem}
.card-title{display:flex;align-items:center;gap:.7rem;font-size:1.25rem;font-weight:600;color:#fff;margin-bottom:.8rem}
.sub-title{display:flex;align-items:center;gap:.5rem;font-weight:600;color:#fff;margin:1rem 0 .6rem}
.check{width:34px;height:34px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;color:#fff;font-weight:700}
.banner{display:flex;align-items:center;justify-content:center;gap:2rem;border:1px solid;border-radius:12px;padding:.9rem 1rem}
.banner .name{font-size:2.2rem;font-weight:700;color:#fff}
.banner .vr{width:1px;height:38px;background:#ffffff40}
.banner .conf{font-size:1.2rem;color:#e6ebff}
.banner .conf b{font-size:2rem;color:#fff}
.barrow{display:flex;align-items:center;gap:1rem;margin:.55rem 0}
.barrow .lbl{flex:0 0 130px;color:#e6ebff}
.barrow .track{flex:1;height:12px;background:#1a2547;border-radius:999px;overflow:hidden}
.barrow .fill{height:100%;border-radius:999px}
.barrow .pct{flex:0 0 70px;text-align:right;color:#e6ebff}
.imgcard{position:relative;border-radius:16px;overflow:hidden;border:1px solid #2a3a73;background:#0d1530}
.imgcard img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block}
.tag{position:absolute;background:rgba(10,16,35,.82);color:#fff;border-radius:10px;padding:.4rem .8rem;font-size:.85rem}
.tag.tl{left:14px;bottom:14px}.tag.br{right:14px;bottom:14px}
.donutwrap{display:flex;align-items:center;gap:1.4rem}
.donut{width:150px;height:150px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex:0 0 150px}
.donut .hole{width:104px;height:104px;border-radius:50%;background:#0e1632;display:flex;align-items:center;justify-content:center;font-size:1.5rem;font-weight:700;color:#fff}
.legend div{display:flex;align-items:center;gap:.6rem;margin:.45rem 0;color:#e6ebff;font-size:.92rem}
.legend span.sw{width:12px;height:12px;border-radius:50%;display:inline-block}
.legend span.v{margin-left:auto;padding-left:1.4rem}
.summary{display:flex;gap:.8rem;color:#e6ebff;line-height:1.5}
.summary .ok{flex:0 0 28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700}
.warn{margin-top:.7rem;color:#fbbf24;font-size:.85rem}
.note{margin-top:.7rem;color:#8ea0d0;font-size:.8rem}
.kv{display:grid;grid-template-columns:220px 1fr;row-gap:.65rem;color:#e6ebff}
.kv .k{color:#8ea0d0}
.mgrid{display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;margin-bottom:1rem}
.metric{background:linear-gradient(180deg,rgba(20,32,70,.75),rgba(14,22,50,.75));border:1px solid #22325f;border-radius:14px;padding:1rem;text-align:center}
.metric .mv{font-size:1.8rem;font-weight:700;color:#fff}
.metric .ml{color:#8ea0d0;font-size:.85rem;margin-top:.2rem}
div[data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(180deg,rgba(20,32,70,.75),rgba(14,22,50,.75));border:1px solid #22325f!important;border-radius:16px}
div[data-testid="stFileUploaderDropzone"]{background:#0c1430;border:1.5px dashed #3a4d8f;border-radius:12px}
div[data-testid="stFileUploaderDropzone"] button{background:#16224a;color:#fff;border:1px solid #2c3f7a;border-radius:10px}
div[data-testid="stSelectbox"] > div > div{background:#0c1430;border-color:#22325f;border-radius:10px}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def H(*parts):
    """Join HTML pieces on one line (Markdown treats indented/blank lines as code)."""
    return "".join(parts)


@st.cache_resource(show_spinner="Loading model...")
def get_model(path: str):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, names, size = load_model(path, dev)
    return model, names, size, dev


@st.cache_data(show_spinner=False)
def get_meta(path: str):
    ckpt = _torch_load(path, "cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return {}


LOGO = ('<svg width="72" height="72" viewBox="0 0 64 64" fill="none"><path d="M14 10h36a6 6 0 0 1 6 6v6M50 28H20a6 6 0 0 0-6 6v2a6 6 0 0 0 6 6h30a6 6 0 0 1 6 6v2a6 6 0 0 1-6 6H22" '
        'stroke="#4f6bff" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/></svg>')
SHIELD = ('<svg width="54" height="54" viewBox="0 0 64 64" fill="none"><path d="M32 6 54 14v16c0 14-10 24-22 28C20 54 10 44 10 30V14z" fill="#1d3f9a55" stroke="#4f8cff" stroke-width="3.5" stroke-linejoin="round"/>'
          '<path d="M32 22v18M23 31h18" stroke="#fff" stroke-width="4.5" stroke-linecap="round"/></svg>')

# ------------------------------------------------------------------ sidebar
PAGES = [("Home", "🏠"), ("Model Info", "🧠"), ("Evaluation Metrics", "📊"), ("About", "ℹ️")]
if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    st.markdown(H('<div class="brand">', LOGO, '<h3>Colorectal Cancer<br>Classifier</h3><p>AI-Powered Pathology Analysis</p></div>'),
                unsafe_allow_html=True)
    for label, icon in PAGES:
        active = st.session_state.page == label
        if st.button(f"{icon}  {label}", key=f"nav_{label}", type="primary" if active else "secondary",
                     use_container_width=True):
            st.session_state.page = label
            st.rerun()
    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    model_files = sorted(MODELS_DIR.glob("*.pt"))
    if not model_files:
        st.error("No .pt files in models/")
        st.stop()
    st.markdown('<div class="side-label">📦 Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Model", [f.name for f in model_files], label_visibility="collapsed")

model_path = str(MODELS_DIR / model_name)
load_error = None
try:
    model, names, img_size, device = get_model(model_path)
except Exception as e:  # noqa: BLE001
    load_error = str(e)
    names, device, img_size = [], "cpu", 224

with st.sidebar:
    dev_txt = "GPU (" + torch.cuda.get_device_name(0) + ")" if device == "cuda" else "CPU"
    st.markdown(H('<div class="side-label">🖥️ Device</div><div class="side-val">', dev_txt, '</div>',
                  '<div class="side-label">🗂️ Classes</div><div class="side-val">', html.escape(str(names)), '</div>',
                  '<div class="side-foot">© 2026 Colorectal Cancer Classifier<br>Research Prototype</div>'),
                unsafe_allow_html=True)

# ------------------------------------------------------------------ top bar
ok_color = "#22c55e" if not load_error else "#ef4444"
ok_text = "Model Ready" if not load_error else "Model Error"
st.markdown(H('<div class="topbar"><div class="left">', SHIELD,
              '<div><h1>Colorectal Cancer Classification</h1><div class="subt">Model Tester &amp; Evaluation Dashboard</div></div></div>',
              '<div class="badge"><span class="dot" style="background:', ok_color, '"></span>', ok_text, '</div></div>'),
            unsafe_allow_html=True)

if load_error:
    st.error("Could not load the model.")
    st.code(load_error)
    st.stop()

page = st.session_state.page

# ------------------------------------------------------------------ HOME
if page == "Home":
    st.markdown(H('<div class="info"><div class="i">i</div><div>Binary classifier: Adenocarcinoma vs Benign colon tissue (LC25000-style H&amp;E patches, EfficientNet-B0).<br>',
                  'Research prototype only – not a clinical diagnosis. It only knows these two classes, so images of other tissue types (or CT/X-ray) will still be forced into one of them.</div></div>'),
                unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-title" style="margin-bottom:.4rem">☁️ Upload Histopathology Image</div>', unsafe_allow_html=True)
        up = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if up is None:
        st.markdown('<div class="note" style="text-align:center;margin-top:1.5rem">Upload an H&amp;E histopathology patch to see the prediction.</div>',
                    unsafe_allow_html=True)
    else:
        raw = up.getvalue()
        img = Image.open(up).convert("RGB")
        with torch.no_grad():
            x = get_transform(img_size)(img).unsqueeze(0).to(device)
            probs = torch.softmax(model(x), 1)[0].cpu().tolist()
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        top = order[0]
        name, p = names[top], probs[top]
        col = COLORS.get(name, "#6366f1")
        b64 = base64.b64encode(raw).decode()
        mime = up.type or "image/jpeg"

        left, right = st.columns([1, 1.28], gap="large")
        with left:
            st.markdown(H('<div class="imgcard"><img src="data:', mime, ';base64,', b64, '"/>',
                          '<div class="tag tl">🖼️ ', html.escape(up.name), '</div>',
                          '<div class="tag br">', f"{len(raw) / 1024:.1f} KB", '</div></div>'),
                        unsafe_allow_html=True)
        with right:
            rows = ""
            for i in order:
                c = col if i == top else PURPLE
                rows += H('<div class="barrow"><div class="lbl">', html.escape(names[i]), '</div><div class="track"><div class="fill" style="width:',
                          f"{probs[i] * 100:.2f}", '%;background:', c, '"></div></div><div class="pct">', f"{probs[i] * 100:.2f}%", '</div></div>')
            st.markdown(H('<div class="card"><div class="card-title"><span class="check" style="background:', col, '">✓</span>Prediction Result</div>',
                          '<div class="banner" style="border-color:', col, ';background:linear-gradient(90deg,', col, '40,', col, '1f)">',
                          '<span class="name">', html.escape(name), '</span><span class="vr"></span>',
                          '<span class="conf">Confidence: <b>', f"{p * 100:.1f}%", '</b></span></div>',
                          '<div class="sub-title">📶 Class Confidence</div>', rows, '</div>'),
                        unsafe_allow_html=True)

            c1, c2 = st.columns(2, gap="medium")
            with c1:
                stops, start = [], 0.0
                for i in order:
                    end = start + probs[i] * 100
                    stops.append(f"{col if i == top else PURPLE} {start:.2f}% {end:.2f}%")
                    start = end
                legend = ""
                for i in order:
                    legend += H('<div><span class="sw" style="background:', col if i == top else PURPLE, '"></span>', html.escape(names[i]),
                                '<span class="v">', f"{probs[i] * 100:.1f}%", '</span></div>')
                st.markdown(H('<div class="card"><div class="card-title" style="font-size:1.05rem">🥧 Confidence Distribution</div><div class="donutwrap">',
                              '<div class="donut" style="background:conic-gradient(', ",".join(stops), ')"><div class="hole">', f"{p * 100:.1f}%", '</div></div>',
                              '<div class="legend">', legend, '</div></div></div>'), unsafe_allow_html=True)
            with c2:
                extra = '<div class="warn">⚠️ Confidence is relatively low – treat this result with extra caution.</div>' if p < 0.85 else ""
                st.markdown(H('<div class="card"><div class="card-title" style="font-size:1.05rem">💡 Analysis Summary</div>',
                              '<div class="summary"><span class="ok" style="background:', col, '">✓</span><div>The image is classified as <b>', html.escape(name),
                              '</b> with <b>', f"{p * 100:.1f}%", '</b> confidence.</div></div>', extra,
                              '<div class="note">Valid only for H&amp;E colon histology patches. Not a diagnosis.</div></div>'),
                            unsafe_allow_html=True)

# ------------------------------------------------------------------ MODEL INFO
elif page == "Model Info":
    meta = get_meta(model_path)
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = Path(model_path).stat().st_size / 1024 ** 2
    kv = [("Checkpoint file", model_name), ("File size", f"{size_mb:.1f} MB"),
          ("Architecture", "EfficientNet-B0 (torchvision)"), ("Parameters", f"{n_params:,}"),
          ("Classifier head", "Dropout(0.2) → Linear(1280, 2)"), ("Input size", f"{img_size} × {img_size} RGB"),
          ("Normalization", "ImageNet mean/std"), ("Classes", "0 = " + names[0] + ",  1 = " + names[1] if len(names) == 2 else str(names)),
          ("Checkpoint format", "Full dict (weights + metadata)" if meta else "Raw state_dict (weights only)"),
          ("Stored test accuracy", f"{meta['test_accuracy'] * 100:.2f}%" if "test_accuracy" in meta else "not stored in this file"),
          ("Training data", "LC25000 colon: 5,000 adenocarcinoma + 5,000 benign (70/15/15 split)"),
          ("Training recipe", "Frozen backbone 25 epochs (lr 1e-4) → fine-tune last 2 blocks 10 epochs (lr 1e-5), AdamW"),
          ("Running on", "GPU" if device == "cuda" else "CPU")]
    body = "".join(f'<div class="k">{html.escape(k)}</div><div>{html.escape(v)}</div>' for k, v in kv)
    st.markdown(H('<div class="card"><div class="card-title">🧠 Model Information</div><div class="kv">', body, '</div></div>'),
                unsafe_allow_html=True)

# ------------------------------------------------------------------ METRICS
elif page == "Evaluation Metrics":
    csv = MODELS_DIR / "evaluation_metrics.csv"
    if not csv.exists():
        st.info("Put evaluation_metrics.csv in the models/ folder to see it here.")
    else:
        df = pd.read_csv(csv)
        r = df.iloc[0]
        cards = [("Accuracy", "Accuracy"), ("Precision", "Precision"), ("Recall", "Recall"), ("F1_Score", "F1 Score")]
        html_cards = ""
        for key, lab in cards:
            if key in r:
                html_cards += H('<div class="metric"><div class="mv">', f"{float(r[key]) * 100:.2f}%", '</div><div class="ml">', lab, '</div></div>')
        if "ROC_AUC" in r:
            html_cards += H('<div class="metric"><div class="mv">', f"{float(r['ROC_AUC']):.4f}", '</div><div class="ml">ROC-AUC</div></div>')
        st.markdown(H('<div class="mgrid">', html_cards, '</div>'), unsafe_allow_html=True)

        splits = [(c.replace("_", " "), int(r[c])) for c in ("Training_Images", "Validation_Images", "Testing_Images") if c in r]
        if splits:
            st.markdown(H('<div class="card"><div class="card-title">🗂️ Dataset Split</div><div class="kv">',
                          "".join(f'<div class="k">{k}</div><div>{v:,}</div>' for k, v in splits), '</div></div>'), unsafe_allow_html=True)
        st.markdown('<div class="card-title">📄 Raw metrics file</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        cm = ROOT / "results" / "confusion_matrix.png"
        if cm.exists():
            st.markdown('<div class="card-title">🧮 Confusion matrix (from src/evaluate.py)</div>', unsafe_allow_html=True)
            st.image(str(cm), width=420)
        st.markdown('<div class="note">Caution: LC25000 consists of augmented copies of ~1,250 original images, so random train/test splits can leak near-duplicates and inflate these scores. Validate on an independent dataset.</div>',
                    unsafe_allow_html=True)

# ------------------------------------------------------------------ ABOUT
else:
    st.markdown(H('<div class="card"><div class="card-title">ℹ️ About</div>',
                  '<p>A research prototype that classifies H&amp;E-stained colon histopathology patches as <b>Adenocarcinoma</b> or <b>Benign</b> using a fine-tuned EfficientNet-B0.</p>',
                  '<p><b>Intended use:</b> testing and demonstration of the trained model.</p>',
                  '<p><b>Limitations:</b></p><ul>',
                  '<li>Only two classes – anything else (other tissue types, CT, X-ray, photos) is still forced into one of them.</li>',
                  '<li>Trained on a single dataset (LC25000); performance on other scanners, stains or hospitals is untested.</li>',
                  '<li>Patch-level prediction only; it does not analyse whole slides.</li>',
                  '<li>Not a medical device. Not for diagnosis or treatment decisions.</li></ul></div>'), unsafe_allow_html=True)
