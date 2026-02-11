import streamlit as st
import pickle
import numpy as np
import time
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pytesseract
import io

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Suicidal Tweet Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Kill ALL Streamlit top chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stAlert { display: none !important; }
[data-testid="stHeader"]          { display: none !important; }
[data-testid="stToolbar"]         { display: none !important; }
[data-testid="stDecoration"]      { display: none !important; }
[data-testid="stStatusWidget"]    { display: none !important; }
.reportview-container .main .block-container { padding-top: 0 !important; }
div[data-testid="stAppViewBlockContainer"] > div:first-child { padding-top: 0 !important; }

/* ── Viewport lock ── */
html, body {
    height: 100vh;
    overflow: hidden;
    margin: 0; padding: 0;
}
[data-testid="stAppViewContainer"] {
    height: 100vh;
    overflow: hidden;
    padding-top: 0 !important;
}

/* ── Animated gradient background ── */
.stApp {
    background: linear-gradient(-45deg, #4a1c8c, #764ba2, #a855f7, #6366f1);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
    font-family: 'Inter', sans-serif;
    height: 100vh;
    overflow: hidden;
    padding-top: 0 !important;
}
@keyframes gradientShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}

/* ── Block container: zero top padding ── */
.main .block-container {
    max-width: 100% !important;
    padding: 0.4rem 0.8rem 0.4rem 0.8rem !important;
    margin: 0 !important;
    height: 100vh;
    overflow: hidden;
}

/* ── Columns: fixed height, scroll inside ── */
[data-testid="column"] {
    padding: 0 0.35rem !important;
    height: calc(100vh - 1rem);
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.25) transparent;
}
[data-testid="column"]::-webkit-scrollbar { width: 3px; }
[data-testid="column"]::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.25);
    border-radius: 4px;
}

/* ── Glass panel ── */
[data-testid="column"]:nth-of-type(1),
[data-testid="column"]:nth-of-type(2),
[data-testid="column"]:nth-of-type(3) {
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 0.8rem 0.9rem !important;
    border: 1px solid rgba(255,255,255,0.14);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
}

/* ── Typography ── */
h1, h2, h3, h4 {
    color: #fff !important;
    font-weight: 700 !important;
    text-shadow: 0 1px 4px rgba(0,0,0,0.25);
}
h2 { font-size: 1.15rem !important; margin: 0 0 0.35rem !important; line-height: 1.25; }
h3 { font-size: 0.88rem !important; margin: 0.35rem 0 0.25rem !important; }
h4 { font-size: 0.82rem !important; margin: 0.2rem 0 !important; }
p, li { color: rgba(255,255,255,0.9) !important; font-size: 0.78rem; line-height: 1.5; margin: 0.08rem 0; }
strong { color: #fff !important; font-weight: 600 !important; }
em     { color: rgba(255,255,255,0.75) !important; font-style: italic; }
a      { color: #d8b4fe !important; }

/* ── App title ── */
.app-header {
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 0.3rem;
}
.app-header-icon { font-size: 1.3rem; }
.app-header-title { font-size: 1.05rem; font-weight: 700; color: #fff; line-height: 1.2; }
.app-subtitle { font-size: 0.72rem; color: rgba(255,255,255,0.75); line-height: 1.4; margin-bottom: 0.5rem; }

/* ── Divider ── */
.divider {
    border: none; border-top: 1px solid rgba(255,255,255,0.12);
    margin: 0.5rem 0;
}

/* ── Text area ── */
.stTextArea label {
    color: #fff !important; font-weight: 600 !important;
    font-size: 0.78rem !important; margin-bottom: 0.15rem !important;
}
.stTextArea textarea {
    background: rgba(0,0,0,0.28) !important; color: #fff !important;
    border: 1.5px solid rgba(255,255,255,0.22) !important;
    border-radius: 12px !important; font-size: 0.78rem !important;
    padding: 0.6rem 0.75rem !important; line-height: 1.5; resize: none;
    transition: border-color 0.25s, box-shadow 0.25s;
}
.stTextArea textarea:focus {
    border-color: rgba(200,150,255,0.7) !important;
    box-shadow: 0 0 14px rgba(200,150,255,0.3) !important;
    background: rgba(0,0,0,0.38) !important; outline: none !important;
}
.stTextArea textarea::placeholder { color: rgba(255,255,255,0.38) !important; font-style: italic; }

/* ── File uploader ── */
[data-testid="stFileUploader"] label {
    color: #fff !important; font-weight: 600 !important; font-size: 0.78rem !important;
}
[data-testid="stFileUploader"] section {
    background: rgba(0,0,0,0.2) !important;
    border: 1.5px dashed rgba(255,255,255,0.3) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(200,150,255,0.6) !important;
    background: rgba(0,0,0,0.3) !important;
}
[data-testid="stFileUploader"] section p {
    font-size: 0.72rem !important; color: rgba(255,255,255,0.6) !important;
}

/* ── Input mode tabs ── */
.input-tab {
    display: inline-block; padding: 0.2rem 0.7rem;
    border-radius: 20px; font-size: 0.72rem; font-weight: 600;
    cursor: pointer; margin-right: 0.3rem;
    border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.6);
    background: rgba(255,255,255,0.06);
}
.input-tab.active {
    background: linear-gradient(135deg,#c084fc,#f472b6);
    color: #fff; border-color: transparent;
}

/* ── Buttons ── */
.stButton { margin: 0.25rem 0; }
.stButton > button {
    background: linear-gradient(135deg, #c084fc 0%, #f472b6 100%) !important;
    color: #fff !important; font-weight: 600 !important;
    padding: 0 1rem !important; border-radius: 50px !important;
    border: none !important; font-size: 0.76rem !important;
    box-shadow: 0 4px 14px rgba(196,100,252,0.4) !important;
    transition: all 0.25s ease !important;
    width: 100%; height: 36px;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 20px rgba(196,100,252,0.6) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #c084fc 0%, #f472b6 100%) !important;
    color: #fff !important; font-weight: 600 !important;
    border-radius: 50px !important; border: none !important;
    font-size: 0.76rem !important; width: 100%; height: 36px;
    box-shadow: 0 4px 14px rgba(196,100,252,0.4) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.1) !important;
    border-radius: 9px !important; color: #fff !important;
    font-weight: 600 !important; padding: 0.38rem 0.7rem !important;
    font-size: 0.76rem !important; border: 1px solid rgba(255,255,255,0.12) !important;
}
.streamlit-expanderHeader:hover { background: rgba(255,255,255,0.17) !important; }
.streamlit-expanderContent {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 0 0 9px 9px !important; padding: 0.55rem !important;
}

/* ── Crisis card ── */
.crisis-card {
    background: rgba(255,255,255,0.09);
    border-radius: 10px; padding: 0.5rem 0.6rem;
    margin: 0.2rem 0; border: 1px solid rgba(255,255,255,0.15);
    font-size: 0.74rem; line-height: 1.65;
}
.crisis-card a { color: #d8b4fe !important; }

/* ── Immediate support pill cards ── */
.support-pill {
    background: rgba(255,255,255,0.1);
    border-radius: 10px; padding: 0.42rem 0.55rem;
    margin: 0.15rem 0; border: 1px solid rgba(255,255,255,0.15);
    font-size: 0.72rem; line-height: 1.6;
    text-align: center;
}
.support-pill strong { display: block; margin-bottom: 0.15rem; font-size: 0.74rem; }

/* ── Remember card ── */
.remember-card {
    background: rgba(255,255,255,0.07);
    border-radius: 10px; padding: 0.45rem 0.65rem;
    border: 1px solid rgba(255,255,255,0.13);
    font-size: 0.72rem; line-height: 1.9;
    text-align: center; margin-top: 0.4rem;
}
.remember-card span {
    color: rgba(255,255,255,0.82) !important;
    display: inline-block; margin: 0 0.3rem;
}

/* ── Result card ── */
.result-card {
    background: rgba(255,255,255,0.1);
    border-radius: 14px; padding: 0.65rem 0.8rem;
    margin: 0.3rem 0; border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
    animation: slideUp 0.35s ease-out;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Stat cards ── */
.stat-row { display: flex; gap: 0.3rem; margin-bottom: 0.3rem; }
.stat-card {
    flex: 1; background: rgba(255,255,255,0.1);
    border-radius: 10px; padding: 0.45rem 0.3rem;
    text-align: center; border: 1px solid rgba(255,255,255,0.16);
}
.stat-number { font-size: 1.2rem; font-weight: 700; color: #fff; }
.stat-label  { font-size: 0.6rem; color: rgba(255,255,255,0.68); text-transform: uppercase; letter-spacing: 0.7px; }

/* ── Confidence badge ── */
.conf-badge {
    display: inline-block; padding: 0.2rem 0.65rem;
    border-radius: 20px; font-size: 0.72rem; font-weight: 600;
}
.conf-high   { background: linear-gradient(135deg,#10b981,#34d399); color:#fff; }
.conf-medium { background: linear-gradient(135deg,#c084fc,#f472b6); color:#fff; }
.conf-low    { background: linear-gradient(135deg,#fbbf24,#1e40af); color:#fff; }

/* ── Risk badge ── */
.risk-high { color: #fca5a5 !important; font-weight: 700 !important; }
.risk-low  { color: #86efac !important; font-weight: 700 !important; }

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg,#c084fc,#f472b6);
    border-radius: 6px; height: 8px;
}
.stProgress > div > div { background: rgba(255,255,255,0.15); border-radius: 6px; }

/* ── Alert boxes ── */
.stWarning {
    background: rgba(251,191,36,0.14) !important; color:#fff !important;
    border-left: 3px solid #fbbf24 !important; border-radius: 9px !important;
    padding: 0.45rem 0.7rem !important; font-size: 0.75rem;
}
.stInfo {
    background: rgba(99,102,241,0.2) !important; color:#fff !important;
    border-left: 3px solid #818cf8 !important; border-radius: 9px !important;
    padding: 0.45rem 0.7rem !important; font-size: 0.75rem;
}
.stError {
    background: rgba(239,68,68,0.2) !important; color:#fff !important;
    border-left: 3px solid #f87171 !important; border-radius: 9px !important;
    padding: 0.45rem 0.7rem !important; font-size: 0.75rem; font-weight:600;
}

/* ── Footer ── */
.col-footer {
    font-size: 0.65rem; color: rgba(255,255,255,0.45);
    text-align: center; border-top: 1px solid rgba(255,255,255,0.1);
    padding-top: 0.35rem; margin-top: 0.4rem;
}

.js-plotly-plot { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_TWEETS = {
    "Positive 😊": "Just got promoted at work! Feeling blessed and grateful for this opportunity.",
    "Negative 😔": "I feel like nobody cares anymore. I am so depressed. What's the point of trying?"
}

# ─── Session state ──────────────────────────────────────────────────────────────
if 'analytics' not in st.session_state:
    st.session_state.analytics = {'total_analyses': 0, 'positive_count': 0, 'negative_count': 0, 'history': []}
if 'user_input'     not in st.session_state: st.session_state.user_input     = ""
if 'should_analyze' not in st.session_state: st.session_state.should_analyze = False
if 'last_result'    not in st.session_state: st.session_state.last_result    = None
if 'input_mode'     not in st.session_state: st.session_state.input_mode     = "text"  # "text" or "image"

# ─── Patch stale history entries (fixes KeyError on reloads) ──────────────────
for entry in st.session_state.analytics.get('history', []):
    entry.setdefault('cls',  'Unknown')
    entry.setdefault('ts',   '')
    entry.setdefault('prob', 0.0)
    entry.setdefault('txt',  '')

# ─── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = load_model("lstm_model.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

model, tokenizer = load_model_and_tokenizer()

# ─── Helpers ────────────────────────────────────────────────────────────────────
def clear_text():
    st.session_state.user_input     = ""
    st.session_state["text_area"]   = ""
    st.session_state.should_analyze = False
    st.session_state.last_result    = None

def update_analytics(prob, text):
    a = st.session_state.analytics
    a['total_analyses'] += 1
    cls = "Positive" if prob >= 0.5 else "Negative"
    if prob >= 0.5:
        a['positive_count'] += 1
    else:
        a['negative_count'] += 1
    a['history'].append({
        'ts':   datetime.now().strftime("%H:%M"),
        'cls':  cls,
        'prob': float(prob),
        'txt':  text[:38] + "…" if len(text) > 38 else text
    })
    if len(a['history']) > 10:
        a['history'] = a['history'][-10:]

def run_analysis(text):
    seq  = tokenizer.texts_to_sequences([text])
    pad  = pad_sequences(seq, maxlen=100)
    t0   = time.time()
    prob = model.predict(pad, verbose=0)[0][0]
    ms   = (time.time() - t0) * 1000
    update_analytics(prob, text)
    return float(prob), ms

def extract_text_from_image(image_file):
    """Extract text from uploaded image using OCR (pytesseract)."""
    try:
        img = Image.open(image_file).convert("RGB")
        text = pytesseract.image_to_string(img, config='--psm 6')
        return text.strip()
    except Exception as e:
        return None

def gauge(prob):
    if prob >= 0.5:
        intensity = (prob - 0.5) * 2
        clr = "#34d399"
        lbl = "Positive"
    else:
        intensity = (0.5 - prob) * 2
        clr = "#f87171"
        lbl = "Negative"

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=intensity * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{lbl} Intensity", 'font': {'color': 'white', 'size': 11}},
        number={'suffix': "%", 'font': {'color': 'white', 'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': 'white', 'tickfont': {'size': 8}},
            'bar': {'color': clr},
            'bgcolor': 'rgba(255,255,255,0.07)',
            'borderwidth': 1, 'bordercolor': 'rgba(255,255,255,0.3)',
            'steps': [
                {'range': [0,  33],  'color': 'rgba(255,255,255,0.05)'},
                {'range': [33, 66],  'color': 'rgba(255,255,255,0.09)'},
                {'range': [66, 100], 'color': 'rgba(255,255,255,0.13)'}
            ],
            'threshold': {'line': {'color': 'white', 'width': 2}, 'thickness': 0.65, 'value': 80}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}, height=165,
        margin=dict(l=6, r=6, t=28, b=2)
    )
    return fig

# ════════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════════════
colA, colB, colC = st.columns([1.0, 1.25, 1.05])
is_high_risk = False

# ── COL A — Input ────────────────────────────────────────────────────────────
with colA:
    st.markdown("""
    <div class="app-header">
        <span class="app-header-title">Suicidal Tweet Detector</span>
    </div>
    <p class="app-subtitle">LSTM model · detects suicidal ideation in tweets.<br><em>Enter text or upload a screenshot to begin.</em></p>
    <hr class="divider">
    """, unsafe_allow_html=True)

    # ── Input mode toggle ────────────────────────────────────────────────────
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        if st.button(" Type Text",
                     use_container_width=True,
                     type="primary" if st.session_state.input_mode == "text" else "secondary"):
            st.session_state.input_mode = "text"
            st.rerun()
    with mode_col2:
        if st.button("🖼️ Upload Image",
                     use_container_width=True,
                     type="primary" if st.session_state.input_mode == "image" else "secondary"):
            st.session_state.input_mode = "image"
            st.rerun()

    st.markdown('<div style="margin-top:0.3rem"></div>', unsafe_allow_html=True)

    # ── TEXT MODE ────────────────────────────────────────────────────────────
    if st.session_state.input_mode == "text":

        with st.expander(" Try Sample Tweet", expanded=False):
            for label, tweet in SAMPLE_TWEETS.items():
                if st.button(label, key=f"sample_{label}", use_container_width=True):
                    st.session_state.user_input     = tweet
                    st.session_state["text_area"]   = tweet
                    st.session_state.should_analyze = True
                    st.rerun()

        user_input = st.text_area(
            "Enter Your Tweet Here to Begin:",
            height=108,
            placeholder="Type or paste a tweet here…",
            value=st.session_state.user_input,
            key="text_area"
        )
        st.session_state.user_input = user_input

        b1, b2 = st.columns([1.6, 1])
        with b1: analyze_btn = st.button("🔍 Analyze", use_container_width=True, key="analyze_text")
        with b2: st.button("🗑️ Clear", use_container_width=True, on_click=clear_text)

        if analyze_btn:
            if user_input.strip():
                p, ms = run_analysis(user_input)
                st.session_state.last_result = {'prob': p, 'ms': ms, 'text': user_input, 'ok': True}
            else:
                st.session_state.last_result = {'ok': False, 'empty': True}
            st.rerun()

        if st.session_state.should_analyze and st.session_state.user_input.strip():
            st.session_state.should_analyze = False
            p, ms = run_analysis(st.session_state.user_input)
            st.session_state.last_result = {'prob': p, 'ms': ms, 'text': st.session_state.user_input, 'ok': True}
            st.rerun()

    # ── IMAGE MODE ───────────────────────────────────────────────────────────
    else:
        st.markdown(
            '<p style="font-size:0.75rem;font-weight:600;margin-bottom:0.2rem">📸 Upload a screenshot:</p>',
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(
            "Upload screenshot",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, use_container_width=True, caption="Uploaded screenshot")

        img_b1, img_b2 = st.columns([1.6, 1])
        with img_b1:
            analyze_img_btn = st.button("🔍 Analyze Image", use_container_width=True, key="analyze_image")
        with img_b2:
            st.button("🗑️ Clear", use_container_width=True, on_click=clear_text, key="clear_image")

        if analyze_img_btn:
            if uploaded_file:
                with st.spinner("Reading text from image…"):
                    extracted = extract_text_from_image(uploaded_file)
                if extracted:
                    p, ms = run_analysis(extracted)
                    st.session_state.last_result = {
                        'prob': p, 'ms': ms, 'text': extracted, 'ok': True, 'from_image': True
                    }
                else:
                    st.session_state.last_result = {'ok': False, 'ocr_fail': True}
            else:
                st.session_state.last_result = {'ok': False, 'no_image': True}
            st.rerun()

    st.markdown('<div class="col-footer">Built with ❤️ Streamlit + LSTM · Mental Health Awareness</div>', unsafe_allow_html=True)


# ── COL B — Crisis info + Result ────────────────────────────────────────────
with colB:

    # ── Always-visible Crisis Resources (TOP of col B) ──────────────────────
    st.markdown("""
    <p style="font-size:0.72rem;font-weight:700;color:rgba(255,255,255,0.85);margin:0 0 0.25rem;letter-spacing:0.5px">
        🆘 CRISIS HELPLINES — Available 24/7
    </p>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.25rem;margin-bottom:0.4rem">
        <div class="support-pill"><strong>🇰🇪 Kenya</strong>📞 <em>Kenya Red Cross</em><br>1199<br>📞 <em>Befrienders</em><br>+254 722 178 177</div>
        <div class="support-pill"><strong>🇺🇸 US</strong>📞 <em>Suicide & Crisis Lifeline</em><br>988<br>💬 <em>Crisis Text Line</em><br>HOME → 741741</div>
        <div class="support-pill"><strong>🇬🇧 UK</strong>📞 <em>Samaritans</em><br>116 123</div>
        <div class="support-pill"><strong>🌍 Intl</strong>🔗 <em>Find A Helpline</em><br><a href="https://findahelpline.com" target="_blank">findahelpline.com</a></div>
    </div>
    <hr class="divider">
    """, unsafe_allow_html=True)

    r = st.session_state.last_result

    # ── Error states ─────────────────────────────────────────────────────────
    if r and r.get('ok') is False:
        if r.get('empty'):
            st.warning("⚠️ Please enter some text first.")
        elif r.get('no_image'):
            st.warning("⚠️ Please upload an image first.")
        elif r.get('ocr_fail'):
            st.warning("⚠️ Could not read text from the image. Try a clearer screenshot.")

    if r and r.get('ok'):
        prob = r['prob']
        is_high_risk = prob < 0.5

        # Show OCR source badge if result came from image
        if r.get('from_image'):
            st.markdown(
                '<div style="text-align:center;margin-bottom:0.3rem">'
                '<span style="background:rgba(99,102,241,0.35);color:#c7d2fe;padding:3px 12px;'
                'border-radius:999px;font-size:0.67rem;font-weight:600;border:1px solid rgba(99,102,241,0.4)">'
                '🖼️ Result from image OCR</span></div>',
                unsafe_allow_html=True
            )

        label    = "🔴 Suicidal / Negative"  if prob < 0.5 else "🟢 Non-Suicidal / Positive"
        color    = "#f87171"                  if prob < 0.5 else "#34d399"
        risk_lbl = "HIGH RISK"                if prob < 0.5 else "LOW RISK"
        risk_cls = "risk-high"                if prob < 0.5 else "risk-low"
        conf     = prob if prob >= 0.5 else (1 - prob)

        if conf >= 0.8:   cl, cc = "High Confidence",   "conf-high"
        elif conf >= 0.6: cl, cc = "Medium Confidence", "conf-medium"
        else:             cl, cc = "Low Confidence",    "conf-low"

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:1rem;font-weight:700;color:{color};text-align:center;margin:0 0 0.3rem">{label}</p>',
            unsafe_allow_html=True
        )
        st.plotly_chart(gauge(prob), use_container_width=True)
        st.markdown(
            f'<p style="font-size:0.76rem;margin:0.15rem 0"><strong>Risk:</strong> <span class="{risk_cls}">{risk_lbl}</span></p>',
            unsafe_allow_html=True
        )
        st.progress(int(prob * 100) if prob >= 0.5 else int((1 - prob) * 100))
        st.markdown(
            f'<div style="text-align:center;margin:0.25rem 0"><span class="conf-badge {cc}">{cl}: {conf:.1%}</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="text-align:center;margin:0.25rem 0 0.4rem"><span style="background:linear-gradient(135deg,#43e97b,#38f9d7);color:#0b1727;padding:4px 14px;border-radius:999px;font-size:0.68rem;font-weight:600">⚡ {r["ms"]:.1f}ms</span></div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        res_txt = (f"Tweet:\n{r['text']}\n\nPrediction: {label.strip()}\n"
                   f"Risk: {risk_lbl}\nConfidence: {conf:.1%}\n"
                   f"Latency: {r['ms']:.1f}ms\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        st.text_area("📋 Copy result:", res_txt, height=72)
        st.download_button("📄 Download Result", res_txt, file_name="analysis.txt", use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        if is_high_risk:
            st.error("🚨 **CRISIS ALERT** — High-risk content detected! Please use the helplines above.")

    # ── Always-visible Remember message (BOTTOM of col B) ───────────────────
    st.markdown("""
    <div class="remember-card">
        <strong style="font-size:0.74rem;color:#fff;display:block;margin-bottom:0.15rem">💙 Remember</strong>
        <span>🤝 You are not alone</span>
        <span style="color:rgba(255,255,255,0.35)">·</span>
        <span>🕐 Help is available 24/7</span>
        <span style="color:rgba(255,255,255,0.35)">·</span>
        <span>💬 Speaking to someone can make a difference</span>
    </div>
    """, unsafe_allow_html=True)


# ── COL C — Analytics ────────────────────────────────────────────────────────
with colC:
    st.markdown('<h3 style="text-align:center;margin:0 0 0.4rem">📊 Analytics</h3>', unsafe_allow_html=True)

    a = st.session_state.analytics
    if a['total_analyses'] > 0:
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-label">Total</div>
                <div class="stat-number">{a['total_analyses']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Positive</div>
                <div class="stat-number" style="color:#34d399">{a['positive_count']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Negative</div>
                <div class="stat-number" style="color:#f87171">{a['negative_count']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig_pie = go.Figure(go.Pie(
            labels=['Positive', 'Negative'],
            values=[a['positive_count'], a['negative_count']],
            marker_colors=['#34d399', '#f87171'],
            hole=0.38,
            textfont_size=10,
            textfont_color='white'
        ))
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}, height=185,
            margin=dict(l=5, r=5, t=8, b=5),
            legend=dict(font=dict(color='white', size=9), orientation='v', x=1.0, y=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.74rem;font-weight:600;margin-bottom:0.2rem">📝 Recent Analyses</p>', unsafe_allow_html=True)

        for item in reversed(a['history'][-5:]):
            cls  = item.get('cls',  'Unknown')
            ts   = item.get('ts',   '')
            prob = item.get('prob', 0.0)
            txt  = item.get('txt',  '')
            emoji = "🟢" if cls == "Positive" else "🔴"
            st.markdown(
                f'<p style="margin:0.1rem 0;font-size:0.69rem">{emoji} <strong>{cls}</strong> · {ts} · {prob:.0%}<br>'
                f'<em style="color:rgba(255,255,255,0.6)">{txt}</em></p>',
                unsafe_allow_html=True
            )
    else:
        st.markdown("""
        <div style="text-align:center;padding:2rem 0.5rem;color:rgba(255,255,255,0.5)">
            <div style="font-size:2rem;margin-bottom:0.5rem">📊</div>
            <p style="font-size:0.76rem">No analyses yet.<br>Run your first scan to see stats here.</p>
        </div>
        """, unsafe_allow_html=True)
