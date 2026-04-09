import streamlit as st
import pickle
import numpy as np
import time
import re
import os
import datetime
import subprocess
import tempfile
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pytesseract

st.set_page_config(
    page_title="MindGuard — Suicidal Ideation Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS  (your original styles kept + new additions)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

#MainMenu, footer, header { visibility: hidden; }
.stAlert { display: none !important; }
[data-testid="stHeader"]       { display: none !important; height: 0 !important; }
[data-testid="stToolbar"]      { display: none !important; }
[data-testid="stDecoration"]   { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }

html, body { margin: 0; padding: 0; }
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }

.stApp {
    background: linear-gradient(-45deg, #0d1b2a, #132233, #1a0a2e, #0d2233);
    background-size: 400% 400%;
    animation: gradientShift 22s ease infinite;
    font-family: 'Inter', sans-serif;
}
@keyframes gradientShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}

.main .block-container {
    max-width: 100% !important;
    padding: 0.6rem 0.8rem 0.4rem 0.8rem !important;
    margin: 0 !important;
}

/* Tab styling */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stTabs"] button[role="tab"] {
    color: rgba(255,255,255,0.6) !important;
    border-radius: 9px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 6px 14px !important;
    border: none !important;
    transition: all 0.2s;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg,#0d9488,#5eead4) !important;
    color: #0d1b2a !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: #fff !important;
    background: rgba(255,255,255,0.1) !important;
}

h1,h2,h3,h4 { color:#fff !important; font-weight:700 !important; text-shadow:0 1px 4px rgba(0,0,0,0.25); }
h2 { font-size:1.1rem !important; margin:0 0 0.3rem !important; }
h3 { font-size:0.88rem !important; margin:0.3rem 0 0.2rem !important; }
p,li { color:rgba(255,255,255,0.88) !important; font-size:0.78rem; line-height:1.5; margin:0.06rem 0; }
strong { color:#fff !important; font-weight:600 !important; }
em { color:rgba(255,255,255,0.72) !important; font-style:italic; }
a  { color:#5eead4 !important; }

.app-header { display:flex; align-items:center; gap:0.5rem; margin-bottom:0.2rem; }
.app-header-title { font-size:1.2rem; font-weight:700; color:#fff; }
.app-subtitle { font-size:0.72rem; color:rgba(255,255,255,0.7); margin-bottom:0.4rem; }
.divider { border:none; border-top:1px solid rgba(255,255,255,0.1); margin:0.45rem 0; }

/* Text inputs */
.stTextArea label, .stTextInput label {
    color:#fff !important; font-weight:600 !important; font-size:0.78rem !important;
}
.stTextArea textarea, .stTextInput input {
    background:rgba(0,0,0,0.28) !important; color:#fff !important;
    border:1.5px solid rgba(255,255,255,0.2) !important;
    border-radius:12px !important; font-size:0.78rem !important;
    padding:0.55rem 0.7rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color:rgba(94,234,212,0.7) !important;
    box-shadow:0 0 14px rgba(94,234,212,0.2) !important;
    outline:none !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color:rgba(255,255,255,0.35) !important; font-style:italic;
}

/* File uploader */
[data-testid="stFileUploader"] label { color:#fff !important; font-size:0.78rem !important; font-weight:600 !important; }
[data-testid="stFileUploader"] section {
    background:rgba(0,0,0,0.2) !important;
    border:1.5px dashed rgba(255,255,255,0.25) !important;
    border-radius:12px !important; padding:0.5rem !important;
}
[data-testid="stFileUploader"] section p { font-size:0.72rem !important; color:rgba(255,255,255,0.55) !important; }

/* Buttons */
.stButton > button {
    background:linear-gradient(135deg,#0d9488,#5eead4) !important;
    color:#0d1b2a !important; font-weight:700 !important;
    padding:0 1rem !important; border-radius:50px !important;
    border:none !important; font-size:0.76rem !important;
    box-shadow:0 4px 14px rgba(13,148,136,0.35) !important;
    transition:all 0.25s ease !important; width:100%; height:36px;
}
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 7px 20px rgba(13,148,136,0.55) !important; }
[data-testid="stDownloadButton"] > button {
    background:rgba(255,255,255,0.1) !important; color:#fff !important;
    border:1px solid rgba(255,255,255,0.2) !important; border-radius:50px !important;
    font-size:0.72rem !important; height:32px; padding:0 0.8rem !important;
}

/* Cards */
.result-card {
    background:rgba(255,255,255,0.07); border-radius:14px;
    padding:0.65rem 0.8rem; margin:0.3rem 0;
    border:1px solid rgba(255,255,255,0.14);
    box-shadow:0 4px 18px rgba(0,0,0,0.12);
    animation:slideUp 0.35s ease-out;
}
@keyframes slideUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }

.post-card {
    background:rgba(255,255,255,0.06); border-radius:10px;
    padding:0.55rem 0.7rem; margin:0.25rem 0;
    border-left:4px solid #0d9488; font-size:0.74rem;
}
.post-card.high   { border-left-color:#ef4444; }
.post-card.medium { border-left-color:#f59e0b; }
.post-card.low    { border-left-color:#22c55e; }

.resource-card {
    background:rgba(255,255,255,0.06); border-radius:10px;
    padding:0.45rem 0.65rem; margin:0.2rem 0;
    border-left:4px solid #7c3aed; font-size:0.74rem;
}
.socio-tag {
    display:inline-block; background:rgba(13,148,136,0.2);
    color:#5eead4; border-radius:6px; padding:2px 8px;
    margin:2px; font-size:0.72rem;
    border:1px solid rgba(13,148,136,0.3);
}

.stat-row { display:flex; gap:0.3rem; margin-bottom:0.3rem; }
.stat-card {
    flex:1; background:rgba(255,255,255,0.08);
    border-radius:10px; padding:0.4rem 0.25rem;
    text-align:center; border:1px solid rgba(255,255,255,0.13);
}
.stat-number { font-size:1.15rem; font-weight:700; color:#5eead4; }
.stat-label  { font-size:0.6rem; color:rgba(255,255,255,0.6); text-transform:uppercase; letter-spacing:0.6px; }

.conf-badge { display:inline-block; padding:0.2rem 0.6rem; border-radius:20px; font-size:0.72rem; font-weight:600; }
.conf-high   { background:linear-gradient(135deg,#0d9488,#5eead4); color:#0d1b2a; }
.conf-medium { background:linear-gradient(135deg,#7c3aed,#a78bfa); color:#fff; }
.conf-low    { background:linear-gradient(135deg,#b45309,#fbbf24); color:#0d1b2a; }

.risk-high { color:#fca5a5 !important; font-weight:700 !important; }
.risk-low  { color:#86efac !important; font-weight:700 !important; }

.stProgress > div > div > div > div { background:linear-gradient(90deg,#0d9488,#5eead4); border-radius:6px; height:8px; }
.stProgress > div > div { background:rgba(255,255,255,0.12); border-radius:6px; }

.stWarning { background:rgba(251,191,36,0.12) !important; color:#fff !important; border-left:3px solid #fbbf24 !important; border-radius:9px !important; padding:0.4rem 0.65rem !important; font-size:0.75rem; }
.stInfo    { background:rgba(13,148,136,0.15) !important;  color:#fff !important; border-left:3px solid #0d9488 !important; border-radius:9px !important; padding:0.4rem 0.65rem !important; font-size:0.75rem; }
.stError   { background:rgba(239,68,68,0.18) !important;   color:#fff !important; border-left:3px solid #f87171 !important; border-radius:9px !important; padding:0.4rem 0.65rem !important; font-size:0.75rem; font-weight:600; }
.stSuccess { background:rgba(13,148,136,0.18) !important;  color:#fff !important; border-left:3px solid #5eead4 !important; border-radius:9px !important; padding:0.4rem 0.65rem !important; font-size:0.75rem; }

.section-label {
    font-size:0.7rem; font-weight:700; color:#5eead4;
    letter-spacing:0.1em; text-transform:uppercase;
    margin:0.6rem 0 0.3rem;
}

.remember-card {
    background:rgba(255,255,255,0.06); border-radius:10px;
    padding:0.4rem 0.6rem; border:1px solid rgba(255,255,255,0.12);
    font-size:0.72rem; text-align:center; margin-top:0.3rem;
}
.support-pill {
    background:rgba(255,255,255,0.08); border-radius:10px;
    padding:0.35rem 0.5rem; margin:0.12rem 0;
    border:1px solid rgba(255,255,255,0.13);
    font-size:0.7rem; line-height:1.7; text-align:center;
}
.support-pill strong { display:block; margin-bottom:0.1rem; }

.col-footer {
    font-size:0.65rem; color:rgba(255,255,255,0.4);
    text-align:center; border-top:1px solid rgba(255,255,255,0.08);
    padding-top:0.3rem; margin-top:0.5rem;
}

.tiktok-badge {
    display:inline-block; background:rgba(255,255,255,0.1);
    border:1px solid rgba(255,255,255,0.18); border-radius:8px;
    padding:0.3rem 0.6rem; font-size:0.72rem; margin-bottom:0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_TWEETS = {
    "Positive 😊": "Just got promoted at work! Feeling blessed and grateful for this opportunity.",
    "Negative 😔": "I feel like nobody cares anymore. I am so depressed. What's the point of trying?"
}

STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can't","cannot","could","couldn't","did",
    "didn't","do","does","doesn't","doing","don't","down","during","each",
    "few","for","from","further","get","got","had","hadn't","has","hasn't",
    "have","haven't","having","he","her","here","hers","herself","him",
    "himself","his","how","i","if","in","into","is","isn't","it","its",
    "itself","me","more","most","my","myself","no","nor","not","of","off",
    "on","once","only","or","other","our","ours","ourselves","out","over",
    "own","same","she","should","shouldn't","so","some","such","than","that",
    "the","their","theirs","them","themselves","then","there","these","they",
    "this","those","through","to","too","under","until","up","very","was",
    "wasn't","we","were","weren't","what","when","where","which","while",
    "who","whom","why","will","with","won't","would","wouldn't","you","your",
    "yours","yourself","yourselves",
}

SOCIOECONOMIC_KEYWORDS = {
    "💼 Employment": [
        "unemployed","lost job","fired","laid off","no income","can't find work",
        "job rejection","jobless","redundant","no work","rejected again",
    ],
    "🏠 Housing": [
        "evicted","homeless","can't pay rent","losing house","sleeping rough",
        "no home","couch surfing","rent behind","foreclosure","shelter",
    ],
    "💸 Financial": [
        "broke","debt","can't afford","no money","bankruptcy","loan",
        "can't pay","struggling financially","poverty","bills piling",
    ],
    "💔 Relationships": [
        "divorce","breakup","separated","cheated","alone","abandoned",
        "no one cares","nobody loves","relationship ended","widowed",
        "nobody cares","lost my partner",
    ],
    "🏥 Health": [
        "chronic pain","terminal","diagnosis","mental illness","addiction",
        "substance abuse","alcoholic","overdose","hospitalized","incurable",
    ],
}

RESOURCES = {
    "🇰🇪 Kenya": [
        {"name":"Befrienders Kenya",        "contact":"+254 722 178 177", "type":"Crisis line"},
        {"name":"Kenya Red Cross",           "contact":"1199",             "type":"Emergency"},
        {"name":"Chiromo Hospital Group",    "contact":"+254 20 4291000",  "type":"Mental health"},
        {"name":"Mathare Hospital MH Unit",  "contact":"+254 20 2012185",  "type":"Hospital"},
    ],
    "🇺🇸 USA": [
        {"name":"988 Suicide & Crisis Lifeline","contact":"Call/text 988",      "type":"Crisis line"},
        {"name":"Crisis Text Line",             "contact":"Text HOME → 741741", "type":"Text-based"},
        {"name":"NAMI Helpline",                "contact":"1-800-950-6264",     "type":"Mental health"},
    ],
    "🇬🇧 UK": [
        {"name":"Samaritans",           "contact":"116 123",       "type":"Crisis line"},
        {"name":"PAPYRUS (under 35s)",  "contact":"0800 068 4141", "type":"Youth crisis"},
        {"name":"MIND",                 "contact":"0300 123 3393", "type":"Mental health"},
    ],
    "🌍 International": [
        {"name":"Find A Helpline",  "contact":"findahelpline.com",                       "type":"Global directory"},
        {"name":"IASP",             "contact":"https://www.iasp.info/resources/Crisis_Centres/", "type":"Global directory"},
    ],
}

SIX_MONTHS_AGO = datetime.datetime.utcnow() - datetime.timedelta(days=182)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "analytics":      {"total_analyses":0,"positive_count":0,"negative_count":0,"history":[]},
    "user_input":     "",
    "should_analyze": False,
    "last_result":    None,
    "input_mode":     "text",
    "download_text":  "",
    "reddit_results": None,
    "tiktok_result":  None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

for entry in st.session_state.analytics.get("history", []):
    for field, default in [("cls","Unknown"),("ts",""),("prob",0.0),("txt","")]:
        entry.setdefault(field, default)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = load_model("lstm_model.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

model, tokenizer = load_model_and_tokenizer()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — text & prediction
# ══════════════════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 2)

def predict_one(text: str):
    """Return (prob, latency_ms). prob≥0.5 → non-suicidal, <0.5 → suicidal."""
    seq  = tokenizer.texts_to_sequences([text])
    pad  = pad_sequences(seq, maxlen=100)
    t0   = time.time()
    prob = float(model.predict(pad, verbose=0)[0][0])
    ms   = (time.time() - t0) * 1000
    return prob, ms

def predict_batch(texts: list) -> np.ndarray:
    """Return array of risk scores (1 = suicidal) for many texts at once."""
    seqs  = tokenizer.texts_to_sequences(texts)
    pads  = pad_sequences(seqs, maxlen=100)
    preds = model.predict(pads, verbose=0).flatten()
    # Invert: model outputs high=non-suicidal; we want high=suicidal for risk score
    return 1.0 - preds

def risk_label(score: float):
    """score is suicidal risk [0,1]; returns (label, hex_color, css_class)"""
    if score < 0.35:   return "Low Risk",      "#22c55e", "low"
    elif score < 0.55: return "Moderate Risk",  "#f59e0b", "medium"
    elif score < 0.75: return "High Risk",      "#f97316", "high"
    else:              return "Critical Risk",  "#ef4444", "high"

def update_analytics(prob, text):
    a   = st.session_state.analytics
    a["total_analyses"] += 1
    cls = "Positive" if prob >= 0.5 else "Negative"
    if prob >= 0.5: a["positive_count"] += 1
    else:           a["negative_count"] += 1
    a["history"].append({
        "ts":   datetime.datetime.now().strftime("%H:%M"),
        "cls":  cls,
        "prob": prob,
        "txt":  (text[:38] + "…") if len(text) > 38 else text,
    })
    if len(a["history"]) > 10:
        a["history"] = a["history"][-10:]

def run_analysis(text):
    prob, ms = predict_one(text)
    update_analytics(prob, text)
    return prob, ms

def build_download_text(text, prob, ms):
    label = "Suicidal / Negative" if prob < 0.5 else "Non-Suicidal / Positive"
    risk  = "HIGH RISK"           if prob < 0.5 else "LOW RISK"
    conf  = prob if prob >= 0.5 else (1 - prob)
    return (f"Text:\n{text}\n\nPrediction: {label}\nRisk: {risk}\n"
            f"Confidence: {conf:.1%}\nLatency: {ms:.1f}ms\n"
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def detect_socioeconomic(posts: list) -> dict:
    full = " ".join(p["text"].lower() for p in posts)
    return {cat: [kw for kw in kws if kw in full]
            for cat, kws in SOCIOECONOMIC_KEYWORDS.items()}

def clear_text():
    st.session_state.user_input     = ""
    st.session_state["text_area"]   = ""
    st.session_state.should_analyze = False
    st.session_state.last_result    = None
    st.session_state.download_text  = ""

def extract_text_from_image(image_file):
    try:
        img  = Image.open(image_file).convert("RGB")
        text = pytesseract.image_to_string(img, config="--psm 6")
        return text.strip()
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — Reddit
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_reddit_posts(username: str, client_id: str, client_secret: str) -> list:
    import praw
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent="MindGuard:v2.0 (mental health research)",
    )
    posts = []
    try:
        redditor = reddit.redditor(username)
        for sub in redditor.submissions.new(limit=200):
            dt = datetime.datetime.utcfromtimestamp(sub.created_utc)
            if dt < SIX_MONTHS_AGO:
                break
            text = f"{sub.title} {sub.selftext}".strip()
            if len(text) > 10:
                posts.append({"text": text, "date": dt,
                              "subreddit": str(sub.subreddit),
                              "type": "post",
                              "url": f"https://reddit.com{sub.permalink}"})
        for c in redditor.comments.new(limit=500):
            dt = datetime.datetime.utcfromtimestamp(c.created_utc)
            if dt < SIX_MONTHS_AGO:
                break
            text = c.body.strip()
            if len(text) > 10 and text not in ("[deleted]", "[removed]"):
                posts.append({"text": text, "date": dt,
                              "subreddit": str(c.subreddit),
                              "type": "comment",
                              "url": f"https://reddit.com{c.permalink}"})
    except Exception as e:
        raise RuntimeError(str(e))
    posts.sort(key=lambda x: x["date"])
    return posts

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — TikTok
# ══════════════════════════════════════════════════════════════════════════════
def download_tiktok_audio(url: str, out_dir: str) -> str:
    """Download TikTok video audio track as mp3. Returns path to file."""
    out_template = os.path.join(out_dir, "audio.%(ext)s")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "5",
        "--no-playlist",
        "--quiet",
        "-o", out_template,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error: {result.stderr.strip()}")
    mp3 = os.path.join(out_dir, "audio.mp3")
    if not os.path.exists(mp3):
        # fallback — find whatever was downloaded
        files = list(Path(out_dir).glob("audio.*"))
        if not files:
            raise RuntimeError("Audio file not found after download.")
        mp3 = str(files[0])
    return mp3

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to text using faster-whisper (tiny model, CPU)."""
    try:
        from faster_whisper import WhisperModel
        # tiny model — fast on CPU, ~75MB download on first run
        wm = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = wm.transcribe(audio_path, beam_size=3)
        return " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def gauge(prob):
    """Original gauge chart — prob≥0.5 → positive (green), <0.5 → negative (red)."""
    if prob >= 0.5:
        intensity = (prob - 0.5) * 2;  clr = "#5eead4"; lbl = "Non-Suicidal"
    else:
        intensity = (0.5 - prob) * 2;  clr = "#f87171"; lbl = "Suicidal Risk"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=intensity*100,
        domain={"x":[0,1],"y":[0,1]},
        title={"text": lbl, "font":{"color":"white","size":11}},
        number={"suffix":"%","font":{"color":"white","size":24}},
        gauge={
            "axis":{"range":[None,100],"tickwidth":1,"tickcolor":"white","tickfont":{"size":8}},
            "bar":{"color":clr},
            "bgcolor":"rgba(255,255,255,0.06)",
            "borderwidth":1,"bordercolor":"rgba(255,255,255,0.25)",
            "steps":[
                {"range":[0,33],  "color":"rgba(255,255,255,0.04)"},
                {"range":[33,66], "color":"rgba(255,255,255,0.07)"},
                {"range":[66,100],"color":"rgba(255,255,255,0.11)"},
            ],
            "threshold":{"line":{"color":"white","width":2},"thickness":0.65,"value":80}
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"white"}, height=165, margin=dict(l=6,r=6,t=28,b=2)
    )
    return fig

def timeline_chart(df):
    import pandas as pd
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    weekly = (df.groupby("week")["risk_score"]
                .agg(["mean","max","count"])
                .reset_index()
                .rename(columns={"mean":"avg","max":"peak","count":"posts"}))
    fig = go.Figure()
    for y0,y1,col,lbl in [
        (0.00,0.35,"rgba(34,197,94,0.07)","Low"),
        (0.35,0.55,"rgba(245,158,11,0.07)","Moderate"),
        (0.55,0.75,"rgba(249,115,22,0.07)","High"),
        (0.75,1.00,"rgba(239,68,68,0.09)","Critical"),
    ]:
        fig.add_hrect(y0=y0,y1=y1,fillcolor=col,line_width=0,
                      annotation_text=lbl,annotation_position="right",
                      annotation=dict(font_color="rgba(255,255,255,0.5)",font_size=9))
    fig.add_bar(x=weekly["week"],y=weekly["posts"],name="Posts/week",
                marker_color="rgba(13,148,136,0.2)",yaxis="y2",
                hovertemplate="Week %{x}<br>Posts: %{y}<extra></extra>")
    fig.add_scatter(x=weekly["week"],y=weekly["avg"],mode="lines+markers",
                    name="Avg risk",line=dict(color="#5eead4",width=2),
                    marker=dict(size=5,color="#5eead4"),
                    hovertemplate="%{x}<br>Avg: %{y:.1%}<extra></extra>")
    fig.add_scatter(x=weekly["week"],y=weekly["peak"],mode="lines",
                    name="Peak risk",line=dict(color="#ef4444",width=1.5,dash="dot"),
                    hovertemplate="%{x}<br>Peak: %{y:.1%}<extra></extra>")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(19,34,51,0.5)",
        font_color="rgba(255,255,255,0.7)",
        yaxis=dict(title="Risk",tickformat=".0%",range=[0,1],
                   gridcolor="rgba(255,255,255,0.07)",color="rgba(255,255,255,0.6)"),
        yaxis2=dict(overlaying="y",side="right",showgrid=False,
                    color="rgba(255,255,255,0.4)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)",color="rgba(255,255,255,0.6)"),
        legend=dict(orientation="h",y=-0.22,font_color="rgba(255,255,255,0.7)",font_size=10),
        margin=dict(l=40,r=50,t=10,b=40), height=280,
    )
    return fig

def subreddit_chart(df):
    sub = (df.groupby("subreddit")
             .agg(posts=("risk_score","count"),avg_risk=("risk_score","mean"))
             .reset_index()
             .sort_values("avg_risk",ascending=False).head(12))
    fig = px.bar(sub,x="avg_risk",y="subreddit",orientation="h",
                 color="avg_risk",
                 color_continuous_scale=["#22c55e","#f59e0b","#f97316","#ef4444"],
                 range_color=[0,1],
                 text=sub["posts"].astype(str)+" posts")
    fig.update_traces(textposition="outside",textfont_color="rgba(255,255,255,0.7)")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(19,34,51,0.5)",
        font_color="rgba(255,255,255,0.7)",coloraxis_showscale=False,
        xaxis=dict(tickformat=".0%",gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        margin=dict(l=10,r=60,t=10,b=20),
        height=max(200,len(sub)*32),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <span style="font-size:1.7rem">🧠</span>
    <div>
        <div class="app-header-title">MindGuard</div>
        <div class="app-subtitle">Early detection of suicidal ideation · Bi-LSTM NLP model</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab_text, tab_reddit, tab_tiktok, tab_resources = st.tabs([
    "✍️ Text / Image",
    "🔴 Reddit Analysis",
    "🎵 TikTok Video",
    "📍 Resources & Info",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Text / Image  (your original feature, preserved)
# ══════════════════════════════════════════════════════════════════════════════
with tab_text:
    colA, colB, colC = st.columns([1.0, 1.25, 1.05])
    is_high_risk = False

    # ── Col A — input ──────────────────────────────────────────────────────
    with colA:
        st.markdown('<h2>Input</h2>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            if st.button("✍️ Type Text", use_container_width=True):
                st.session_state.input_mode = "text"; st.rerun()
        with mode_col2:
            if st.button("🖼️ Upload Image", use_container_width=True):
                st.session_state.input_mode = "image"; st.rerun()

        st.markdown('<div style="margin-top:0.3rem"></div>', unsafe_allow_html=True)

        if st.session_state.input_mode == "text":
            with st.expander("🔖 Try a sample", expanded=False):
                for label, tweet in SAMPLE_TWEETS.items():
                    if st.button(label, key=f"sample_{label}", use_container_width=True):
                        st.session_state.user_input     = tweet
                        st.session_state["text_area"]   = tweet
                        st.session_state.should_analyze = True
                        st.rerun()

            user_input = st.text_area(
                "Enter text to analyse:",
                height=108,
                placeholder="Type or paste text here…",
                value=st.session_state.user_input,
                key="text_area",
            )
            st.session_state.user_input = user_input

            b1, b2 = st.columns([1.6, 1])
            with b1: analyze_btn = st.button("🔍 Analyse", use_container_width=True, key="analyze_text")
            with b2: st.button("🗑️ Clear", use_container_width=True, on_click=clear_text)

            if analyze_btn:
                if user_input.strip():
                    p, ms = run_analysis(user_input)
                    st.session_state.last_result   = {"prob":p,"ms":ms,"text":user_input,"ok":True}
                    st.session_state.download_text = build_download_text(user_input,p,ms)
                else:
                    st.session_state.last_result = {"ok":False,"empty":True}
                st.rerun()

            if st.session_state.should_analyze and st.session_state.user_input.strip():
                st.session_state.should_analyze = False
                p, ms = run_analysis(st.session_state.user_input)
                st.session_state.last_result   = {"prob":p,"ms":ms,"text":st.session_state.user_input,"ok":True}
                st.session_state.download_text = build_download_text(st.session_state.user_input,p,ms)
                st.rerun()

        else:  # image mode
            uploaded_file = st.file_uploader(
                "Upload a screenshot:",
                type=["png","jpg","jpeg","webp"],
                label_visibility="collapsed"
            )
            if uploaded_file:
                st.image(Image.open(uploaded_file), use_container_width=True, caption="Uploaded screenshot")

            img_b1, img_b2 = st.columns([1.6, 1])
            with img_b1:
                analyze_img_btn = st.button("🔍 Analyse Image", use_container_width=True, key="analyze_image")
            with img_b2:
                st.button("🗑️ Clear", use_container_width=True, on_click=clear_text, key="clear_image")

            if analyze_img_btn:
                if uploaded_file:
                    with st.spinner("Reading text from image…"):
                        extracted = extract_text_from_image(uploaded_file)
                    if extracted:
                        p, ms = run_analysis(extracted)
                        st.session_state.last_result   = {"prob":p,"ms":ms,"text":extracted,"ok":True,"from_image":True}
                        st.session_state.download_text = build_download_text(extracted,p,ms)
                    else:
                        st.session_state.last_result = {"ok":False,"ocr_fail":True}
                else:
                    st.session_state.last_result = {"ok":False,"no_image":True}
                st.rerun()

        st.markdown('<div class="col-footer">MindGuard v2 · Bi-LSTM · Mental Health Research</div>', unsafe_allow_html=True)

    # ── Col B — result ─────────────────────────────────────────────────────
    with colB:
        st.markdown("""
        <p class="section-label">🆘 Crisis Helplines — 24/7</p>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.2rem;margin-bottom:0.35rem">
            <div class="support-pill"><strong>🇰🇪 Kenya</strong>📞 Befrienders<br>+254 722 178 177<br>📞 Red Cross: 1199</div>
            <div class="support-pill"><strong>🇺🇸 USA</strong>📞 988 Lifeline<br>988<br>💬 Text HOME → 741741</div>
            <div class="support-pill"><strong>🇬🇧 UK</strong>📞 Samaritans<br>116 123</div>
            <div class="support-pill"><strong>🌍 Global</strong>🔗 <a href="https://findahelpline.com" target="_blank">findahelpline.com</a></div>
        </div>
        <hr class="divider">
        """, unsafe_allow_html=True)

        r = st.session_state.last_result

        if r and not r.get("ok"):
            if r.get("empty"):    st.warning("⚠️ Please enter some text first.")
            elif r.get("no_image"): st.warning("⚠️ Please upload an image first.")
            elif r.get("ocr_fail"): st.warning("⚠️ Could not read text from the image.")

        if r and r.get("ok"):
            prob = r["prob"]
            is_high_risk = prob < 0.5

            label    = "🔴 Suicidal / Negative"  if prob < 0.5 else "🟢 Non-Suicidal / Positive"
            color    = "#f87171"                  if prob < 0.5 else "#5eead4"
            risk_lbl = "HIGH RISK"                if prob < 0.5 else "LOW RISK"
            risk_cls = "risk-high"                if prob < 0.5 else "risk-low"
            conf     = prob if prob >= 0.5 else (1 - prob)

            if conf >= 0.8:   cl,cc = "High Confidence",   "conf-high"
            elif conf >= 0.6: cl,cc = "Medium Confidence", "conf-medium"
            else:             cl,cc = "Low Confidence",    "conf-low"

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:1rem;font-weight:700;color:{color};text-align:center;margin:0 0 0.3rem">{label}</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(gauge(prob), use_container_width=True)
            st.markdown(f'<p style="font-size:0.76rem;margin:0.12rem 0"><strong>Risk:</strong> <span class="{risk_cls}">{risk_lbl}</span></p>',
                        unsafe_allow_html=True)
            st.progress(int(prob*100) if prob >= 0.5 else int((1-prob)*100))
            st.markdown(f'<div style="text-align:center;margin:0.2rem 0"><span class="conf-badge {cc}">{cl}: {conf:.1%}</span></div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div style="text-align:center;margin:0.2rem 0 0.35rem"><span style="background:rgba(13,148,136,0.25);color:#5eead4;padding:3px 12px;border-radius:999px;font-size:0.68rem;font-weight:600">⚡ {r["ms"]:.1f}ms</span></div>',
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            if is_high_risk:
                st.error("🚨 CRISIS ALERT — High-risk content detected! Please use the helplines above.")

            if st.session_state.download_text:
                st.download_button("📄 Download report", st.session_state.download_text,
                                   file_name="mindguard_report.txt", use_container_width=True)

        st.markdown("""
        <div class="remember-card">
            <strong style="font-size:0.74rem;color:#fff;display:block;margin-bottom:0.1rem">💙 Remember</strong>
            🤝 You are not alone &nbsp;·&nbsp; 🕐 Help is 24/7 &nbsp;·&nbsp; 💬 Talk to someone
        </div>""", unsafe_allow_html=True)

    # ── Col C — analytics ──────────────────────────────────────────────────
    with colC:
        st.markdown('<h3 style="text-align:center;margin:0 0 0.35rem">📊 Session Analytics</h3>',
                    unsafe_allow_html=True)
        a = st.session_state.analytics
        if a["total_analyses"] > 0:
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card"><div class="stat-label">Total</div><div class="stat-number">{a['total_analyses']}</div></div>
                <div class="stat-card"><div class="stat-label">Positive</div><div class="stat-number" style="color:#5eead4">{a['positive_count']}</div></div>
                <div class="stat-card"><div class="stat-label">At-Risk</div><div class="stat-number" style="color:#f87171">{a['negative_count']}</div></div>
            </div>""", unsafe_allow_html=True)

            fig_pie = go.Figure(go.Pie(
                labels=["Non-Suicidal","Suicidal Risk"],
                values=[a["positive_count"],a["negative_count"]],
                marker_colors=["#5eead4","#f87171"],
                hole=0.38, textfont_size=10, textfont_color="white"
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font={"color":"white"}, height=180,
                margin=dict(l=5,r=5,t=8,b=5),
                legend=dict(font=dict(color="white",size=9),orientation="v",x=1.0,y=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.74rem;font-weight:600;margin-bottom:0.15rem">📝 Recent</p>', unsafe_allow_html=True)
            for item in reversed(a["history"][-5:]):
                emoji = "🟢" if item["cls"] == "Positive" else "🔴"
                st.markdown(
                    f'<p style="margin:0.08rem 0;font-size:0.69rem">{emoji} <strong>{item["cls"]}</strong> · {item["ts"]} · {item["prob"]:.0%}<br>'
                    f'<em style="color:rgba(255,255,255,0.55)">{item["txt"]}</em></p>',
                    unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:2rem 0.5rem;color:rgba(255,255,255,0.45)">
                <div style="font-size:2rem;margin-bottom:0.4rem">📊</div>
                <p style="font-size:0.76rem">No analyses yet.<br>Run a scan to see stats.</p>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Reddit Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_reddit:
    import pandas as pd

    rA, rB = st.columns([1, 2])

    with rA:
        st.markdown('<h2>Reddit User Analysis</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.75rem;color:rgba(255,255,255,0.65);">Fetches up to 6 months of posts & comments, runs each through the Bi-LSTM, and shows the full risk timeline.</p>',
            unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        reddit_user = st.text_input("Reddit username (without u/)",
                                    placeholder="e.g.  spez",
                                    key="reddit_username")
        with st.expander("🔑 Reddit API credentials", expanded=True):
            st.markdown('<p style="font-size:0.7rem;">Free at <a href="https://www.reddit.com/prefs/apps" target="_blank">reddit.com/prefs/apps</a> → create a <em>script</em> app.</p>', unsafe_allow_html=True)
            r_id     = st.text_input("Client ID",     value=os.getenv("REDDIT_CLIENT_ID",""),     placeholder="under app name")
            r_secret = st.text_input("Client Secret", value=os.getenv("REDDIT_CLIENT_SECRET",""), type="password")

        min_risk = st.slider("Show posts above risk score", 0.0, 1.0, 0.0, 0.05)
        n_posts  = st.slider("Max posts to display",        5,   50,  20,   5)

        fetch_btn = st.button("🔍 Analyse Reddit User", use_container_width=True, key="reddit_fetch")
        if fetch_btn:
            if not reddit_user.strip():
                st.warning("Enter a username.")
            elif not r_id or not r_secret:
                st.error("Enter your Reddit API credentials.")
            else:
                uname = reddit_user.strip().lstrip("u/")
                with st.spinner(f"Fetching posts for u/{uname} …"):
                    try:
                        raw = fetch_reddit_posts(uname, r_id, r_secret)
                    except RuntimeError as e:
                        st.error(str(e)); raw = []

                if raw:
                    with st.spinner(f"Running Bi-LSTM on {len(raw)} posts …"):
                        texts  = [clean_text(p["text"]) for p in raw]
                        scores = predict_batch(texts)

                    df = pd.DataFrame(raw)
                    df["risk_score"] = scores
                    df["date"]       = pd.to_datetime(df["date"])

                    overall  = float(np.percentile(scores, 85))
                    n_high   = int((scores >= 0.55).sum())
                    signals  = detect_socioeconomic(raw)

                    st.session_state.reddit_results = {
                        "username": uname,
                        "df":       df,
                        "overall":  overall,
                        "n_high":   n_high,
                        "signals":  signals,
                        "n_posts":  len(raw),
                        "min_risk": min_risk,
                        "n_show":   n_posts,
                    }
                    st.rerun()
                elif raw == []:
                    st.warning(f"No posts found for u/{reddit_user.strip()} in the last 6 months.")

    with rB:
        res = st.session_state.reddit_results
        if res is None:
            st.markdown("""
            <div style="text-align:center;padding:4rem 1rem;color:rgba(255,255,255,0.4)">
                <div style="font-size:3rem;margin-bottom:0.5rem">🔴</div>
                <p style="font-size:0.85rem">Enter a username and click <strong style="color:rgba(255,255,255,0.7)">Analyse Reddit User</strong> to begin.</p>
            </div>""", unsafe_allow_html=True)
        else:
            df       = res["df"]
            overall  = res["overall"]
            n_high   = res["n_high"]
            signals  = res["signals"]
            uname    = res["username"]

            lbl, col, cls = risk_label(overall)

            # ── Overall banner ──────────────────────────────────────────
            st.markdown(
                f'<h3>u/{uname} &nbsp;<span style="font-size:0.85rem;color:{col};font-weight:700">{lbl}</span></h3>',
                unsafe_allow_html=True)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Overall Risk", f"{overall:.1%}")
            m2.metric("Posts Analysed", str(res["n_posts"]))
            m3.metric("High-Risk Posts", str(n_high))
            m4.metric("Period", "6 months")

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            # ── Sub-tabs ────────────────────────────────────────────────
            s1, s2, s3 = st.tabs(["📈 Timeline", "📝 Posts", "💼 Socio-Economic"])

            with s1:
                st.plotly_chart(timeline_chart(df), use_container_width=True)
                if df["subreddit"].nunique() > 1:
                    st.plotly_chart(subreddit_chart(df), use_container_width=True)

            with s2:
                filtered = df[df["risk_score"] >= res["min_risk"]].sort_values("risk_score", ascending=False).head(res["n_show"])
                for _, row in filtered.iterrows():
                    score = row["risk_score"]
                    lbl2, col2, cls2 = risk_label(score)
                    preview = row["text"][:260] + ("…" if len(row["text"]) > 260 else "")
                    date_s  = row["date"].strftime("%d %b %Y") if hasattr(row["date"],"strftime") else str(row["date"])
                    st.markdown(f"""
                    <div class="post-card {cls2}">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
                            <span style="color:rgba(255,255,255,0.55);font-size:0.7rem">
                                r/{row['subreddit']} · {row['type']} · {date_s}
                            </span>
                            <span style="color:{col2};font-weight:700;font-size:0.78rem">{score:.1%} — {lbl2}</span>
                        </div>
                        <p style="color:rgba(255,255,255,0.85);margin:0;font-size:0.76rem;line-height:1.5">{preview}</p>
                        <a href="{row['url']}" target="_blank" style="color:#5eead4;font-size:0.7rem;text-decoration:none">View on Reddit ↗</a>
                    </div>""", unsafe_allow_html=True)

            with s3:
                any_found = False
                for cat, kws in signals.items():
                    if kws:
                        any_found = True
                        tags = " ".join(f'<span class="socio-tag">{kw}</span>' for kw in kws)
                        st.markdown(f'<p style="font-size:0.78rem;font-weight:700;color:#fff;margin:0.5rem 0 0.2rem">{cat}</p>{tags}',
                                    unsafe_allow_html=True)
                if not any_found:
                    st.info("No socio-economic distress keywords detected in this user's posts.")

                found_cats = {c:kws for c,kws in signals.items() if kws}
                if found_cats:
                    fig = px.pie(
                        names=list(found_cats.keys()),
                        values=[len(v) for v in found_cats.values()],
                        hole=0.45,
                        color_discrete_sequence=["#0d9488","#7c3aed","#f97316","#f59e0b","#ef4444"],
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="rgba(255,255,255,0.8)",
                                      legend=dict(font_color="rgba(255,255,255,0.7)"),
                                      margin=dict(t=10,b=10,l=10,r=10),height=240)
                    st.plotly_chart(fig, use_container_width=True)

            if overall >= 0.55:
                st.error(f"🚨 CRISIS ALERT — u/{uname} shows {lbl}. Please consider reaching out or directing to crisis resources.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TikTok Video
# ══════════════════════════════════════════════════════════════════════════════
with tab_tiktok:
    tA, tB = st.columns([1, 1.4])

    with tA:
        st.markdown('<h2>TikTok Video Analysis</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.75rem;color:rgba(255,255,255,0.65);">'
            'Paste any public TikTok video URL. MindGuard will download the audio, '
            'transcribe the speech using Whisper, then predict suicidal ideation risk.</p>',
            unsafe_allow_html=True)

        st.markdown("""
        <div class="tiktok-badge">
            🎵 <strong>How it works:</strong>&nbsp;
            URL → yt-dlp download → ffmpeg audio strip → Whisper transcription → Bi-LSTM prediction
        </div>""", unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        tiktok_url = st.text_input(
            "TikTok video URL",
            placeholder="https://www.tiktok.com/@username/video/1234567890",
            key="tiktok_url_input",
        )

        st.markdown('<p style="font-size:0.72rem;color:rgba(255,255,255,0.5);margin-top:-0.2rem">⚠️ Public videos only. First run downloads the Whisper tiny model (~75MB).</p>',
                    unsafe_allow_html=True)

        tt_btn = st.button("🎵 Transcribe & Analyse", use_container_width=True, key="tiktok_analyse")

        if tt_btn:
            url = tiktok_url.strip()
            if not url:
                st.warning("Please paste a TikTok video URL.")
            elif "tiktok.com" not in url and "vm.tiktok" not in url:
                st.warning("That doesn't look like a TikTok URL. Please check and try again.")
            else:
                st.session_state.tiktok_result = None
                progress = st.progress(0)
                status   = st.empty()

                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        status.markdown('<p style="font-size:0.78rem;color:#5eead4">⬇️ Downloading audio…</p>', unsafe_allow_html=True)
                        progress.progress(15)
                        audio_path = download_tiktok_audio(url, tmpdir)

                        status.markdown('<p style="font-size:0.78rem;color:#5eead4">🎙️ Transcribing speech (Whisper tiny)…</p>', unsafe_allow_html=True)
                        progress.progress(50)
                        transcript = transcribe_audio(audio_path)

                    progress.progress(75)
                    if not transcript.strip():
                        status.markdown('<p style="font-size:0.78rem;color:#fbbf24">⚠️ No speech detected — the video may be music-only or silent.</p>', unsafe_allow_html=True)
                        st.session_state.tiktok_result = {"ok": False, "reason": "no_speech", "url": url}
                    else:
                        status.markdown('<p style="font-size:0.78rem;color:#5eead4">🧠 Running Bi-LSTM prediction…</p>', unsafe_allow_html=True)
                        prob, ms = predict_one(clean_text(transcript))
                        # Convert to risk score (same inversion as batch)
                        risk = 1.0 - prob
                        progress.progress(100)
                        status.empty()
                        st.session_state.tiktok_result = {
                            "ok": True, "url": url,
                            "transcript": transcript,
                            "prob": prob, "risk": risk, "ms": ms,
                        }
                        update_analytics(prob, transcript)

                except RuntimeError as e:
                    progress.progress(100)
                    status.empty()
                    st.session_state.tiktok_result = {"ok": False, "reason": "error", "msg": str(e), "url": url}

                st.rerun()

    with tB:
        tt = st.session_state.tiktok_result

        if tt is None:
            st.markdown("""
            <div style="text-align:center;padding:4rem 1rem;color:rgba(255,255,255,0.4)">
                <div style="font-size:3rem;margin-bottom:0.5rem">🎵</div>
                <p style="font-size:0.85rem">Paste a TikTok URL and click <strong style="color:rgba(255,255,255,0.7)">Transcribe & Analyse</strong>.</p>
                <p style="font-size:0.75rem;margin-top:0.5rem;color:rgba(255,255,255,0.3)">Works with any public TikTok video that contains speech.</p>
            </div>""", unsafe_allow_html=True)

        elif not tt.get("ok"):
            if tt.get("reason") == "no_speech":
                st.warning("⚠️ No speech detected in this video. Try a video where someone is talking.")
            else:
                st.error(f"❌ {tt.get('msg','Download or transcription failed.')}")
                st.markdown('<p style="font-size:0.74rem;color:rgba(255,255,255,0.5);">Common causes: private video, region-locked, or URL expired. Try opening the URL in your browser first.</p>', unsafe_allow_html=True)

        else:
            risk  = tt["risk"]
            prob  = tt["prob"]
            lbl, col, cls = risk_label(risk)

            st.markdown('<p class="section-label">📝 Transcript</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:rgba(0,0,0,0.25);border-radius:10px;padding:0.6rem 0.8rem;'
                f'border:1px solid rgba(255,255,255,0.12);font-size:0.78rem;line-height:1.6;'
                f'color:rgba(255,255,255,0.88);max-height:160px;overflow-y:auto;">{tt["transcript"]}</div>',
                unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<p class="section-label">🧠 Prediction Result</p>', unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            r1.metric("Risk Score",  f"{risk:.1%}")
            r2.metric("Risk Level",  lbl)
            r3.metric("Latency",     f"{tt['ms']:.0f}ms")

            st.plotly_chart(gauge(prob), use_container_width=True)

            if risk >= 0.55:
                st.error("🚨 CRISIS ALERT — High-risk content detected in this video. Please direct to crisis resources (Resources tab).")
            elif risk >= 0.35:
                st.warning("⚠️ Moderate risk detected. Consider monitoring and providing support resources.")
            else:
                st.success("✅ Low risk detected in this video's speech content.")

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<p class="section-label">💼 Socio-Economic Signals in Transcript</p>', unsafe_allow_html=True)
            sig_posts = [{"text": tt["transcript"]}]
            signals   = detect_socioeconomic(sig_posts)
            any_sig   = False
            for cat, kws in signals.items():
                if kws:
                    any_sig = True
                    tags = " ".join(f'<span class="socio-tag">{kw}</span>' for kw in kws)
                    st.markdown(f'<p style="font-size:0.76rem;font-weight:700;margin:0.3rem 0 0.15rem;color:#fff">{cat}</p>{tags}',
                                unsafe_allow_html=True)
            if not any_sig:
                st.markdown('<p style="color:rgba(255,255,255,0.45);font-size:0.75rem">No socio-economic distress keywords detected.</p>', unsafe_allow_html=True)

            # Download
            dl = (f"TikTok URL: {tt['url']}\n\n"
                  f"Transcript:\n{tt['transcript']}\n\n"
                  f"Risk Score: {risk:.1%}\nRisk Level: {lbl}\n"
                  f"Latency: {tt['ms']:.0f}ms\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            st.download_button("📄 Download report", dl, file_name="tiktok_report.txt", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Resources & Info
# ══════════════════════════════════════════════════════════════════════════════
with tab_resources:
    rc1, rc2 = st.columns([1, 1])

    with rc1:
        st.markdown('<h2>📍 Crisis Resources by Region</h2>', unsafe_allow_html=True)
        selected_region = st.selectbox("Select your region", list(RESOURCES.keys()), key="res_region")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        for r in RESOURCES[selected_region]:
            st.markdown(f"""
            <div class="resource-card">
                <div style="font-weight:700;color:#fff;font-size:0.82rem">{r['name']}</div>
                <div style="color:rgba(255,255,255,0.5);font-size:0.7rem;margin:2px 0">{r['type']}</div>
                <div style="color:#5eead4;font-weight:600;font-size:0.8rem">{r['contact']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:0.8rem;padding:0.55rem 0.7rem;background:rgba(239,68,68,0.1);
                    border-radius:8px;border:1px solid rgba(239,68,68,0.3)">
            <p style="color:#fca5a5;font-size:0.78rem;margin:0">
            ⚠️ <strong>If someone is in immediate danger, call emergency services immediately.</strong><br>
            MindGuard is a research tool only — it does not replace clinical assessment.
            </p>
        </div>""", unsafe_allow_html=True)

    with rc2:
        st.markdown('<h2>ℹ️ About MindGuard</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.78rem;line-height:1.8;color:rgba(255,255,255,0.8)">

        <p class="section-label">🧠 Model Architecture</p>
        <p>Embedding(20K vocab, 128dim) → SpatialDropout(0.2) →
        <strong>BiLSTM(64)</strong> → Dropout(0.5) →
        <strong>BiLSTM(32)</strong> → Dropout(0.4) → Dense(sigmoid)</p>

        <p class="section-label">📊 Risk Tiers</p>
        <p>🟢 <strong style="color:#22c55e">Low</strong> &lt; 35% &nbsp;·&nbsp;
           🟡 <strong style="color:#f59e0b">Moderate</strong> 35–55% &nbsp;·&nbsp;
           🟠 <strong style="color:#f97316">High</strong> 55–75% &nbsp;·&nbsp;
           🔴 <strong style="color:#ef4444">Critical</strong> &gt; 75%</p>
        <p>Overall Reddit risk = 85th percentile of all post scores (penalises escalation).</p>

        <p class="section-label">📡 Data Sources</p>
        <p>✍️ <strong>Text/Image</strong> — direct input or OCR screenshot<br>
           🔴 <strong>Reddit</strong> — 6 months posts & comments via PRAW (public API)<br>
           🎵 <strong>TikTok</strong> — public video speech via yt-dlp + Whisper ASR</p>

        <p class="section-label">💼 Socio-Economic Signals</p>
        <p>Employment · Housing · Financial · Relationships · Health<br>
        Keyword-matched across all posts to surface upstream risk factors.</p>

        <p class="section-label">🔒 Ethics</p>
        <p>Ethics approval: TUM-SERC MSC/028/2025A<br>
        NACOSTI Application #535883<br>
        For research use only. Consent-first. No data stored between sessions.</p>

        </div>""", unsafe_allow_html=True)
