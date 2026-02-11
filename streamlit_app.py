import streamlit as st
import pickle
import numpy as np
import time
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px

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

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stAlert { display: none !important; }

/* ── Full-viewport single-page lock ── */
html, body, [data-testid="stAppViewContainer"] {
    height: 100vh;
    overflow: hidden;
}

/* ── Animated gradient background ── */
.stApp {
    background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    height: 100vh;
    overflow: hidden;
}
@keyframes gradientShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}

/* ── Main block: tight padding, no overflow ── */
.main .block-container {
    max-width: 100% !important;
    padding: 0.6rem 1.2rem !important;
    margin: 0 !important;
    height: 100vh;
    overflow: hidden;
}

/* ── Each column scrolls independently ── */
[data-testid="column"] {
    padding: 0 0.5rem !important;
    height: calc(100vh - 1.4rem);
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.3) transparent;
}
[data-testid="column"]::-webkit-scrollbar { width: 4px; }
[data-testid="column"]::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.3);
    border-radius: 4px;
}

/* ── Glass panel per column ── */
[data-testid="column"]:nth-of-type(1),
[data-testid="column"]:nth-of-type(2),
[data-testid="column"]:nth-of-type(3) {
    background: rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 1rem !important;
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

/* ── Typography ── */
h1, h2, h3 {
    color: #fff !important;
    font-weight: 700 !important;
    text-shadow: 1px 1px 6px rgba(0,0,0,0.3);
}
h2 { font-size: 1.4rem !important; margin: 0.3rem 0 0.5rem !important; text-align: center; line-height: 1.3; }
h3 { font-size: 0.95rem !important; margin: 0.5rem 0 0.3rem !important; text-align: center; }
p, li { color: rgba(255,255,255,0.93) !important; font-size: 0.82rem; line-height: 1.5; margin: 0.1rem 0; }
strong { color: #fff !important; font-weight: 600 !important; }
em     { color: rgba(255,255,255,0.8) !important; font-style: italic; }

.app-title { text-align: center; margin-bottom: 0.4rem; }
.app-title h2 { font-size: 1.25rem !important; margin: 0 !important; }
.app-subtitle {
    font-size: 0.75rem; color: rgba(255,255,255,0.8) !important;
    text-align: center; line-height: 1.4; margin-bottom: 0.6rem;
}

/* ── Text area ── */
.stTextArea label {
    color: #fff !important; font-weight: 600 !important;
    font-size: 0.82rem !important; margin-bottom: 0.2rem !important;
}
.stTextArea textarea {
    background: rgba(0,0,0,0.32) !important; color: #fff !important;
    border: 1.5px solid rgba(255,255,255,0.25) !important;
    border-radius: 14px !important; font-size: 0.82rem !important;
    padding: 0.7rem !important; backdrop-filter: blur(8px);
    transition: all 0.3s ease; line-height: 1.5; resize: none;
}
.stTextArea textarea:focus {
    border-color: rgba(255,255,255,0.5) !important;
    box-shadow: 0 0 18px rgba(255,255,255,0.25) !important;
    background: rgba(0,0,0,0.42) !important; outline: none !important;
}
.stTextArea textarea::placeholder { color: rgba(255,255,255,0.45) !important; font-style: italic; }

/* ── Buttons ── */
.stButton { text-align: center; margin: 0.3rem 0; }
.stButton > button {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    color: white !important; font-weight: 600 !important;
    padding: 0.4rem 1rem !important; border-radius: 50px !important;
    border: none !important; font-size: 0.8rem !important;
    box-shadow: 0 5px 15px rgba(245,87,108,0.4) !important;
    transition: all 0.3s ease !important; width: 100%; height: 40px;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(245,87,108,0.6) !important; }
.stDownloadButton > button {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    color: white !important; font-weight: 600 !important;
    padding: 0.4rem 1rem !important; border-radius: 50px !important;
    border: none !important; font-size: 0.8rem !important;
    box-shadow: 0 5px 15px rgba(245,87,108,0.4) !important;
    width: 100%; height: 40px;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background-color: rgba(255,255,255,0.13) !important; border-radius: 10px !important;
    color: white !important; font-weight: 600 !important;
    padding: 0.45rem 0.8rem !important; font-size: 0.8rem !important;
}
.streamlit-expanderHeader:hover { background-color: rgba(255,255,255,0.2) !important; }
.streamlit-expanderContent {
    background-color: rgba(255,255,255,0.07) !important;
    border-radius: 0 0 10px 10px !important; padding: 0.6rem !important;
}

/* ── Crisis phone card ── */
.crisis-phone {
    background: rgba(255,255,255,0.1); border-radius: 10px;
    padding: 0.55rem 0.65rem; margin: 0.25rem 0;
    border: 1px solid rgba(255,255,255,0.18);
    font-size: 0.76rem; line-height: 1.6;
}
.crisis-phone a { color: #f093fb !important; }

/* ── Stat cards ── */
.stat-card {
    background: rgba(255,255,255,0.13); border-radius: 12px;
    padding: 0.55rem 0.3rem; text-align: center;
    border: 1px solid rgba(255,255,255,0.18); margin: 0.15rem;
}
.stat-number { font-size: 1.35rem; font-weight: 700; color: #fff; margin: 0.1rem 0; }
.stat-label  { font-size: 0.65rem; color: rgba(255,255,255,0.72); text-transform: uppercase; letter-spacing: 0.8px; }

/* ── Result card ── */
.result-card {
    background: rgba(255,255,255,0.13); border-radius: 16px;
    padding: 0.7rem 0.9rem; margin: 0.4rem 0;
    backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.22);
    box-shadow: 0 5px 20px rgba(0,0,0,0.12);
    animation: slideUp 0.4s ease-out;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Confidence badge ── */
.confidence-badge {
    display: inline-block; padding: 0.25rem 0.75rem;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin: 0.25rem;
}
.confidence-high   { background: linear-gradient(135deg,#11998e,#38ef7d); color:#fff; }
.confidence-medium { background: linear-gradient(135deg,#f093fb,#f5576c); color:#fff; }
.confidence-low    { background: linear-gradient(135deg,#ffd89b,#19547b); color:#fff; }

/* ── Progress bar ── */
.stProgress > div > div > div > div { background: linear-gradient(90deg,#f093fb,#f5576c); border-radius: 8px; height: 9px; }
.stProgress > div > div { background-color: rgba(255,255,255,0.18); border-radius: 8px; }

/* ── Alert boxes ── */
.stWarning {
    background-color: rgba(255,193,7,0.16) !important; color:#fff !important;
    border-left: 4px solid #ffc107 !important; border-radius: 10px !important;
    padding: 0.5rem 0.8rem !important; font-size: 0.78rem;
}
.stInfo {
    background-color: rgba(33,150,243,0.16) !important; color:#fff !important;
    border-left: 4px solid #2196f3 !important; border-radius: 10px !important;
    padding: 0.5rem 0.8rem !important; font-size: 0.78rem;
}
.stError {
    background-color: rgba(244,67,54,0.22) !important; color:#fff !important;
    border-left: 4px solid #f44336 !important; border-radius: 10px !important;
    padding: 0.5rem 0.8rem !important; font-weight: 600 !important; font-size: 0.78rem;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #f5576c !important; }

/* ── Footer ── */
.footer {
    text-align: center; font-size: 0.7rem;
    padding: 0.35rem 0 0.1rem;
    color: rgba(255,255,255,0.55);
    border-top: 1px solid rgba(255,255,255,0.12);
    margin-top: 0.5rem;
}

/* ── Plotly ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.main-container { animation: fadeInUp 0.5s ease-out; }
</style>
""", unsafe_allow_html=True)

# ─── Sample tweets ──────────────────────────────────────────────────────────────
SAMPLE_TWEETS = {
    "Positive 😊": "Just got promoted at work! Feeling blessed and grateful for this opportunity.",
    "Negative 😔": "I feel like nobody cares anymore. I am so depressed. What's the point of trying?"
}

# ─── Session state ───────────────────────────────────────────────────────────────
if 'analytics' not in st.session_state:
    st.session_state.analytics = {
        'total_analyses': 0, 'positive_count': 0,
        'negative_count': 0, 'history': []
    }
if 'user_input'     not in st.session_state: st.session_state.user_input     = ""
if 'should_analyze' not in st.session_state: st.session_state.should_analyze = False
if 'last_result'    not in st.session_state: st.session_state.last_result    = None

# ─── Load model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = load_model("lstm_model.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        st.success("✅ Model loaded!")
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ {str(e)}")
        st.stop()

model, tokenizer = load_model_and_tokenizer()

# ─── Helpers ─────────────────────────────────────────────────────────────────────
def clear_text():
    st.session_state.user_input    = ""
    st.session_state["text_area"]  = ""
    st.session_state.should_analyze = False

def update_analytics(prob, text):
    st.session_state.analytics['total_analyses'] += 1
    classification = "Positive" if prob >= 0.5 else "Negative"
    if prob >= 0.5: st.session_state.analytics['positive_count'] += 1
    else:           st.session_state.analytics['negative_count'] += 1
    st.session_state.analytics['history'].append({
        'timestamp':      datetime.now().strftime("%H:%M:%S"),
        'classification': classification,
        'probability':    float(prob),
        'text_preview':   text[:40] + "…" if len(text) > 40 else text
    })
    if len(st.session_state.analytics['history']) > 10:
        st.session_state.analytics['history'] = st.session_state.analytics['history'][-10:]

def run_analysis(text):
    sequence   = tokenizer.texts_to_sequences([text])
    padded     = pad_sequences(sequence, maxlen=100)
    start      = time.time()
    prob       = model.predict(padded, verbose=0)[0][0]
    elapsed_ms = (time.time() - start) * 1000
    update_analytics(prob, text)
    return prob, elapsed_ms

def create_sentiment_gauge(prob):
    if prob >= 0.5:
        intensity = (prob - 0.5) * 2; color = "#51cf66"; sentiment = "Positive"
    else:
        intensity = (0.5 - prob) * 2; color = "#ff6b6b"; sentiment = "Negative"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=intensity * 100,
        domain={'x': [0,1], 'y': [0,1]},
        title={'text': f"{sentiment} Intensity", 'font': {'color':'white','size':12}},
        number={'suffix':"%", 'font':{'color':'white','size':26}},
        gauge={
            'axis': {'range':[None,100],'tickwidth':1,'tickcolor':"white"},
            'bar':  {'color': color},
            'bgcolor': "rgba(255,255,255,0.08)",
            'borderwidth': 1, 'bordercolor': "white",
            'steps': [
                {'range':[0,33],  'color':'rgba(255,255,255,0.07)'},
                {'range':[33,66], 'color':'rgba(255,255,255,0.12)'},
                {'range':[66,100],'color':'rgba(255,255,255,0.17)'}
            ],
            'threshold': {'line':{'color':"white",'width':3},'thickness':0.7,'value':80}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color':'white'}, height=185,
        margin=dict(l=8, r=8, t=30, b=2)
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT  —  single page, 3 columns
# ═══════════════════════════════════════════════════════════════════════════════
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    colA, colB, colC = st.columns([1.0, 1.3, 1.1])
    is_high_risk = False

    # ════════════════════════════════════
    # COL A  —  Input
    # ════════════════════════════════════
    with colA:
        st.markdown('<div class="app-title"><h2>🧠 Suicidal Tweet Detector</h2></div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="app-subtitle">LSTM model · detects possible suicidal ideation in tweets.<br><em>Enter text below to begin.</em></p>',
            unsafe_allow_html=True
        )

        with st.expander("✨ Try Sample Tweets"):
            st.markdown("<strong>Click to load:</strong>", unsafe_allow_html=True)
            for label, tweet in SAMPLE_TWEETS.items():
                if st.button(label, key=f"sample_{label}", use_container_width=True):
                    st.session_state.user_input     = tweet
                    st.session_state["text_area"]   = tweet
                    st.session_state.should_analyze = True
                    st.rerun()

        user_input = st.text_area(
            "📝 Your tweet:",
            height=115,
            placeholder="Enter tweet text to analyze…",
            value=st.session_state.user_input,
            key="text_area"
        )
        st.session_state.user_input = user_input

        btn1, btn2 = st.columns([1.5, 1])
        with btn1:
            analyze_button = st.button("🔍 Analyze", use_container_width=True)
        with btn2:
            st.button("🗑️ Clear", use_container_width=True, on_click=clear_text)

        # Handle analyze button
        if analyze_button:
            if user_input.strip():
                prob, elapsed_ms = run_analysis(user_input)
                st.session_state.last_result = {
                    'prob': prob, 'elapsed_ms': elapsed_ms,
                    'text': user_input, 'analyzed': True, 'empty': False
                }
            else:
                st.session_state.last_result = {'analyzed': True, 'empty': True}
            st.rerun()

        # Handle sample tweet auto-analyze
        if st.session_state.should_analyze and st.session_state.user_input.strip():
            st.session_state.should_analyze = False
            prob, elapsed_ms = run_analysis(st.session_state.user_input)
            st.session_state.last_result = {
                'prob': prob, 'elapsed_ms': elapsed_ms,
                'text': st.session_state.user_input, 'analyzed': True, 'empty': False
            }

        st.markdown(
            '<div class="footer">Built with ❤️ Streamlit + LSTM · Mental Health Awareness</div>',
            unsafe_allow_html=True
        )

    # ════════════════════════════════════
    # COL B  —  Crisis & Support
    # ════════════════════════════════════
    with colB:
        result = st.session_state.last_result

        if result and result.get('analyzed') and result.get('empty'):
            st.warning("⚠️ Please enter some text before analyzing.")

        if result and result.get('analyzed') and not result.get('empty'):
            prob         = result['prob']
            is_high_risk = prob < 0.5

            if is_high_risk:
                st.error("🚨 **CRISIS ALERT** — High-risk content detected!")
                st.markdown("#### 🆘 Immediate Support")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown("""<div class="crisis-phone">
                    <strong>🇰🇪 Kenya</strong><br>
                    📞 1199<br>📞 +254 722 178 177
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown("""<div class="crisis-phone">
                    <strong>🇺🇸 US</strong><br>
                    📞 988<br>💬 HOME → 741741
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown("""<div class="crisis-phone">
                    <strong>🇬🇧 UK</strong><br>
                    📞 116 123<br>(Samaritans)
                    </div>""", unsafe_allow_html=True)
                with c4:
                    st.markdown("""<div class="crisis-phone">
                    <strong>🌍 Intl</strong><br>
                    🔗 <a href="https://findahelpline.com" target="_blank">findahelpline.com</a>
                    </div>""", unsafe_allow_html=True)

                st.info("⚠️ For informational use only. Seek professional help if in crisis.")

        # Always-visible full crisis resources
        with st.expander("🆘 Crisis Resources & Support", expanded=is_high_risk):
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown("""<div class="crisis-phone">
                <strong>🇰🇪 Kenya</strong><br>
                📞 Kenya Red Cross: 1199<br>
                📞 Befrienders: +254 722 178 177<br>
                📞 Lifeline: +254 20 272 1806
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown("""<div class="crisis-phone">
                <strong>🇺🇸 United States</strong><br>
                📞 Lifeline: 988<br>
                💬 Crisis Text: HOME → 741741<br><br>
                <strong>🇬🇧 United Kingdom</strong><br>
                📞 Samaritans: 116 123
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown("""<div class="crisis-phone">
                <strong>🌍 International</strong><br>
                🔗 <a href="https://findahelpline.com" target="_blank">findahelpline.com</a><br><br>
                <strong>Remember:</strong><br>
                • You are not alone<br>
                • Help is available 24/7<br>
                • Talking helps 💙
                </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════
    # COL C  —  Analytics + Results
    # ════════════════════════════════════
    with colC:
        # Analytics Dashboard
        with st.expander("📊 Analytics Dashboard", expanded=False):
            if st.session_state.analytics['total_analyses'] > 0:
                a = st.session_state.analytics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="stat-card"><div class="stat-label">Total</div><div class="stat-number">{a["total_analyses"]}</div></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="stat-card"><div class="stat-label">✅ Pos</div><div class="stat-number" style="color:#51cf66">{a["positive_count"]}</div></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="stat-card"><div class="stat-label">⚠️ Neg</div><div class="stat-number" style="color:#ff6b6b">{a["negative_count"]}</div></div>', unsafe_allow_html=True)

                fig_pie = px.pie(
                    values=[a['positive_count'], a['negative_count']],
                    names=['Positive','Negative'],
                    color_discrete_sequence=['#51cf66','#ff6b6b']
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color':'white'}, height=170,
                    margin=dict(l=5,r=5,t=8,b=5),
                    legend=dict(font=dict(color='white', size=9))
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("<strong style='font-size:0.78rem'>📝 Recent Analyses</strong>", unsafe_allow_html=True)
                for item in reversed(a['history'][-4:]):
                    emoji = "🟢" if item['classification'] == "Positive" else "🔴"
                    st.markdown(
                        f"<p style='margin:0.08rem 0;font-size:0.7rem'>{emoji} <strong>{item['classification']}</strong> · {item['timestamp']} · {item['probability']:.0%}<br><em>{item['text_preview']}</em></p>",
                        unsafe_allow_html=True
                    )
            else:
                st.info("No analyses yet — run your first scan!")

        # Result card + Gauge (shown after analysis)
        result = st.session_state.last_result
        if result and result.get('analyzed') and not result.get('empty'):
            prob       = result['prob']
            elapsed_ms = result['elapsed_ms']

            label      = "🔴 Suicidal / Negative" if prob < 0.5 else "🟢 Non-Suicidal / Positive"
            color      = "#ff6b6b"                 if prob < 0.5 else "#51cf66"
            risk_level = "HIGH RISK"               if prob < 0.5 else "LOW RISK"

            confidence_pct = prob if prob >= 0.5 else (1 - prob)
            if confidence_pct >= 0.8:
                conf_label, conf_class = "High Confidence",   "confidence-high"
            elif confidence_pct >= 0.6:
                conf_label, conf_class = "Medium Confidence", "confidence-medium"
            else:
                conf_label, conf_class = "Low Confidence",    "confidence-low"

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(
                f'<h2 style="color:{color};font-size:1rem!important;margin:0.1rem 0 0.3rem!important;text-align:center">{label}</h2>',
                unsafe_allow_html=True
            )

            gauge_fig = create_sentiment_gauge(prob)
            st.plotly_chart(gauge_fig, use_container_width=True)

            st.markdown(f"<p style='font-size:0.78rem;margin:0.2rem 0'><strong>Risk Assessment:</strong> {risk_level}</p>", unsafe_allow_html=True)
            st.progress(int(prob * 100) if prob >= 0.5 else int((1 - prob) * 100))

            st.markdown(
                f'<div style="text-align:center;margin:0.3rem 0"><span class="confidence-badge {conf_class}">{conf_label}: {confidence_pct:.1%}</span></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="text-align:center;margin-bottom:0.4rem"><div style="display:inline-block;padding:5px 14px;border-radius:999px;background:linear-gradient(135deg,#43e97b,#38f9d7);color:#0b1727;font-size:0.7rem;font-weight:600;box-shadow:0 3px 12px rgba(67,233,123,0.5)">⚡ {elapsed_ms:.1f}ms</div></div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            result_text = (
                f"Tweet:\n{result['text']}\n\n"
                f"Prediction: {label.strip()}\nRisk: {risk_level}\n"
                f"Confidence: {confidence_pct:.1%}\nLatency: {elapsed_ms:.1f}ms\n"
                f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            st.text_area("📋 Copy result:", result_text, height=85)
            st.download_button("📄 Download Result", result_text, file_name="analysis.txt", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
