import streamlit as st
import pickle
import numpy as np
import time
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Suicidal Tweet Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS with animated gradient background
st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* CRITICAL: Hide the success message */
    .stAlert {
        display: none !important;
    }
    
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Main container - enhanced glassmorphism */
    .main .block-container {
        max-width: 100%;
        padding: 1.5rem 2rem;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.4);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Column spacing improvement */
    [data-testid="column"] {
        padding: 0 1rem !important;
    }

    /* First column (left) */
    [data-testid="column"]:nth-of-type(1) {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
    }

    /* Second column (middle) - results */
    [data-testid="column"]:nth-of-type(2) {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        min-height: 400px;
    }

    /* Third column (right) */
    [data-testid="column"]:nth-of-type(3) {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    
    h2 {
        font-size: 2.2rem !important;
        margin: 1rem 0 !important;
        letter-spacing: -0.5px;
        text-align: center;
    }
    
    h3 {
        font-size: 1.3rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        text-align: center;
    }

    /* Paragraph text */
    p {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Result summary glass card */
    .summary-card {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 1.5rem 1.5rem 1.2rem 1.5rem;
        margin: 1.5rem 0 0.5rem 0;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.22);
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }

    /* Make the summary textarea match your theme */
    .summary-card .stTextArea textarea {
        background: rgba(0, 0, 0, 0.30) !important;
        border: 2px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 16px !important;
        color: #fff !important;
    }

    /* Title inside summary card */
    .summary-title {
        color: rgba(255,255,255,0.95);
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 0.8rem;
        text-align: center;
    }

    /* Center the download button in the card */
    .summary-card .stDownloadButton {
        display: flex;
        justify-content: center;
        margin-top: 0.8rem;
    }

    /* Stat cards for Analytics Dashboard */
    .stat-card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.3rem;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.3rem 0;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Text area label */
    .stTextArea label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Text area input - dark glass effect */
    .stTextArea textarea {
        background: rgba(0, 0, 0, 0.35) !important;
        color: #ffffff !important;
        border: 2px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 18px !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        line-height: 1.6;
    }

    .stTextArea textarea:focus {
        border-color: rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 0 25px rgba(255, 255, 255, 0.3) !important;
        background: rgba(0, 0, 0, 0.45) !important;
        outline: none !important;
    }

    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
        font-style: italic;
    }

    /* Primary Button - gradient and hover effect */
    .stButton {
        text-align: center;
        margin: 0.8rem 0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 50px !important;
        border: none !important;
        font-size: 0.9rem !important;
        box-shadow: 0 8px 20px rgba(245, 87, 108, 0.4) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.3px;
        cursor: pointer;
        width: 100%;
        height: 48px;
    }

    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px rgba(245, 87, 108, 0.6) !important;
    }

    .stButton>button:active {
        transform: translateY(-1px) !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        height: 14px;
    }
    
    .stProgress > div > div {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }

    /* Warning styling */
    .stWarning {
        background-color: rgba(255, 193, 7, 0.18) !important;
        color: #fff !important;
        border-left: 5px solid #ffc107 !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
        padding: 1rem 1.2rem !important;
        font-size: 0.95rem;
    }
    
    /* Info/Success boxes */
    .stInfo {
        background-color: rgba(33, 150, 243, 0.18) !important;
        color: #fff !important;
        border-left: 5px solid #2196f3 !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
        padding: 1rem 1.2rem !important;
    }
    
    /* Error boxes - for Crisis Alert */
    .stError {
        background-color: rgba(244, 67, 54, 0.25) !important;
        color: #fff !important;
        border-left: 5px solid #f44336 !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
        padding: 1rem 1.2rem !important;
        font-weight: 600 !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #f5576c !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.7rem 1rem !important;
        backdrop-filter: blur(10px);
        font-size: 0.95rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border-radius: 0 0 12px 12px !important;
        backdrop-filter: blur(10px);
        padding: 1rem !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding-top: 1rem;
        color: rgba(255, 255, 255, 0.75);
        border-top: 1px solid rgba(255, 255, 255, 0.15);
        font-weight: 500;
    }

    /* Markdown text */
    .stMarkdown {
        color: rgba(255, 255, 255, 0.95) !important;
    }

    /* Code blocks (for confidence) */
    code {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        padding: 0.3rem 0.6rem !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Strong/Bold text */
    strong {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Result card */
    .result-card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem;
    }

    /* Make download button match Analyze button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 50px !important;
        border: none !important;
        font-size: 0.9rem !important;
        box-shadow: 0 8px 20px rgba(245, 87, 108, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100%;
        height: 48px;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px rgba(245, 87, 108, 0.6) !important;
    }

    .confidence-high {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.4);
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #ffd89b, #19547b);
        color: white;
        box-shadow: 0 4px 15px rgba(255, 216, 155, 0.4);
    }

    /* Fade-in animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .main-container {
        animation: fadeInUp 0.8s ease-out;
    }

    /* Enhance italic text */
    em {
        color: rgba(255, 255, 255, 0.85) !important;
        font-style: italic;
    }
    
    /* Columns - proper spacing */
    .row-widget.stHorizontal {
        gap: 1.5rem !important;
    }

    /* Crisis alert box enhancement */
    .crisis-phone {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
    }

    </style>
""", unsafe_allow_html=True)

# Sample tweets for quick testing - ONLY 2 OPTIONS
SAMPLE_TWEETS = {
    "Positive 😊": "Just got promoted at work! Feeling blessed and grateful for this opportunity.",
    "Negative 😔": "I feel like nobody cares anymore. I am so depressed. What's the point of trying?"
}

# ===== Initialize Analytics Session State =====
if 'analytics' not in st.session_state:
    st.session_state.analytics = {
        'total_analyses': 0,
        'positive_count': 0,
        'negative_count': 0,
        'history': []
    }

# Initialize session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'should_analyze' not in st.session_state:
    st.session_state.should_analyze = False

# Load model & tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = load_model("lstm_model.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        st.success("✅ Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model or tokenizer: {str(e)}")
        st.info("Make sure 'lstm_model.h5' and 'tokenizer.pkl' are in your repository.")
        st.stop()

model, tokenizer = load_model_and_tokenizer()

# -------- Clear Function --------
def clear_text():
    st.session_state.user_input = ""
    st.session_state["text_area"] = ""
    st.session_state.should_analyze = False

# ===== Function to update analytics =====
def update_analytics(prob, text):
    st.session_state.analytics['total_analyses'] += 1
    
    if prob >= 0.5:
        st.session_state.analytics['positive_count'] += 1
        classification = "Positive"
    else:
        st.session_state.analytics['negative_count'] += 1
        classification = "Negative"
    
    st.session_state.analytics['history'].append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'classification': classification,
        'probability': float(prob),
        'text_preview': text[:50] + "..." if len(text) > 50 else text
    })
    
    if len(st.session_state.analytics['history']) > 10:
        st.session_state.analytics['history'] = st.session_state.analytics['history'][-10:]

# ===== Function to create sentiment intensity gauge =====
def create_sentiment_gauge(prob):
    if prob >= 0.5:
        intensity = (prob - 0.5) * 2
        color = "#51cf66"
        sentiment = "Positive"
    else:
        intensity = (0.5 - prob) * 2
        color = "#ff6b6b"
        sentiment = "Negative"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=intensity * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{sentiment} Sentiment Intensity", 'font': {'color': 'white', 'size': 18}},
        number={'suffix': "%", 'font': {'color': 'white', 'size': 36}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 33], 'color': 'rgba(255,255,255,0.1)'},
                {'range': [33, 66], 'color': 'rgba(255,255,255,0.15)'},
                {'range': [66, 100], 'color': 'rgba(255,255,255,0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=280,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# ─────────────────────────────────────────────
# Main app layout
# ─────────────────────────────────────────────
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # 3 columns: Left (input) | Middle (crisis/support) | Right (analytics + gauge)
    colA, colB, colC = st.columns([1.1, 1.4, 1.1])

    # safe default so right-side panel won't crash before first analysis
    is_high_risk = False

    # Store analysis results in session state so colC can access them
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

    # =========================
    # COLUMN A (Left): Input + Samples + Buttons
    # =========================
    with colA:
        st.markdown("## Suicidal Tweet Detector")
        st.markdown("""
        This tool uses an LSTM model to analyze the emotional tone of tweets and detect possible suicidal ideation.  
        _Enter a tweet below to begin._
        """, unsafe_allow_html=True)

        with st.expander(" Try Sample Tweets"):
            st.markdown("**Click a button to test:**")
            for label, tweet in SAMPLE_TWEETS.items():
                if st.button(label, key=f"sample_{label}", use_container_width=True):
                    st.session_state.user_input = tweet
                    st.session_state["text_area"] = tweet
                    st.session_state.should_analyze = True
                    st.rerun()

        user_input = st.text_area(
            "Type your tweet here:",
            height=160,
            placeholder="Enter tweet text to analyze...",
            value=st.session_state.user_input,
            key="text_area"
        )

        st.session_state.user_input = user_input

        btn1, btn2 = st.columns([1.5, 1])
        with btn1:
            analyze_button = st.button("🔍 Analyze Tweet", use_container_width=True)
        with btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True, on_click=clear_text)

    # =========================
    # Run analysis (before rendering colB & colC so results are available)
    # =========================
    if analyze_button or st.session_state.should_analyze:
        st.session_state.should_analyze = False

        if st.session_state.user_input.strip() != "":
            sequence = tokenizer.texts_to_sequences([st.session_state.user_input])
            padded = pad_sequences(sequence, maxlen=100)
            start_time = time.time()
            prob = model.predict(padded, verbose=0)[0][0]
            elapsed_ms = (time.time() - start_time) * 1000

            update_analytics(prob, st.session_state.user_input)

            st.session_state.last_result = {
                'prob': prob,
                'elapsed_ms': elapsed_ms,
                'text': st.session_state.user_input,
                'analyzed': True,
                'empty': False
            }
        else:
            st.session_state.last_result = {'analyzed': True, 'empty': True}

    # =========================
    # COLUMN B (Middle): 🆘 Immediate Support + Crisis Resources & Support
    # =========================
    with colB:
        result = st.session_state.last_result

        # Show empty-input warning in middle column
        if result and result.get('analyzed') and result.get('empty'):
            st.warning("⚠️ Please enter some text before analyzing.")

        # Show crisis alert + immediate support if high-risk
        if result and result.get('analyzed') and not result.get('empty'):
            prob = result['prob']
            is_high_risk = prob < 0.5

            if is_high_risk:
                st.error("🚨 **CRISIS ALERT**: High-risk content detected!")
                st.markdown("### 🆘 Immediate Support Available")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown("""
                    <div class="crisis-phone">
                    <strong>🇰🇪 Kenya</strong><br>
                    📞 1199<br>
                    📞 +254 722 178 177
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown("""
                    <div class="crisis-phone">
                    <strong>🇺🇸 US</strong><br>
                    📞 988<br>
                    💬 Text HOME to 741741
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown("""
                    <div class="crisis-phone">
                    <strong>🇬🇧 UK</strong><br>
                    📞 116 123<br>
                    (Samaritans)
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    st.markdown("""
                    <div class="crisis-phone">
                    <strong>🌍 International</strong><br>
                    🔗 <a href="https://findahelpline.com" target="_blank" style="color: #f093fb;">findahelpline.com</a><br>
                    (IASP Directory)
                    </div>
                    """, unsafe_allow_html=True)

                st.info("⚠️ **Important**: This tool is for informational purposes only. If you or someone you know is in crisis, please seek help immediately.")

        # Crisis resources expander — always visible in colB, auto-expands if high risk
        with st.expander("🆘 Crisis Resources & Support", expanded=is_high_risk):
            st.markdown("""
            ### If you need immediate help:

            **🇰🇪 Kenya:**
            - **Kenya Red Cross:** 1199
            - **Befrienders Kenya:** +254 722 178 177
            - **Lifeline Kenya:** +254 20 272 1806
            
            **🇺🇸 United States:**
            - **National Suicide Prevention Lifeline:** 988
            - **Crisis Text Line:** Text HOME to 741741
            
            **🇬🇧 United Kingdom:**
            - **Samaritans:** 116 123
            
            **🌍 International:**
            - **International Association for Suicide Prevention:** [findahelpline.com](https://findahelpline.com)
            
            ### Remember:
            - You are not alone
            - Help is available 24/7
            - Speaking to someone can make a difference
            """)

    # =========================
    # COLUMN C (Right): Analytics Dashboard + Sentiment Gauge + Result Card
    # =========================
    with colC:
        # Analytics Dashboard expander
        with st.expander("📊 Analytics Dashboard", expanded=False):
            if st.session_state.analytics['total_analyses'] > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">Total</div>
                        <div class="stat-number">{st.session_state.analytics['total_analyses']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">✅ Pos</div>
                        <div class="stat-number" style="color: #51cf66;">{st.session_state.analytics['positive_count']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">⚠️ Neg</div>
                        <div class="stat-number" style="color: #ff6b6b;">{st.session_state.analytics['negative_count']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                fig = px.pie(
                    values=[st.session_state.analytics['positive_count'],
                            st.session_state.analytics['negative_count']],
                    names=['Positive', 'Negative'],
                    title='Classification Distribution',
                    color_discrete_sequence=['#51cf66', '#ff6b6b']
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    title_font_size=16,
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📝 Recent Analyses")
                for item in reversed(st.session_state.analytics['history'][-5:]):
                    emoji = "🟢" if item['classification'] == "Positive" else "🔴"
                    st.markdown(f"""
                    **{emoji} {item['classification']}** - {item['timestamp']}  
                    _{item['text_preview']}_  
                    Confidence: {item['probability']:.1%}
                    """)
                    st.markdown("---")
            else:
                st.info("📊 No analyses yet. Run your first analysis to see statistics!")

        # Sentiment Gauge + Result Card — shown after analysis
        result = st.session_state.last_result
        if result and result.get('analyzed') and not result.get('empty'):
            prob = result['prob']
            elapsed_ms = result['elapsed_ms']

            if prob < 0.5:
                label = "🔴 Suicidal / Negative"
                color = "#ff6b6b"
                risk_level = "HIGH RISK"
            else:
                label = "🟢 Non-Suicidal / Positive"
                color = "#51cf66"
                risk_level = "LOW RISK"

            confidence_pct = prob if prob >= 0.5 else (1 - prob)
            if confidence_pct >= 0.8:
                conf_label = "High Confidence"
                conf_class = "confidence-high"
            elif confidence_pct >= 0.6:
                conf_label = "Medium Confidence"
                conf_class = "confidence-medium"
            else:
                conf_label = "Low Confidence"
                conf_class = "confidence-low"

            st.markdown('<div class="result-card">', unsafe_allow_html=True)

            st.markdown(
                f'<h2 style="color: {color}; text-align: center; margin: 1rem 0;">{label}</h2>',
                unsafe_allow_html=True
            )

            st.markdown("---")

            st.markdown("### 📊 Sentiment Intensity")
            gauge_fig = create_sentiment_gauge(prob)
            st.plotly_chart(gauge_fig, use_container_width=True)

            st.markdown("---")

            st.markdown(f"**Risk Assessment:** {risk_level}")
            st.progress(int(prob * 100) if prob >= 0.5 else int((1 - prob) * 100))

            st.markdown(
                f'<div style="text-align: center; margin: 1rem 0;"><span class="confidence-badge {conf_class}">{conf_label}: {confidence_pct:.1%}</span></div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 1.5rem;">
                    <div style="
                        display:inline-block;
                        padding:10px 24px;
                        border-radius:999px;
                        background:linear-gradient(135deg,#43e97b 0%,#38f9d7 100%);
                        color:#0b1727;
                        font-size:14px;
                        font-weight:600;
                        box-shadow:0 6px 20px rgba(67,233,123,0.5);
                    ">
                        ⚡ Analyzed in {elapsed_ms:.1f}ms
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown('</div>', unsafe_allow_html=True)

            result_text = f"""Tweet:
{result['text']}

Prediction: {label.strip()}
Risk: {risk_level}
Confidence: {confidence_pct:.1%}
Latency: {elapsed_ms:.1f}ms
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

            st.text_area("📋 Result summary (copy this):", result_text, height=140)
            st.download_button("📄 Download Result", result_text, file_name="analysis.txt", use_container_width=True)

    # Footer
    st.markdown(
        '<div class="footer">Built with ❤️ using Streamlit + LSTM • Mental Health Awareness • Stay safe, stay kind</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
