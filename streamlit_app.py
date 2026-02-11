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
    layout="centered",
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

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Main container - enhanced glassmorphism */
    .main .block-container {
        max-width: 850px;
        padding: 3rem 2.5rem;
        background: rgba(255, 255, 255, 0.12);
        border-radius: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.4);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 2rem;
    }

    /* Headers */
    h1, h2, h3 {
        text-align: center;
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    
    h2 {
        font-size: 2.8rem !important;
        margin: 1.5rem 0 !important;
        letter-spacing: -0.5px;
    }
    
    h3 {
        font-size: 1.5rem !important;
        margin-top: 2rem !important;
    }

    /* Paragraph text */
    p {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* Stat cards for Analytics Dashboard */
    .stat-card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Text area label */
    .stTextArea label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Text area input - dark glass effect */
    .stTextArea textarea {
        background: rgba(0, 0, 0, 0.35) !important;
        color: #ffffff !important;
        border: 2px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 18px !important;
        font-size: 1rem !important;
        padding: 1.2rem !important;
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
        margin: 2.5rem 0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.85rem 3.5rem !important;
        border-radius: 50px !important;
        border: none !important;
        font-size: 1.15rem !important;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.5) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px;
        cursor: pointer;
    }

    .stButton>button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 15px 45px rgba(245, 87, 108, 0.7) !important;
    }

    .stButton>button:active {
        transform: translateY(-2px) !important;
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
        font-size: 1rem;
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
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.8rem 1rem !important;
        backdrop-filter: blur(10px);
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(255, 255, 255, 0.15) !important;
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
        font-size: 0.95rem;
        margin-top: 3.5rem;
        padding-top: 2rem;
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
        font-size: 1.05rem !important;
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
        padding: 2rem;
        margin: 2rem 0;
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
    padding: 0.85rem 3.5rem !important;
    border-radius: 50px !important;
    border: none !important;
    font-size: 1.05rem !important;
    box-shadow: 0 10px 30px rgba(245, 87, 108, 0.5) !important;
    transition: all 0.3s ease !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 15px 45px rgba(245, 87, 108, 0.7) !important;
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
    
    /* Columns */
    .stColumns {
        gap: 1rem !important;
    }

    </style>
""", unsafe_allow_html=True)

# Sample tweets for quick testing - ONLY 2 OPTIONS
SAMPLE_TWEETS = {
    "Positive 😊": "Just got promoted at work! Feeling blessed and grateful for this opportunity.",
    "Negative 😔": "I feel like nobody cares anymore. I am so depressed. What's the point of trying?"
}

# ===== NEW: Initialize Analytics Session State =====
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

# ===== NEW: Function to update analytics =====
def update_analytics(prob, text):
    st.session_state.analytics['total_analyses'] += 1
    
    if prob >= 0.5:
        st.session_state.analytics['positive_count'] += 1
        classification = "Positive"
    else:
        st.session_state.analytics['negative_count'] += 1
        classification = "Negative"
    
    # Add to history (keep last 10)
    st.session_state.analytics['history'].append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'classification': classification,
        'probability': float(prob),
        'text_preview': text[:50] + "..." if len(text) > 50 else text
    })
    
    # Keep only last 10 analyses
    if len(st.session_state.analytics['history']) > 10:
        st.session_state.analytics['history'] = st.session_state.analytics['history'][-10:]

# ===== NEW: Function to create sentiment intensity gauge =====
def create_sentiment_gauge(prob):
    # Determine sentiment direction and intensity
    if prob >= 0.5:
        # Positive sentiment
        intensity = (prob - 0.5) * 2  # Scale 0.5-1.0 to 0-1
        color = "#51cf66"
        sentiment = "Positive"
    else:
        # Negative sentiment  
        intensity = (0.5 - prob) * 2  # Scale 0-0.5 to 1-0
        color = "#ff6b6b"
        sentiment = "Negative"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = intensity * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{sentiment} Sentiment Intensity", 'font': {'color': 'white', 'size': 20}},
        number = {'suffix': "%", 'font': {'color': 'white', 'size': 40}},
        gauge = {
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
        height=300
    )
    
    return fig


# Main app layout
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("## Suicidal Tweet Detector")
    st.markdown("""
    This tool uses an LSTM model to analyze the emotional tone of tweets and detect possible suicidal ideation.  
    _Enter a tweet below to begin._
    """, unsafe_allow_html=True)

    # ===== NEW FEATURE 1: Analytics Dashboard =====
    with st.expander("📊 Analytics Dashboard"):
        if st.session_state.analytics['total_analyses'] > 0:
            # Stats cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">Total Analyses</div>
                    <div class="stat-number">{st.session_state.analytics['total_analyses']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">✅ Positive</div>
                    <div class="stat-number" style="color: #51cf66;">{st.session_state.analytics['positive_count']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">⚠️ Negative</div>
                    <div class="stat-number" style="color: #ff6b6b;">{st.session_state.analytics['negative_count']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Pie chart
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
                title_font_size=20
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent history
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

    # Quick sample tweets section - ONLY 2 BUTTONS
    with st.expander(" Try Sample Tweets"):
        st.markdown("Click a button to test with example tweets:")
        cols = st.columns(2)
        for idx, (label, tweet) in enumerate(SAMPLE_TWEETS.items()):
            with cols[idx]:
                if st.button(label, key=f"sample_{idx}", use_container_width=True):
                    st.session_state.user_input = tweet
                    st.session_state["text_area"] = tweet
                    st.session_state.should_analyze = True
                    st.rerun()

    # Input with session state
    user_input = st.text_area(
        "Type your tweet here:", 
        height=150, 
        placeholder="Enter tweet text to analyze...",
        value=st.session_state.user_input,
        key="text_area"
    )
    
    # Update session state when user types
    st.session_state.user_input = user_input

    # Button row with analyze and clear - SMALLER CLEAR BUTTON
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analyze_button = st.button("🔍 Analyze Tweet", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True, on_click=clear_text)

    
    

    # Analysis logic - triggered by button OR sample tweet
    if analyze_button or st.session_state.should_analyze:
        st.session_state.should_analyze = False  # Reset flag
        
        if st.session_state.user_input.strip() == "":
            st.warning("⚠️ Please enter some text before analyzing.")
        else:
            with st.spinner("🤖 Analyzing tweet…"):
                start_time = time.time()

                sequence = tokenizer.texts_to_sequences([st.session_state.user_input])
                padded = pad_sequences(sequence, maxlen=100)
                prob = model.predict(padded, verbose=0)[0][0]

                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000
                
                # ===== NEW: Update analytics =====
                update_analytics(prob, st.session_state.user_input)

            # Determine result and confidence level
            if prob < 0.5:
                label = " Suicidal / Negative"
                emoji = "🔴"
                color = "#ff6b6b"
                risk_level = "HIGH RISK"
            else:
                label = " Non-Suicidal / Positive"
                emoji = "🟢"
                color = "#51cf66"
                risk_level = "LOW RISK"
            
            # Confidence level
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

            # ===== NEW FEATURE 7: Crisis Alert System =====
            if prob < 0.5:
                st.error("🚨 **CRISIS ALERT**: High-risk content detected!")
                st.markdown("### 🆘 Immediate Support Available")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    **🇰🇪 Kenya**  
                    📞 1199  
                    📞 +254 722 178 177
                    """)
                with col2:
                    st.markdown("""
                    **🇺🇸 US**  
                    📞 988  
                    💬 Text HOME to 741741
                    """)
                with col3:
                    st.markdown("""
                    **🇬🇧 UK**  
                    📞 116 123  
                    (Samaritans)
                    """)

            # Display results
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            st.markdown(f'<div style="text-align: center; font-size: 4rem; margin: 1rem 0;">{emoji}</div>', unsafe_allow_html=True)
            st.markdown(f'<h2 style="color: {color}; text-align: center; margin: 1rem 0;">{label}</h2>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ===== NEW FEATURE 2: Sentiment Intensity Meter =====
            st.markdown("### 📊 Sentiment Intensity")
            gauge_fig = create_sentiment_gauge(prob)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            st.markdown("---")
            
            # Risk level indicator
            st.markdown(f"**Risk Assessment:** {risk_level}")
            st.progress(int(prob * 100) if prob >= 0.5 else int((1 - prob) * 100))
            
            # Confidence badge
            st.markdown(
                f'<div style="text-align: center; margin: 1rem 0;"><span class="confidence-badge {conf_class}">{conf_label}: {confidence_pct:.1%}</span></div>',
                unsafe_allow_html=True
            )
            
            # Response time
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 1.5rem;">
                    <div style="
                        display:inline-block;
                        padding:10px 24px;
                        border-radius:999px;
                        background:linear-gradient(135deg,#43e97b 0%,#38f9d7 100%);
                        color:#0b1727;
                        font-size:15px;
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
             # Crisis Resources Section
    with st.expander("🆘 Crisis Resources & Support", expanded=(prob < 0.5)):
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

            
            # ===== EXISTING: Result Summary (Copy / Download) =====
            result_text = f"""Tweet:
            {st.session_state.user_input}
            Prediction: {label.strip()}
            Risk: {risk_level}
            Confidence: {confidence_pct:.1%}
            Latency: {elapsed_ms:.1f}ms
            Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
            """
            st.text_area("Result summary (copy this):", result_text, height=180)
            st.download_button("📄 Download Result", result_text, file_name="analysis.txt")
            
            # Crisis resources if high risk
            if prob < 0.5:
                st.info("⚠️ **Important**: This tool is for informational purposes only. If you or someone you know is in crisis, please seek help immediately.")

   
    # Footer
    st.markdown('<div class="footer">Built with ❤️ using Streamlit + LSTM • Mental Health Awareness • Stay safe, stay kind</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
