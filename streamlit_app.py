import streamlit as st
import pickle
import numpy as np
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Suicidal Tweet Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS - ONLY STYLING CHANGES
st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Hide the success message */
    .element-container:has(.stSuccess) {
        display: none;
    }

    /* Main container - enhanced glassmorphism */
    .main .block-container {
        max-width: 800px;
        padding: 3rem 2rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    /* Headers */
    h1, h2, h3 {
        text-align: center;
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h2 {
        font-size: 2.5rem !important;
        margin: 1rem 0 !important;
    }

    /* Text area label */
    .stTextArea label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* Text area input - improved styling */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .stTextArea textarea:focus {
        border-color: rgba(255, 255, 255, 0.6) !important;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.3) !important;
        background: rgba(255, 255, 255, 0.2) !important;
    }

    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }

    /* Button - gradient and hover effect */
    .stButton {
        text-align: center;
        margin: 2rem 0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 3rem !important;
        border-radius: 50px !important;
        border: none !important;
        font-size: 1.1rem !important;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px;
    }

    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 40px rgba(245, 87, 108, 0.6) !important;
    }

    .stButton>button:active {
        transform: translateY(-1px) !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        height: 12px;
    }

    /* Warning styling */
    .stWarning {
        background-color: rgba(255, 193, 7, 0.15) !important;
        color: #fff !important;
        border-left: 4px solid #ffc107 !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
        padding: 1rem !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #f5576c !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 2rem;
        color: rgba(255, 255, 255, 0.7);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Markdown text */
    .stMarkdown {
        color: rgba(255, 255, 255, 0.95) !important;
    }

    /* Code blocks (for confidence) */
    code {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 5px !important;
        font-weight: 600 !important;
    }

    /* Animation for smooth entrance */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .main-container {
        animation: fadeIn 0.6s ease-out;
    }

    </style>
""", unsafe_allow_html=True)

# Load model & tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Load the model - this works with both Keras 2 and Keras 3
        model = load_model("lstm_model.h5")
        
        # Load tokenizer
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        
        st.success("✅ Model and tokenizer loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Error loading model or tokenizer: {str(e)}")
        st.info("Make sure 'lstm_model.keras' and 'tokenizer.pkl' are in your repository.")
        st.stop()

model, tokenizer = load_model_and_tokenizer()

# Main app layout
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("## 🧠 Suicidal Tweet Detector")
    st.markdown("""
    This tool uses an LSTM model to analyze the emotional tone of tweets and detect possible suicidal ideation.  
    _Enter a tweet below to begin._
    """, unsafe_allow_html=True)

    # Input
    user_input = st.text_area("Type your tweet here:", height=150, placeholder="Enter tweet text to analyze...")

    # 🔎 Analyze
    if st.button("🔍 Analyze Tweet"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text before analyzing.")
        else:
            with st.spinner("🤖 Analyzing tweet…"):
                start_time = time.time()

                sequence = tokenizer.texts_to_sequences([user_input])
                padded = pad_sequences(sequence, maxlen=100)
                prob = model.predict(padded, verbose=0)[0][0]

                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000

            label = "🟥 Suicidal / Negative" if prob < 0.5 else "🟩 Non-Suicidal / Positive"

            st.markdown("### Prediction:")
            st.markdown(f"<h2>{label}</h2>", unsafe_allow_html=True)
            st.progress(int(prob * 100))
            st.markdown(f"**Confidence:** `{prob:.2%}`")

            st.markdown(
                f"""
                <div style="
                    margin-top:15px;
                    display:inline-block;
                    padding:8px 20px;
                    border-radius:999px;
                    background:linear-gradient(135deg,#43e97b 0%,#38f9d7 100%);
                    color:#0b1727;
                    font-size:14px;
                    font-weight:600;
                    box-shadow:0 4px 15px rgba(67,233,123,0.4);
                ">
                    ⚡ Response time: {elapsed_ms:.1f} ms
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown('<div class="footer">Built with ❤️ using Streamlit + LSTM • Stay safe, stay kind</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
