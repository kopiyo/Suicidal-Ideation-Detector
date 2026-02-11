import streamlit as st
import pickle
import numpy as np
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Suicidal Tweet Detector",
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

    /* Button - gradient and hover effect */
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

    /* Spinner */
    .stSpinner > div {
        border-top-color: #f5576c !important;
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
        st.info("Make sure 'lstm_model.h5' and 'tokenizer.pkl' are in your repository.")
        st.stop()

model, tokenizer = load_model_and_tokenizer()

# Main app layout
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("## Suicidal Tweet Detector")
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
                    margin-top:20px;
                    display:inline-block;
                    padding:10px 24px;
                    border-radius:999px;
                    background:linear-gradient(135deg,#43e97b 0%,#38f9d7 100%);
                    color:#0b1727;
                    font-size:15px;
                    font-weight:600;
                    box-shadow:0 6px 20px rgba(67,233,123,0.5);
                ">
                    ⚡ Response time: {elapsed_ms:.1f} ms
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown('<div class="footer">Built with ❤️ using Streamlit + LSTM • Stay safe, stay kind</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
