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

# Custom CSS for glassmorphism + layout
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background: linear-gradient(to right, #141e30, #243b55);
        font-family: 'Segoe UI', sans-serif;
        color: white;
    }

    .main-container {
        max-width: 700px;
        margin: auto;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba( 31, 38, 135, 0.37 );
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    h1, h2, h3 {
        text-align: center;
        color: #ffc107;
    }

    .stTextArea textarea {
        background-color: #1e1e1e !important;
        color: white !important;
        border: 1px solid #555;
        border-radius: 8px;
    }

    .stButton>button {
        background-color: #ff6b6b;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
        margin-top: 15px;
        transition: 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #ff4c4c;
    }

    .footer {
        text-align: center;
        font-size: 13px;
        margin-top: 30px;
        color: #bbb;
    }

    </style>
""", unsafe_allow_html=True)

# Load model & tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Load the model - this works with both Keras 2 and Keras 3
        model = load_model("lstm_model.keras")
        
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
    st.markdown("## Suicidal Tweet Detector")
    st.markdown("""
    This tool uses an LSTM model to analyze the emotional tone of tweets and detect possible suicidal ideation.  
    _Enter a tweet below to begin._
    """, unsafe_allow_html=True)

    # Input
    user_input = st.text_area("Type your tweet here:", height=150)

    # 🔎 Analyze
    if st.button("Analyze Tweet"):
        if user_input.strip() == "":
            st.warning("Please enter some text before analyzing.")
        else:
            with st.spinner("Analyzing tweet…"):
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
                    margin-top:10px;
                    display:inline-block;
                    padding:6px 14px;
                    border-radius:999px;
                    background:linear-gradient(135deg,#00e676,#69f0ae);
                    color:#0b1727;
                    font-size:13px;
                    font-weight:600;
                    box-shadow:0 2px 6px rgba(0,0,0,0.3);
                ">
                    ⚡ Response time: {elapsed_ms:.1f} ms
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown('<div class="footer">Built with Streamlit + LSTM • Stay safe, stay kind</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
