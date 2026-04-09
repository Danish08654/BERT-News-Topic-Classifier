import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Page Config
st.set_page_config(
    page_title="BERT News Classifier",
    page_icon="📰",
    layout="centered"
)

# Custom CSS (UI Styling)

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #ffffff;
    }
    .subtitle {
        text-align: center;
        color: #aaaaaa;
        margin-bottom: 30px;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

#  MODEL PATH 
model_path = r"F:\bert-news-classifier\news_bert_model\news_bert_model"

# Debug check 
if not os.path.exists(model_path):
    st.error(f" Model path not found: {model_path}")
    st.stop()

# Load Model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

labels = ["🌍 World", "⚽ Sports", "💼 Business", "🔬 Sci/Tech"]

# UI Header
st.markdown('<div class="title">📰 News Topic Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by BERT • NLP • Transformers</div>', unsafe_allow_html=True)

# Input Box

headline = st.text_input("Enter a news headline:")

# Prediction
# --------------------------
if st.button("🔍 Analyze News"):
    if headline.strip() == "":
        st.warning(" Please enter a headline")
    else:
        with st.spinner("Analyzing... "):
            inputs = tokenizer(
                headline,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)

            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = labels[pred_idx]
            confidence = probs[0][pred_idx].item()

        
        # Result Display
        st.success(f"###  Prediction: {pred_label}")
        st.info(f"Confidence: {confidence:.2%}")

        st.markdown("---")

        # Probability Bars
        st.subheader(" Class Probabilities")

        for i, label in enumerate(labels):
            st.write(f"{label}")
            st.progress(float(probs[0][i]))

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("Made Danish Zulfiqar by using BERT & Streamlit")