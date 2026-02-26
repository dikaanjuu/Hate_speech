import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# ===============================
# PATH MODEL
# ===============================
MODEL_DIR = r"D:\hate_speech_streamlit\ultra_indobert_hate_model_3.12"

# ===============================
# LOAD MODEL & TOKENIZER
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# ===============================
# PREPROCESSING
# ===============================
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words_id = set(stopwords.words('indonesian'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words_id]
    return " ".join(words)

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Hate Speech Detector", page_icon="⚠")
st.title("⚠ Hate Speech Detection App")
st.write("Masukkan teks untuk mendeteksi apakah termasuk Hate Speech atau Neutral.")

user_input = st.text_area("Masukkan teks di sini")

if st.button("Deteksi"):
    if user_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        cleaned = clean_text(user_input)
        # tokenisasi
        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # prediksi
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()

        # label mapping (sesuaikan dengan training)
        label_mapping = {0: "Neutral", 1: "Hate Speech"}
        st.success(f"Hasil: {label_mapping[pred]}" if pred==0 else f"Hasil: {label_mapping[pred]}")