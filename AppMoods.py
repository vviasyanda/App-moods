import os
import re
import string
import json
import joblib
import pandas as pd
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import streamlit as st
from PIL import Image
import nltk

# Unduh resource NLTK jika diperlukan
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')

# Fungsi Pembersihan Teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r"(@[A-Za-z0-9]+)|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def normalize_text(text, alay_dict):
    words = text.split()
    normalized_words = [alay_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

# Muat Kamus Emoji dan Alay
try:
    emoji_dict_path = os.path.join(os.getcwd(), 'emoji.json')
    with open(emoji_dict_path, 'r', encoding='utf-8') as f:
        emoji_dict = json.load(f)
except FileNotFoundError:
    emoji_dict = {}

alay_dict_path = os.path.join(os.getcwd(), 'kamus_alay.csv')
alay_dict_df = pd.read_csv(alay_dict_path, sep=',')
alay_dict = dict(zip(alay_dict_df['slang'], alay_dict_df['formal']))

# Stemming dan Stopword
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop = stopwords.words('indonesian')

# Fungsi tokenisasi menggunakan regex
def tokenize_with_regex(text):
    pattern = r"\b\w+\b"
    return re.findall(pattern, text)

def preprocess_text(text):
    text = clean_text(text)
    text = normalize_text(text, alay_dict)
    text = ' '.join([word for word in text.split() if word not in stop])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    tokens = tokenize_with_regex(text)  # Tokenisasi menggunakan regex
    return ' '.join(tokens)  # Gabungkan kembali token untuk kompatibilitas dengan TF-IDF

# Lokasi file model dan vectorizer
model_path = os.path.join(os.getcwd(), "mlp_model.pkl")
vectorizer_path = os.path.join(os.getcwd(), "bow_vectorizer.pkl")

# Muat model dan vectorizer
model_sentimen = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Fungsi untuk menganalisis ulasan
def analisis_sentimen(ulasan, rating):
    if not ulasan or ulasan == "Masukkan Ulasan...":
        st.error("Harap masukkan ulasan.")
        return

    # Proses pembersihan teks
    ulasan_cleaned = preprocess_text(ulasan)

    # Prediksi model
    ulasan_vectorized = vectorizer.transform([ulasan_cleaned])
    model_prediksi = model_sentimen.predict(ulasan_vectorized)[0]

    # Penanganan hasil "Netral"
    if model_prediksi == "Netral" and (rating == 3):
        hasil_akhir = "Netral"
    elif model_prediksi == "Positif" and rating >= 4:
        hasil_akhir = "Positif"
    elif model_prediksi == "Negatif" and rating <= 2:
        hasil_akhir = "Negatif"
    else:
        hasil_akhir = model_prediksi

    # Menampilkan hasil dan gambar
    if hasil_akhir == "Positif":
        hasil_gambar = "positif.jpg"
        emotikon_gambar = "positif_emotikon.jpg"
    elif hasil_akhir == "Negatif":
        hasil_gambar = "negatif.jpg"
        emotikon_gambar = "negatif_emotikon.jpg"
    else:
        hasil_gambar = "netral.jpg"
        emotikon_gambar = "netral_emotikon.jpg"

    st.image(hasil_gambar, width=500)
    st.image(emotikon_gambar, width=500)

# Streamlit UI
st.title("Analisis Sentimen Ulasan Shopee")

# Inisialisasi session state untuk ulasan
if "ulasan" not in st.session_state:
    st.session_state.ulasan = ""  # Awal kosong

# Fungsi untuk mengelola input ulasan
def clear_placeholder():
    if st.session_state.ulasan == "Masukkan Ulasan...":
        st.session_state.ulasan = ""

# Input teks ulasan dengan placeholder
ulasan = st.text_area(
    "Masukkan Ulasan Anda:",
    value=st.session_state.ulasan or "Masukkan Ulasan...",
    height=150,
    on_change=clear_placeholder,
)

# Input rating dengan slider
rating = st.slider("Pilih Rating (1-5):", min_value=1, max_value=5, value=3)

# Tombol analisis
if st.button("Analisis Sentimen"):
    analisis_sentimen(ulasan, rating)
