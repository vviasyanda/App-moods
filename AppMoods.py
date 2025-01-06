import os
import re
import string
import json
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import streamlit as st
from PIL import Image
import nltk

# Cek dan download stopwords jika belum tersedia
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')

# Cek dan download tokenizer punkt jika belum tersedia
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
emoji_dict_path = os.path.join(os.getcwd(), 'emoji.json')
with open(emoji_dict_path, 'r', encoding='utf-8') as f:
    emoji_dict = json.load(f)

alay_dict_path = os.path.join(os.getcwd(), 'kamus_alay.csv')
alay_dict_df = pd.read_csv(alay_dict_path, sep=',')
alay_dict = dict(zip(alay_dict_df['slang'], alay_dict_df['formal']))

# Stemming dan Stopword
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop = stopwords.words('indonesian')

def preprocess_text(text):
    text = clean_text(text)
    text = normalize_text(text, alay_dict)
    text = ' '.join([word for word in text.split() if word not in stop])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    tokens = word_tokenize(text)  # Tokenisasi menggunakan nltk
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

# Input teks ulasan
ulasan = st.text_area("Masukkan Ulasan Anda:", "Masukkan Ulasan...", height=150)

# Input rating dengan slider
rating = st.slider("Pilih Rating (1-5):", min_value=1, max_value=5, value=3)

# Tombol analisis
if st.button("Analisis Sentimen"):
    analisis_sentimen(ulasan, rating)

# Gambar hasil analisis
positif_image_path = os.path.join(os.getcwd(), "positif.jpg")
negatif_image_path = os.path.join(os.getcwd(), "negatif.jpg")
netral_image_path = os.path.join(os.getcwd(), "netral.jpg")
positif_emotikon_path = os.path.join(os.getcwd(), "positif_emotikon.jpg")
negatif_emotikon_path = os.path.join(os.getcwd(), "negatif_emotikon.jpg")
netral_emotikon_path = os.path.join(os.getcwd(), "netral_emotikon.jpg")