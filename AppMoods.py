import os
import re
import string
import json
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tkinter import *
from tkinter import messagebox
import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd

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
    tokens = word_tokenize(text)
    return ' '.join(tokens)  # Gabungkan kembali token untuk kompatibilitas dengan TF-IDF

# Lokasi file model dan vectorizer
model_path = os.path.join(os.getcwd(), "mlp_model.pkl")
vectorizer_path = os.path.join(os.getcwd(), "bow_vectorizer.pkl")

# Muat model dan vectorizer
model_sentimen = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Fungsi untuk menganalisis ulasan
def analisis_sentimen():
    ulasan = text_ulasan.get("1.0", END).strip()
    rating = rating_var.get()

    if not ulasan or ulasan == "Masukkan Ulasan...":
        messagebox.showerror("Error", "Harap masukkan ulasan.")
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

    if hasil_akhir == "Positif":
        hasil_gambar = positif_image
        emotikon_gambar = positif_emotikon
    elif hasil_akhir == "Negatif":
        hasil_gambar = negatif_image
        emotikon_gambar = negatif_emotikon
    else:
        hasil_gambar = netral_image
        emotikon_gambar = netral_emotikon

    label_hasil.config(image=hasil_gambar)
    label_hasil.image = hasil_gambar
    label_emotikon.config(image=emotikon_gambar)
    label_emotikon.image = emotikon_gambar

# Jendela utama
layar = tk.Tk()
layar.title("Analisis Sentimen Ulasan Shopee")
layar.geometry('1000x650')
layar.config(bg="#F4E7CE")

# Latar belakang
gambar_latar_path = os.path.join(os.getcwd(), "background final.png")
image = Image.open(gambar_latar_path)
photo = ImageTk.PhotoImage(image)
label_gambar = Label(layar, image=photo)
label_gambar.image = photo
label_gambar.place(x=0, y=0)

# Placeholder pada Text
def on_focus_in(event):
    if text_ulasan.get("1.0", END).strip() == "Masukkan Ulasan...":
        text_ulasan.delete("1.0", END)
        text_ulasan.config(fg="black")

def on_focus_out(event):
    if not text_ulasan.get("1.0", END).strip():
        text_ulasan.insert("1.0", "Masukkan Ulasan...")
        text_ulasan.config(fg="black")

text_ulasan = Text(layar, wrap=WORD, border=0, width=65, height=6, font=("Century", 13), fg="black", bg="#f8f0ea")
text_ulasan.place(x=165, y=162)
text_ulasan.insert("1.0", "Masukkan Ulasan...")
text_ulasan.bind("<FocusIn>", on_focus_in)
text_ulasan.bind("<FocusOut>", on_focus_out)

# Input rating dengan bintang
rating_label = Label(layar, text="Masukkan Rating:", bg='#ffe0d3', font=("Century", 13))
rating_label.place(x=170, y=305)

# Membuat frame untuk bintang rating
rating_frame = Frame(layar, bg="#ffe0d3")
rating_frame.place(x=170, y=340)

stars = []
for i in range(5):
    star_image_path = os.path.join(os.getcwd(), "bintang kosong.jpg")  # Gambar bintang kosong
    star_image_filled_path = os.path.join(os.getcwd(), "bintang.png")  # Gambar bintang penuh

    star_empty = ImageTk.PhotoImage(Image.open(star_image_path).resize((30, 30), Image.Resampling.LANCZOS))
    star_filled = ImageTk.PhotoImage(Image.open(star_image_filled_path).resize((30, 30), Image.Resampling.LANCZOS))

    star_label = Label(rating_frame, image=star_empty, bg="#ffe0d3", border=0)
    star_label.image_empty = star_empty
    star_label.image_filled = star_filled
    star_label.grid(row=0, column=i, padx=5)
    stars.append(star_label)

# Fungsi untuk memperbarui bintang berdasarkan nilai slider
def update_stars(value):
    for i in range(5):
        if i < int(value):
            stars[i].config(image=stars[i].image_filled)
        else:
            stars[i].config(image=stars[i].image_empty)

# Slider untuk memilih rating
rating_var = IntVar(value=3)
rating_slider = Scale(
    layar, from_=1, to=5, orient=HORIZONTAL, variable=rating_var, length=200, 
    bg="#F4E7CE", highlightthickness=0, command=update_stars
)
rating_slider.place(x=170, y=380)

# Inisialisasi bintang pertama kali
update_stars(rating_var.get())

# Tombol analisis
tombol_gambar_path = os.path.join(os.getcwd(), "tombol search.jpg")
search_image = Image.open(tombol_gambar_path).resize((60, 60), Image.Resampling.LANCZOS)
search_photo = ImageTk.PhotoImage(search_image)
tombol_analisis = Button(layar, image=search_photo, bg="#F4E7CE", border=0, command=analisis_sentimen)
tombol_analisis.place(x=780, y=170)

# Gambar hasil analisis
positif_image = ImageTk.PhotoImage(Image.open(os.path.join(os.getcwd(), "positif.jpg")).resize((180, 90), Image.Resampling.LANCZOS))
negatif_image = ImageTk.PhotoImage(Image.open(os.path.join(os.getcwd(), "negatif.jpg")).resize((180, 90), Image.Resampling.LANCZOS))
netral_image = ImageTk.PhotoImage(Image.open(os.path.join(os.getcwd(), "netral.jpg")).resize((180, 90), Image.Resampling.LANCZOS))
positif_emotikon = ImageTk.PhotoImage(Image.open(os.path.join(os.getcwd(), "positif_emotikon.jpg")).resize((250, 150), Image.Resampling.LANCZOS))
negatif_emotikon = ImageTk.PhotoImage(Image.open(os.path.join(os.getcwd(), "negatif_emotikon.jpg")).resize((250, 150), Image.Resampling.LANCZOS))
netral_emotikon = ImageTk.PhotoImage(Image.open(os.path.join(os.getcwd(), "netral_emotikon.jpg")).resize((250, 150), Image.Resampling.LANCZOS))

# Label hasil
label_hasil = Label(layar, bg="#F4E7CE", border=0)
label_hasil.place(x=210, y=480)

label_emotikon = Label(layar, bg="#F4E7CE", border=0)
label_emotikon.place(x=560, y=380)

# Jalankan aplikasi
layar.mainloop()
