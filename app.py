import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
from sklearn.metrics.pairwise import cosine_similarity

# ===========================
# CONFIG
# ===========================
st.set_page_config(layout="wide", page_title="Batik AI")

# ===========================
# HEADER
# ===========================
st.markdown("""

""", unsafe_allow_html=True)

# ===========================
# SESSION
# ===========================
if "history" not in st.session_state:
    st.session_state.history = []

# ===========================
# STYLE
# ===========================
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(150deg, #81d4fa, #0284c7) !important;
}
.title {
    font-size: 30px;
    font-weight:700;
    text-align:center;
}
.card {
    background: rgba(255,255,255,0.9);
    border-radius:15px;
    padding:10px;
    text-align:center;
    transition: 0.3s;
}

/* 🔥 HOVER EFFECT */
.card:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}

.badge {
    background: #16a34a;
    padding:6px 16px;
    border-radius:999px;
    color:white;
}

/* RIWAYAT */
.history-card {
    background: white;
    border-radius: 16px;
    padding: 12px;
    margin-bottom: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.history-label {
    font-weight: 700;
    font-size: 16px;
}
.history-meta {
    font-size: 13px;
    opacity: 0.7;
}
.history-badge {
    background: #16a34a;
    color: white;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    display: inline-block;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# MENU
# ===========================
with st.sidebar:
    menu = st.selectbox("", ["Beranda", "Motif", "Klasifikasi", "Riwayat"])

    st.markdown("### Upload Dataset")
    uploaded_dataset = st.file_uploader(
        "Upload Dataset ZIP",
        type=["zip"]
    )

# ===========================
# MODEL
# ===========================
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

@st.cache_resource
def load_model():
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(14, activation='softmax')
    ])

    model.load_weights("model_efficientnet.keras")

    return model

model = load_model()

# ===========================
# FEATURE EXTRACTOR
# ===========================
@st.cache_resource
def get_feature_extractor():

    dummy = np.zeros((1,224,224,3))
    model.predict(dummy)

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[-3].output
    )

feature_extractor = get_feature_extractor()

# ===========================
# DATABASE
# ===========================
@st.cache_resource
def load_database(uploaded_zip=None):

    import zipfile
    import shutil

    dataset_folder = "dataset_similarity"

    # ===========================
    # UPLOAD DATASET ZIP
    # ===========================
    if uploaded_zip is not None:

        # hapus dataset lama
        if os.path.exists(dataset_folder):
            shutil.rmtree(dataset_folder)

        # simpan zip
        zip_path = "uploaded_dataset.zip"

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        # extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_folder)

    # ===========================
    # VALIDASI
    # ===========================
    if not os.path.exists(dataset_folder):
        st.warning("Silakan upload dataset ZIP terlebih dahulu")
        return np.array([]), [], []

    features = []
    labels = []
    paths = []

    # ===========================
    # LOAD IMAGE
    # ===========================
    for label in os.listdir(dataset_folder):

        folder = os.path.join(dataset_folder, label)

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):

            path = os.path.join(folder, file)

            try:
                img = Image.open(path).convert("RGB").resize((224,224))

                arr = preprocess_input(np.array(img))
                arr = np.expand_dims(arr, axis=0)

                feat = feature_extractor.predict(arr)[0]

                features.append(feat)
                labels.append(label)
                paths.append(path)

            except:
                pass

    return np.array(features), labels, paths

# LOAD DATABASE
db_features, db_labels, db_paths = load_database(uploaded_dataset)

# ===========================
# SIMILARITY
# ===========================
def find_similar(img):

    if len(db_features) == 0:
        return []

    img = img.resize((224,224))

    arr = preprocess_input(np.array(img))
    arr = np.expand_dims(arr, axis=0)

    query_feat = feature_extractor.predict(arr)

    sim = cosine_similarity(query_feat, db_features)[0]

    idx = np.argsort(sim)[-3:][::-1]

    return [(db_labels[i], db_paths[i], sim[i]) for i in idx]

# ===========================
# CLASS
# ===========================
class_names = [
    'barong','celup','cendrawasih','ceplok','dayak','insang',
    'kawung','lontara','mataketeran','megamendung','ondel-ondel',
    'parang','pring','rumah-minang'
]

# ===========================
# DESKRIPSI
# ===========================
deskripsi_motif = {
    "barong":"Motif barong melambangkan kekuatan, keberanian, perlindungan dari kejahatan, dan wibawa.",
    "celup":"Motif celup melambangkan kesatuan, kreativitas, dan keragaman.",
    "cendrawasih":"Motif khas Papua yang melambangkan keanggunan dan kebebasan.",
    "ceplok":"Melambangkan keteraturan hidup dan keseimbangan.",
    "dayak":"Melambangkan hubungan manusia dengan alam dan keberanian.",
    "insang":"Melambangkan napas kehidupan dan dinamika kehidupan.",
    "kawung":"Melambangkan pengendalian diri dan kesucian hati.",
    "lontara":"Melambangkan identitas budaya Bugis Makassar.",
    "mataketeran":"Motif khas Madura yang terinspirasi dari mata burung.",
    "megamendung":"Melambangkan kesabaran dan keteduhan.",
    "ondel-ondel":"Melambangkan perlindungan dan penolak bala.",
    "parang":"Melambangkan perjuangan dan kesinambungan.",
    "pring":"Melambangkan persatuan dan kerukunan.",
    "rumah-minang":"Melambangkan filosofi alam dan budaya Minangkabau."
}

# ===========================
# PREDICT
# ===========================
def predict(img):

    img = img.resize((224,224))

    arr = preprocess_input(np.array(img))
    arr = np.expand_dims(arr, axis=0)

    return model.predict(arr)[0]

# ===========================
# BERANDA
# ===========================
if menu == "Beranda":

    st.markdown(
        "<div class='title'>Sistem Klasifikasi Motif Batik</div>",
        unsafe_allow_html=True
    )

    batik_image_path = os.path.join("assets", "batik.jpg")

    if os.path.exists(batik_image_path):
        st.image(batik_image_path, use_column_width=True)
    else:
        st.warning("Gambar batik tidak ditemukan")

    st.markdown("### Deskripsi Sistem")

    st.info("""
Aplikasi ini dibuat khusus untuk mengklasifikasikan motif batik berdasarkan gambar yang diunggah oleh pengguna.

Model yang digunakan adalah CNN EfficientNetB0.

Terdapat 14 kelas motif batik yang dapat diklasifikasikan.
""")

    st.markdown("### Cara Menggunakan")

    st.success("""
Upload gambar batik → sistem klasifikasi → hasil muncul → otomatis tersimpan di riwayat.
""")

# ===========================
# MOTIF
# ===========================
elif menu == "Motif":

    st.markdown(
        "<div class='title'>Galeri Motif Batik</div>",
        unsafe_allow_html=True
    )

    cols = st.columns(4)

    for i, name in enumerate(class_names):

        with cols[i % 4]:

            st.markdown("<div class='card'>", unsafe_allow_html=True)

            path = os.path.join("assets", name + ".jpg")

            if os.path.exists(path):
                st.image(path, use_column_width=True)
            else:
                st.warning("Tidak ada gambar")

            st.markdown(name.title())

            st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# KLASIFIKASI
# ===========================
elif menu == "Klasifikasi":

    st.markdown(
        "<div class='title'>Klasifikasi Motif Batik</div>",
        unsafe_allow_html=True
    )

    file = st.file_uploader(
        "Upload gambar",
        type=["jpg","png","jpeg"]
    )

    if file:

        img = Image.open(file).convert("RGB")

        col1, col2 = st.columns([1,2])

        with col1:
            st.image(img, use_column_width=True)

        with col2:

            pred = predict(img)

            idx = np.argmax(pred)

            conf = float(pred[idx])

            threshold = 0.6

            if conf >= threshold:

                label = class_names[idx]

                st.markdown(
                    f"<div class='badge'>{label.upper()}</div>",
                    unsafe_allow_html=True
                )

                st.write(f"Confidence: {conf*100:.2f}%")

                st.progress(conf)

                deskripsi = deskripsi_motif.get(
                    label.lower(),
                    "Deskripsi belum tersedia."
                )

                st.markdown(f"""
                <div style='background:#f0fdf4;
                            padding:15px;
                            border-radius:10px;
                            margin-top:10px'>

                <b>Deskripsi:</b><br>{deskripsi}

                </div>
                """, unsafe_allow_html=True)

            else:

                st.warning("Motif tidak dikenali → pakai similarity")

                results = find_similar(img)

                if results:

                    for l, p, s in results:

                        st.image(p, width=150)

                        st.write(f"{l} ({s*100:.2f}%)")

                    label = results[0][0]

                else:
                    label = "Tidak dikenali"

            st.session_state.history.append({
                "Waktu": datetime.now().strftime("%H:%M:%S"),
                "File": file.name,
                "Klasifikasi": label,
                "Confidence": f"{conf*100:.2f}%",
                "Gambar": img.copy()
            })

# ===========================
# RIWAYAT
# ===========================
elif menu == "Riwayat":

    st.markdown(
        "<div class='title'>Riwayat</div>",
        unsafe_allow_html=True
    )

    if st.session_state.history:

        for item in st.session_state.history[::-1]:

            col1, col2 = st.columns([1,4])

            with col1:
                st.image(item["Gambar"], use_column_width=True)

            with col2:

                st.markdown(f"""
                <div class="history-card">

                    <div class="history-label">
                    {item['Klasifikasi'].upper()}
                    </div>

                    <div class="history-meta">
                    File: {item['File']}
                    </div>

                    <div class="history-meta">
                    Waktu: {item['Waktu']}
                    </div>

                    <div class="history-badge">
                    Confidence: {item['Confidence']}
                    </div>

                </div>
                """, unsafe_allow_html=True)

        df = pd.DataFrame([
            {k:v for k,v in item.items() if k != "Gambar"}
            for item in st.session_state.history
        ])

        st.download_button(
            "⬇ Download CSV",
            df.to_csv(index=False),
            "riwayat.csv"
        )

        if st.button("🗑 Hapus Riwayat"):

            st.session_state.history = []

            st.success("Riwayat dihapus")

    else:
        st.info("Belum ada data")

# ===========================
# FOOTER
# ===========================
st.markdown("""
<hr>
<div style='text-align:center; padding:10px'>
    <b>🎓 Sistem Klasifikasi Motif Batik</b><br>
    Menggunakan Deep Learning CNN EfficientNetB0<br><br>

    <span style='font-size:12px; opacity:0.6'>
    © 2026 | Skripsi AI Computer Vision
    </span>
</div>
""", unsafe_allow_html=True)
