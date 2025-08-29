import os
import io
import cv2
import math
import time
import bcrypt
import base64
import numpy as np
import requests
import pandas as pd
from datetime import datetime
from PIL import Image

import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import psycopg2
from psycopg2.extras import RealDictCursor

import torch
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights


# -----------------------------
# Env / Config
# -----------------------------
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = os.getenv("PGPORT", "5432")
PGDATABASE = os.getenv("PGDATABASE", "kacak_yapi_db")
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "postgres")  # .env içine güvenli şekilde yazın

st.set_page_config(page_title="📡 Kaçak Yapı Tespiti", layout="wide")
st.title("📡 Kaçak Yapı Tespiti Uygulaması")

# -----------------------------
# Yardımcılar
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_db():
    conn = psycopg2.connect(
        host=PGHOST, port=PGPORT, dbname=PGDATABASE, user=PGUSER, password=PGPASSWORD,
        cursor_factory=RealDictCursor
    )
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash BYTEA NOT NULL
        );
    """)
    conn.commit()
    return conn

@st.cache_resource(show_spinner=False)
def load_segmentation_model():
    model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT).eval()
    return model

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

def verify_password(password: str, password_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.tobytes() if hasattr(password_hash, "tobytes") else password_hash)
    except Exception:
        return False

def download_static_satellite(lat, lon, zoom=16, size="640x640") -> bytes:
    if not GOOGLE_MAPS_API_KEY:
        raise RuntimeError("GOOGLE_MAPS_API_KEY tanımlı değil (.env dosyanıza ekleyin).")
    url = (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?center={lat},{lon}&zoom={zoom}&size={size}&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
    )
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"Google Static Maps API hatası: {resp.status_code} - {resp.text[:200]}")
    return resp.content

def read_image_from_bytes(b: bytes):
    arr = np.frombuffer(b, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError("Görüntü okunamadı.")
    return im

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def segment_image_bgr(image_bgr: np.ndarray, model) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    inp = preprocess(rgb).unsqueeze(0)
    with torch.no_grad():
        out = model(inp)['out'][0]
    seg = torch.argmax(out, dim=0).byte().cpu().numpy()  # [H,W] sınıf haritası
    seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
    return seg

def analyze_diff(img_old_bgr, seg_old, img_new_bgr, seg_new):
    diff = cv2.absdiff(seg_old, seg_new)
    total_px = diff.size
    change_px = int(np.sum(diff > 0))
    change_pct = (change_px / total_px) * 100.0

    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_buildings = len(contours)
    building_areas = [cv2.contourArea(c) for c in contours]

    # Görsel (kutu çizimi + diff overlay)
    overlay = cv2.cvtColor(img_new_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    ax.set_title("Kaçak Yapı Tespiti (Kareler)")
    ax.axis("off")

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

    diff_colored = np.zeros_like(img_new_bgr)
    diff_colored[diff > 0] = (255, 0, 0)
    ax.imshow(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB), alpha=0.35)

    return {
        "diff": diff,
        "total_px": total_px,
        "change_px": change_px,
        "change_pct": change_pct,
        "detected_buildings": detected_buildings,
        "building_areas": building_areas,
        "fig_detection": fig
    }

def heatmap_plot(diff):
    fig = px.imshow(diff, color_continuous_scale="Inferno", title="📊 İnteraktif Farklılık Isı Haritası")
    fig.update_layout(width=800, height=600)
    return fig


# -----------------------------
# Oturum / Giriş
# -----------------------------
conn = get_db()
cur = conn.cursor()

if "auth" not in st.session_state:
    st.session_state.auth = False
if "username" not in st.session_state:
    st.session_state.username = ""

with st.sidebar:
    st.header("🔐 Oturum")
    if not st.session_state.auth:
        tab_login, tab_register = st.tabs(["Giriş Yap", "Kayıt Ol"])
        with tab_login:
            u = st.text_input("Kullanıcı adı")
            p = st.text_input("Şifre", type="password")
            if st.button("Giriş"):
                cur.execute("SELECT username, password_hash FROM users WHERE username=%s", (u,))
                row = cur.fetchone()
                if row and verify_password(p, row["password_hash"]):
                    st.session_state.auth = True
                    st.session_state.username = u
                    st.success("Giriş başarılı!")
                else:
                    st.error("Hatalı kullanıcı adı/şifre.")
        with tab_register:
            ru = st.text_input("Yeni kullanıcı adı")
            rp = st.text_input("Yeni şifre", type="password")
            if st.button("Kayıt Ol"):
                try:
                    cur.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (ru, psycopg2.Binary(hash_password(rp))))
                    conn.commit()
                    st.success("Kayıt başarılı. Giriş yapabilirsiniz.")
                except psycopg2.Error as e:
                    st.error(f"Kayıt başarısız: {e.pgerror or str(e)}")
    else:
        st.write(f"👤 {st.session_state.username}")
        if st.button("Çıkış Yap"):
            st.session_state.auth = False
            st.session_state.username = ""
            st.rerun()

if not st.session_state.auth:
    st.stop()


# -----------------------------
# Uygulama Sekmeleri
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📍 Koordinatlar & Görüntü", "📊 Analiz", "📈 Raporlama"])

with tab1:
    st.subheader("Koordinat seçimi ve görüntü hazırlığı")

    colA, colB, colC = st.columns(3)
    with colA:
        lat = st.number_input("Enlem (lat)", value=38.40903795, format="%.8f")
    with colB:
        lon = st.number_input("Boylam (lon)", value=38.00127436, format="%.8f")
    with colC:
        zoom = st.slider("Zoom", min_value=5, max_value=20, value=16)

    st.caption("İpucu: Aşağıdan eski görüntüyü yükleyin, yeni görüntüyü Google Static Maps ile indirebilir ya da onu da yükleyebilirsiniz.")

    col1, col2 = st.columns(2)
    with col1:
        old_file = st.file_uploader("Eski uydu görüntüsü (PNG/JPG)", type=["png","jpg","jpeg"], key="old_img")
    with col2:
        new_choice = st.radio("Yeni görüntü kaynağı", ["Google Static Maps ile indir", "Dosya yükle"])
        new_bytes = None
        if new_choice == "Google Static Maps ile indir":
            if st.button("Yeni görüntüyü indir"):
                try:
                    new_bytes = download_static_satellite(lat, lon, zoom=zoom)
                    st.session_state["new_img_bytes"] = new_bytes
                    st.success("Yeni görüntü indirildi.")
                except Exception as e:
                    st.error(str(e))
        else:
            new_upload = st.file_uploader("Yeni uydu görüntüsü (PNG/JPG)", type=["png","jpg","jpeg"], key="new_img")
            if new_upload:
                new_bytes = new_upload.read()
                st.session_state["new_img_bytes"] = new_bytes

    # Önizleme
    col3, col4 = st.columns(2)
    with col3:
        if old_file:
            old_preview = Image.open(old_file).convert("RGB")
            st.image(old_preview, caption="Eski Görüntü", use_container_width=True)
    with col4:
        if "new_img_bytes" in st.session_state:
            new_preview = Image.open(io.BytesIO(st.session_state["new_img_bytes"])).convert("RGB")
            st.image(new_preview, caption="Yeni Görüntü", use_container_width=True)

with tab2:
    st.subheader("Fark Analizi ve Görselleştirme")

    run_it = st.button("🔎 Analizi Başlat")
    if run_it:
        try:
            if not old_file:
                st.error("Lütfen eski görüntüyü yükleyin.")
                st.stop()
            if "new_img_bytes" not in st.session_state:
                st.error("Lütfen yeni görüntüyü indirin veya yükleyin.")
                st.stop()

            # BGR olarak oku
            img_old_bgr = cv2.cvtColor(np.array(Image.open(old_file).convert("RGB")), cv2.COLOR_RGB2BGR)
            img_new_bgr = read_image_from_bytes(st.session_state["new_img_bytes"])

            # Model
            model = load_segmentation_model()

            # Segmentasyon
            seg_old = segment_image_bgr(img_old_bgr, model)
            seg_new = segment_image_bgr(img_new_bgr, model)

            # Analiz
            results = analyze_diff(img_old_bgr, seg_old, img_new_bgr, seg_new)

            # Görseller
            c1, c2 = st.columns(2)
            with c1:
                st.image(cv2.cvtColor(img_old_bgr, cv2.COLOR_BGR2RGB), caption="Eski Uydu Görüntüsü", use_container_width=True)
            with c2:
                st.image(cv2.cvtColor(img_new_bgr, cv2.COLOR_BGR2RGB), caption="Yeni Uydu Görüntüsü", use_container_width=True)

            st.pyplot(results["fig_detection"])

            # Isı haritası
            st.plotly_chart(heatmap_plot(results["diff"]), use_container_width=True)

            # Histogram (bina alanları)
            if results["building_areas"]:
                fig_hist, ax_hist = plt.subplots(figsize=(7,5))
                ax_hist.hist(results["building_areas"], bins=20, edgecolor="black")
                ax_hist.set_title("Yapı Alanlarının Dağılımı (piksel)")
                ax_hist.set_xlabel("Alan (px)")
                ax_hist.set_ylabel("Adet")
                st.pyplot(fig_hist)

            # Kenar panel metrikleri
            st.sidebar.header("📌 Analiz Özeti")
            st.sidebar.metric("Toplam Piksel", f"{results['total_px']:,}")
            st.sidebar.metric("Farklı Alan (px)", f"{results['change_px']:,}")
            st.sidebar.metric("Farklılık Oranı", f"%{results['change_pct']:.2f}")
            st.sidebar.metric("Tespit Edilen Bina", f"{results['detected_buildings']}")

            # Rapor sekmesi için sakla
            st.session_state["report"] = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "lat": lat, "lon": lon, "zoom": zoom,
                "total_px": results["total_px"],
                "change_px": results["change_px"],
                "change_pct": results["change_pct"],
                "detected_buildings": results["detected_buildings"]
            }

        except Exception as e:
            st.error(f"Hata: {e}")

with tab3:
    st.subheader("📈 Raporlama")
    if "report" in st.session_state:
        rep = st.session_state["report"]
        colR1, colR2, colR3 = st.columns(3)
        with colR1:
            st.metric("Tarih/Saat", rep["date"])
            st.metric("Konum (lat,lon)", f"{rep['lat']:.6f}, {rep['lon']:.6f}")
        with colR2:
            st.metric("Zoom", rep["zoom"])
            st.metric("Toplam Piksel", f"{rep['total_px']:,}")
        with colR3:
            st.metric("Farklı Alan (px)", f"{rep['change_px']:,}")
            st.metric("Farklılık Oranı", f"%{rep['change_pct']:.2f}")
            st.metric("Tespit Edilen Bina", f"{rep['detected_buildings']}")

        st.info("Bu rapor demonun bir çıktısıdır. Model (FCN-ResNet50) COCO üzerinde ön-eğitimli olduğu için uydu görüntülerinde mükemmel sonuç beklemeyin; kendi verinizle yeniden eğitme önerilir.")
    else:
        st.warning("Önce Analiz sekmesinden bir çalışma başlatın.")
