import os
import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
from skimage.feature import local_binary_pattern
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(
    page_title="Interior Design Visual Identity Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

IMAGE_SIZE = (224, 224)

# ── Modern CSS inspired by the prototype ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background: #F5F4F0 !important;
    color: #1A1A2E;
}

[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
.block-container {
    max-width: 1250px !important;
    margin: auto !important;
    padding-top: 1rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ── Top nav bar ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 40px;
    height: 64px;
    background: #ffffff;
    border-bottom: 1px solid #E8E6E0;
    position: sticky;
    top: 0;
    z-index: 999;
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.navbar-logo {
    width: 34px; height: 34px;
    background: #3730A3;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    color: #fff;
    font-size: 16px;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
}
.navbar-title {
    font-family: 'Sora', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: #1A1A2E;
    line-height: 1.2;
}
.navbar-title span { display: block; font-weight: 400; font-size: 11px; color: #6B7280; }

.nav-links {
    display: flex;
    gap: 6px;
}
.nav-link {
    padding: 6px 16px;
    border-radius: 8px;
    font-size: 13.5px;
    font-weight: 500;
    color: #374151;
    cursor: pointer;
    text-decoration: none;
    transition: background .15s;
    border: none;
    background: transparent;
}
.nav-link:hover { background: #F3F4F6; }
.nav-link.active {
    background: #EEF2FF;
    color: #3730A3;
    font-weight: 600;
}

/* ── Page wrapper ── */
.page-wrap {
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 32px 80px;
}

/* ── Hero section ── */
.hero-section {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 40px;
    align-items: center;
    margin-bottom: 56px;
}
.hero-text {}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #EEF2FF;
    color: #3730A3;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: .04em;
    text-transform: uppercase;
    padding: 5px 13px;
    border-radius: 100px;
    margin-bottom: 20px;
}
.hero-h1 {
    font-family: 'Sora', sans-serif;
    font-size: clamp(28px, 3.5vw, 42px);
    font-weight: 700;
    line-height: 1.18;
    color: #111827;
    margin-bottom: 18px;
}
.hero-h1 em { font-style: normal; color: #3730A3; }
.hero-p {
    font-size: 16px;
    line-height: 1.7;
    color: #6B7280;
    margin-bottom: 30px;
    max-width: 440px;
}
.btn-primary {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #3730A3;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 14.5px;
    font-weight: 600;
    padding: 13px 26px;
    border-radius: 12px;
    cursor: pointer;
    text-decoration: none !important;
    border: none;
    transition: background .2s, transform .15s;
}
.btn-primary:hover { background: #2e2985; transform: translateY(-1px); }

.hero-img-wrap {
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(55,48,163,.15);
    aspect-ratio: 4/3;
    background: #E5E7EB;
}
.hero-img-wrap img { width: 100%; height: 100%; object-fit: cover; }

/* ── Section title ── */
.section-title {
    font-family: 'Sora', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 8px;
}
.section-sub {
    font-size: 14.5px;
    color: #9CA3AF;
    margin-bottom: 28px;
}

/* ── Cards grid ── */
.cards-grid {
    display: grid;
    gap: 18px;
}
.cards-grid-3 { grid-template-columns: repeat(3, 1fr); }
.cards-grid-5 { grid-template-columns: repeat(5, 1fr); }

.card {
    background: #fff;
    border-radius: 18px;
    padding: 28px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,.05);
    border: 1px solid #F3F4F6;
    transition: box-shadow .2s, transform .2s;
}
.card:hover { box-shadow: 0 8px 32px rgba(55,48,163,.10); transform: translateY(-2px); }

.card-icon {
    width: 44px; height: 44px;
    border-radius: 12px;
    background: #EEF2FF;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    margin-bottom: 16px;
}
.card-title {
    font-family: 'Sora', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 8px;
}
.card-body { font-size: 13.5px; color: #6B7280; line-height: 1.6; }

/* ── Steps pipeline ── */
.step-card {
    background: #fff;
    border-radius: 18px;
    padding: 24px 20px;
    border: 1px solid #F3F4F6;
    box-shadow: 0 2px 12px rgba(0,0,0,.04);
    position: relative;
}
.step-num {
    font-family: 'Sora', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 12px;
}
.step-icon {
    width: 48px; height: 48px;
    border-radius: 14px;
    background: #EEF2FF;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    margin-bottom: 14px;
}
.step-title { font-family: 'Sora', sans-serif; font-size: 13.5px; font-weight: 700; color: #111827; margin-bottom: 6px; }
.step-body { font-size: 12.5px; color: #6B7280; line-height: 1.55; }

/* ── Upload zone ── */
.upload-zone {
    background: #fff;
    border-radius: 20px;
    border: 2px dashed #C7D2FE;
    padding: 48px 32px;
    text-align: center;
    transition: border-color .2s, background .2s;
}
.upload-zone:hover { border-color: #3730A3; background: #F5F3FF; }
.upload-icon { font-size: 40px; margin-bottom: 14px; }
.upload-title { font-family: 'Sora', sans-serif; font-size: 17px; font-weight: 700; color: #111827; margin-bottom: 6px; }
.upload-sub { font-size: 13.5px; color: #9CA3AF; }

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%);
    border: 1px solid #C7D2FE;
    border-radius: 20px;
    padding: 36px 28px;
    text-align: center;
}
.result-label { font-size: 13px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #6B7280; margin-bottom: 10px; }
.result-number {
    font-family: 'Sora', sans-serif;
    font-size: 80px;
    font-weight: 800;
    color: #3730A3;
    line-height: 1;
    margin-bottom: 10px;
}
.result-cluster-name { font-family: 'Sora', sans-serif; font-size: 18px; font-weight: 700; color: #3730A3; margin-bottom: 14px; }
.result-desc { font-size: 13.5px; color: #6B7280; line-height: 1.6; max-width: 260px; margin: 0 auto; }

/* ── Info note ── */
.info-note {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: #F0FDF4;
    border: 1px solid #86EFAC;
    border-radius: 14px;
    padding: 18px 20px;
    margin-top: 24px;
}
.info-note-icon { font-size: 18px; flex-shrink: 0; }
.info-note-text { font-size: 13.5px; color: #166534; line-height: 1.6; }
.info-note-text b { font-weight: 700; }

/* ── Hint note ── */
.hint-note {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #EFF6FF;
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 16px;
    font-size: 13px;
    color: #1D4ED8;
}

/* ── Gallery grid ── */
.gallery-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-top: 20px;
}
.gallery-img {
    border-radius: 14px;
    overflow: hidden;
    aspect-ratio: 4/3;
    background: #E5E7EB;
}
.gallery-img img { width: 100%; height: 100%; object-fit: cover; }

/* ── Divider ── */
.divider { height: 1px; background: #E5E7EB; margin: 48px 0; }

/* ── Streamlit overrides ── */
div[data-testid="stFileUploader"] > label { display: none !important; }
div[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
div[data-testid="stFileUploader"] section > div {
    background: #fff !important;
    border: 2px dashed #C7D2FE !important;
    border-radius: 20px !important;
    padding: 44px 32px !important;
    text-align: center !important;
    transition: border-color .2s !important;
}
div[data-testid="stFileUploader"] section > div:hover {
    border-color: #3730A3 !important;
    background: #F5F3FF !important;
}
div[data-testid="stFileUploader"] section > div > div > span { display: none; }

/* Primary button */
.stButton > button {
    background: #3730A3 !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 13px 30px !important;
    border-radius: 12px !important;
    border: none !important;
    cursor: pointer !important;
    transition: background .2s !important;
    width: 100% !important;
}
.stButton > button:hover { background: #2e2985 !important; }

/* Spinner */
.stSpinner > div { border-top-color: #3730A3 !important; }

/* Hide streamlit image captions */
[data-testid="caption"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Page state ─────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "welcome"

def nav(page):
    st.session_state.page = page
    st.rerun()

def back_button(target_page):
    if st.button("← Back"):
        nav(target_page)

page = st.sidebar.radio(
    "Navigation",
    ["welcome", "about", "how", "predict"],
    index=["welcome", "about", "how", "predict"].index(st.session_state.page)
)

st.session_state.page = page

# ── Model loading (unchanged) ──────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca_model.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    return resnet_model, scaler, pca, kmeans

resnet_model, scaler, pca, kmeans = load_assets()

# ── Feature extraction (unchanged) ────────────────────────────────────────────
def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype(np.float32) / 255.0
    return image

def extract_color_histogram(img, bins=(8, 8, 8)):
    img_uint8 = (img * 255).astype(np.uint8)
    hist = cv2.calcHist([img_uint8], [0, 1, 2], None, bins, [0,256,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_lbp_texture(img, P=8, R=1):
    img_uint8 = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=int(lbp.max()+1), range=(0, lbp.max()+1))
    hist = hist.astype("float") / (hist.sum() + 1e-6)
    return hist

def extract_features(img):
    batch = np.expand_dims(img, axis=0)
    batch_preprocessed = preprocess_input(batch * 255.0)
    deep_features = resnet_model.predict(batch_preprocessed, verbose=0)
    color_features = extract_color_histogram(img).reshape(1, -1)
    texture_features = extract_lbp_texture(img).reshape(1, -1)
    hybrid_features = np.concatenate([
        deep_features,
        color_features * 0.5,
        texture_features * 0.5
    ], axis=1)
    return hybrid_features

def show_cluster_images(pred):
    cluster_folder = f"AI_Deployment/cluster_samples/cluster_{pred}"
    if not os.path.exists(cluster_folder):
        st.warning("No cluster images folder found.")
        return
    image_files = [
        f for f in os.listdir(cluster_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]
    if len(image_files) == 0:
        st.warning("No representative images found.")
        return
    st.markdown(f"""
    <div class="section-title" style="margin-top:40px;">Similar Images from Cluster {pred}</div>
    <div class="section-sub">Images sharing the same visual identity as your upload</div>
    """, unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, img_name in enumerate(image_files[:6]):
        img_path = os.path.join(cluster_folder, img_name)
        with cols[idx % 3]:
            st.image(img_path, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WELCOME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "welcome":
    col_text, col_img = st.columns([1.1, 1], gap="large")

    with col_text:
        st.markdown("""
        <div class="hero-badge">✦ Unsupervised Machine Learning</div>
        <h1 class="hero-h1">Discovering <em>Visual Identities</em><br>in Interior Design Images</h1>
        <p class="hero-p">
            This system uses unsupervised machine learning to automatically discover
            and predict visual identity clusters in interior design images — no labels required.
        </p>
        """, unsafe_allow_html=True)
        if st.button("Start Exploring →", key="hero_cta"):
            nav("about")

    with col_img:
        st.image(
            "https://images.unsplash.com/photo-1600210492493-0946911123ea?w=800&q=80",
            use_container_width=True
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title">What makes this system unique?</div>
    <div class="section-sub">Three pillars that power the prediction engine</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    features = [
        ("🧠", "Deep Feature Extraction", "ResNet50 extracts rich semantic features from each image, capturing high-level design patterns."),
        ("🎨", "Color & Texture Analysis", "HSV color histograms and LBP texture descriptors add perceptual richness beyond deep features."),
        ("🔵", "K-Means Clustering", "Dimensionality-reduced features are grouped into cohesive visual identities without any labeling."),
    ]
    for col, (icon, title, body) in zip([c1, c2, c3], features):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="card-icon">{icon}</div>
                <div class="card-title">{title}</div>
                <div class="card-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "about":
    st.markdown("""
    <div class="hero-badge">📌 Project Overview</div>
    <h1 class="hero-h1">About the <em>Project</em></h1>
    <p class="hero-p" style="max-width:680px">
        Interior design images often share similar visual elements and styles. This project
        automatically discovers visual identities by grouping similar images without
        using labeled training data — a fully unsupervised pipeline.
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    cards = [
        ("🎯", "Problem", "Manual image categorization is time-consuming, inconsistent, and highly subjective — it doesn't scale."),
        ("💡", "Our Goal", "Automatically discover visual identities in interior design images using machine learning."),
        ("🤖", "Why Unsupervised?", "No labeled data required. The system learns patterns and groups images entirely on its own."),
    ]
    for col, (icon, title, body) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="card-icon">{icon}</div>
                <div class="card-title">{title}</div>
                <div class="card-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.image(
            "https://images.unsplash.com/photo-1618221195710-dd6b41faaea6?w=800&q=80",
            use_container_width=True
        )
    with right:
        st.markdown("""
        <div style="padding-top: 20px">
        <div class="section-title">The Challenge of Visual Identity</div>
        <p style="font-size:15px; color:#6B7280; line-height:1.75; margin-bottom:20px">
            Interior design encompasses countless aesthetic styles — Scandinavian minimalism,
            Mediterranean warmth, industrial edge, and more. Manually tagging thousands of
            images across these visual identities is impractical.
        </p>
        <p style="font-size:15px; color:#6B7280; line-height:1.75">
            By combining deep learning feature extraction with classical unsupervised clustering,
            this system finds structure in a sea of images — automatically and objectively.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if st.button("Next: How It Works →", key="about_next"):
        nav("how")

    back_button("welcome")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "how":
    st.markdown("""
    <div class="hero-badge">⚙️ System Architecture</div>
    <h1 class="hero-h1">How the System <em>Works</em></h1>
    <p class="hero-p" style="max-width:680px">
        Our system follows a five-stage pipeline from raw image to predicted visual identity.
        Each stage is designed for both accuracy and interpretability.
    </p>
    """, unsafe_allow_html=True)

    steps = [
        ("☁️", "Step 01", "Upload Image",         "You upload any interior design image in JPG, PNG, or WEBP format."),
        ("🔍", "Step 02", "Feature Extraction",    "ResNet50 + color histograms + LBP texture descriptors are combined into a hybrid feature vector."),
        ("📉", "Step 03", "Dimensionality Reduction","PCA reduces the high-dimensional feature space while preserving the most discriminative information."),
        ("🔵", "Step 04", "K-Means Clustering",    "The reduced features are matched to the nearest cluster centroid — your image's visual identity."),
        ("✅", "Step 05", "Prediction & Results",  "The system shows your predicted cluster and displays visually similar reference images."),
    ]

    cols = st.columns(5, gap="small")
    for col, (icon, num, title, body) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div class="step-icon">{icon}</div>
                <div class="step-title">{title}</div>
                <div class="step-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title">Feature Engineering Details</div>
    <div class="section-sub">Three complementary feature types for robust representation</div>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3, gap="medium")
    feats = [
        ("🧠", "ResNet50 Deep Features", "2048-dimensional embedding from a pre-trained ImageNet model. Captures semantic content like room type, furniture style, and architectural patterns."),
        ("🎨", "Color Histogram", "8×8×8 RGB histogram capturing the overall palette and color mood of the design — warm, cool, neutral."),
        ("🌿", "LBP Texture", "Local Binary Patterns encode surface textures — smooth plaster, rough stone, polished wood — at the pixel level."),
    ]
    for col, (icon, title, body) in zip([f1, f2, f3], feats):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="card-icon">{icon}</div>
                <div class="card-title">{title}</div>
                <div class="card-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if st.button("Next: Try It Out →", key="how_next"):
        nav("predict")

    back_button("about")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD & PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "predict":
    st.markdown("""
    <div class="hero-badge">🖼️ Prediction Tool</div>
    <h1 class="hero-h1">Upload an Interior Design <em>Image</em></h1>
    <p class="hero-p" style="max-width:600px">
        Drop any interior design image below to discover which visual identity cluster it belongs to.
    </p>
    """, unsafe_allow_html=True)

    upload_col, gap_col = st.columns([1.3, 1], gap="large")

    with upload_col:
        st.markdown("""
        <div style="margin-bottom:8px">
            <div class="section-title" style="font-size:17px">Choose your image</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed"
        )
        st.markdown("""
        <div class="hint-note">
            ℹ️ Make sure the image clearly shows an interior or interior-related space.
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        img = preprocess_image(uploaded_file)

        st.markdown('<div class="divider" style="margin:32px 0 28px"></div>', unsafe_allow_html=True)

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<div class="section-title" style="font-size:17px;margin-bottom:14px">Uploaded Image</div>', unsafe_allow_html=True)
            st.image(original_image, use_container_width=True)

        with right:
            st.markdown('<div class="section-title" style="font-size:17px;margin-bottom:14px">Predicted Visual Identity</div>', unsafe_allow_html=True)

            if st.button("⚡  Predict Visual Identity", key="predict_btn"):
                with st.spinner("Analyzing image features..."):
                    features = extract_features(img)
                    features_scaled = scaler.transform(features)
                    features_pca = pca.transform(features_scaled)
                    pred = kmeans.predict(features_pca)[0]
                st.session_state.pred = int(pred)

            if "pred" in st.session_state:
                p = st.session_state.pred
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">Predicted Cluster</div>
                    <div class="result-number">{p}</div>
                    <div class="result-cluster-name">Cluster {p}</div>
                    <div class="result-desc">
                        This image is most similar to other images in Visual Identity {p}
                        based on its visual features.
                    </div>
                </div>
                <div class="info-note">
                    <span class="info-note-icon">✅</span>
                    <span class="info-note-text">
                        <b>Why this cluster?</b><br>
                        The uploaded image shares similar visual characteristics with images in Cluster {p},
                        based on deep features, color distribution, and texture patterns.
                    </span>
                </div>
                """, unsafe_allow_html=True)

        if "pred" in st.session_state:
            show_cluster_images(st.session_state.pred)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            act1, _ = st.columns([1, 3], gap="medium")

            with act1:
                if st.button("⬆  Upload Another Image", key="reset_btn"):
                    del st.session_state.pred
                    st.rerun()
    back_button("how")

# ── Footer note ────────────────────────────────────────────────────────────────
st.markdown('</div>', unsafe_allow_html=True)  # close .page-wrap

st.markdown("""
<div style="background:#fff; border-top:1px solid #E5E7EB; padding:20px 40px;
            display:flex; align-items:center; gap:10px; font-size:13px; color:#9CA3AF;">
    <span style="font-size:16px">ℹ️</span>
    This prototype demonstrates how the system interacts with the user from image upload to final prediction.
</div>
""", unsafe_allow_html=True)
