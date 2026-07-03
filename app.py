# Cleaned Streamlit app for Pişirme Laboratuvarı
# Notes:
# - st.set_page_config is intentionally called once at the top of this file.
# - Product pages are routed with st.session_state.current_page.

import base64
import hashlib
import os
import glob

import cv2
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

st.set_page_config(
    page_title="Pişirme Laboratuvarı",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================== LIBRARY & SESSION AYARLARI ==========================================
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Sayfa Değiştirme Fonksiyonu
def change_page(page_name):
    st.session_state.current_page = page_name

# Resim Okuma Fonksiyonu (Cache'li)
@st.cache_data(show_spinner=False)  
def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# HTML Resim Ortala Fonksiyonu
def centered_local_img(file_path, width=150, height=100):
    img_b64 = get_img_as_base64(file_path)
    if img_b64:
        img_tag = f'<img src="data:image/png;base64,{img_b64}" width="{width}" height="{height}" style="border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); object-fit: cover;">'
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 15px;">
                {img_tag}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Görsel Yok: {file_path}")

# ==========================================
# 2. AÇILIŞ SAYFASI TASARIMI
# ==========================================
def show_home_page():
    # Sidebar'ı gizle
    st.markdown("""<style>[data-testid="stSidebar"] {display: none;}</style>""", unsafe_allow_html=True)
    
    # --- LOGO ALANI ---
    img_b64 = get_img_as_base64("Lab_Logo.png")
    
    st.markdown("<div style='text-align: center; padding-top: 30px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    
    if img_b64:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{img_b64}" width="440" 
                     style="border-radius: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.15);">
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/1046/1046857.png", width=150)
        
    st.markdown("</div>", unsafe_allow_html=True)

    # --- BAŞLIK ALANI ---
    st.markdown(
        """
        <h1 style='text-align: center; color: #2C3E50; font-family: sans-serif; font-size: 3rem; margin-bottom: 10px;'>
            Pişirme Laboratuvarı
        </h1>
        <h3 style='text-align: center; color: #7F8C8D; font-weight: 300; margin-top: 0;'>
            Performans Analiz Paneli
        </h3>
        <br>
        <div style="text-align: center; max-width: 700px; margin: 0 auto; background-color: #f8f9fa; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef;">
            <p style='color: #555; font-size: 1.1rem; margin: 0; line-height: 1.6;'>
                 Aşağıdaki kategorilerden ürün seçimi yaparak 
                <b>pişme oranı, renk dağılımı ve kalite standartlarını</b> yapay zeka ile analiz edebilirsiniz.
            </p>
        </div>
        <br><br>
        """, 
        unsafe_allow_html=True
    )

    # --- KARTLAR ---
    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        centered_local_img("Patates_Logo.png") 
        st.button("Patates Analizi", use_container_width=True, on_click=change_page, args=("Patates",))
    with c2:
        centered_local_img("Pizza_Logo.png")
        st.button("Pizza Analizi", use_container_width=True, on_click=change_page, args=("Pizza",))
    with c3:
        centered_local_img("Borek_Logo.png")
        st.button("Börek Analizi", use_container_width=True, on_click=change_page, args=("Börek",))
    with c4:
        centered_local_img("Smallcake_Logo.png")
        st.button("Kek Analizi", use_container_width=True, on_click=change_page, args=("Small Cake",))


######################################## ANALIZLER ########################################

def run_potato():
    #st.title("Patates Kızartması Analizi")
    #up_files = st.file_uploader("Görsel yükle", type=["jpg","jpeg","png"], accept_multiple_files=True)kıza
    #if not up_files:
        #st.info("Başlamak için görsel yükleyin."); return
    #for up in up_files:
        # ---------- Renkler ----------
        def hex_to_bgr(hex_code):
            h = hex_code.lstrip("#")
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            return (b, g, r)

        CLASS_COLORS = {"dough": "#FF99FF", "cooked": "#FFFF66", "burnt": "#FF0000"}
        COLORS_BGR = {k: hex_to_bgr(v) for k, v in CLASS_COLORS.items()}

        # ---------- Maske ----------
        def chroma_mask_simple(bgr):
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)
            A0, B0 = A.astype(np.float32) - 128, B.astype(np.float32) - 128
            C = np.sqrt(A0*A0 + B0*B0)
            chroma_mask = (C > 12)
            not_gray = (B > 125)
            not_dark = (L > 110)
            mask = (chroma_mask & not_gray & not_dark).astype(np.uint8) * 255
            return mask, L, A, B

        # ---------- KMeans Sınıflandırma ----------
        def classify_simple(L, A, B, mask, bgr):
            m = (mask > 0)
            cls = np.full(mask.shape, -1, np.int8)
            if not np.any(m):
                return cls

            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            H = hsv[...,0].astype(np.float32)
            S = hsv[...,1].astype(np.float32)
            V = hsv[...,2].astype(np.float32)
            Lf, Af, Bf = L.astype(np.float32), A.astype(np.float32), B.astype(np.float32)

            Hr = (H/180.0) * (2*np.pi)
            H_sin = np.sin(Hr); H_cos = np.cos(Hr)

            X = np.stack([Lf/255.0, Af/255.0, Bf/255.0,
                        S/255.0, V/255.0, H_sin, H_cos], axis=-1)[m].astype(np.float32)

            mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-6
            Xn = (X - mu) / sigma

            K = 4
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-3)
            _, labels, centers = cv2.kmeans(Xn, K, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
            C = centers * sigma + mu  # denormalize

            def metrics(c):
                Lc = c[0]*255; Ac = c[1]*255; Bc = c[2]*255
                Sc = c[3]*255; Vc = c[4]*255
                hs, hc = c[5], c[6]
                Hc = (np.degrees(np.arctan2(hs, hc)) % 360.0) / 2.0  # 0..180
                in_yellow = (15 <= Hc <= 35)
                return Lc, Ac, Bc, Sc, Vc, Hc, in_yellow

            km_to_class, chosen = {}, set()

            # ---- BURNT (daha net koyuluk + sarı dışı) ----
            burnt_scores = []
            for i, c in enumerate(C):
                Lc, Ac, Bc, Sc, Vc, Hc, in_y = metrics(c)
                s = 1.6*(255-Vc) + 1.2*Ac + 0.9*max(0, 160-Bc) + (80 if not in_y else -40)
                burnt_scores.append((s, i))
            burnt_idx = max(burnt_scores)[1]
            km_to_class[burnt_idx] = 2; chosen.add(burnt_idx)

            # ---- RAW (çok parlak + düşük S + sarı bant şart) ----
            raw_scores = []
            for i, c in enumerate(C):
                if i in chosen: continue
                Lc, Ac, Bc, Sc, Vc, Hc, in_y = metrics(c)
                s = 2.1*Vc - 1.8*Sc + (90 if in_y else -120)
                # kapı: çiğ ancak gerçekten parlak ve düşük S ve sarıdaysa
                if (Vc < 220) or (Sc > 90) or (not in_y):
                    s -= 9_999
                raw_scores.append((s, i))
            raw_idx = max(raw_scores)[1]
            km_to_class[raw_idx] = 0; chosen.add(raw_idx)

            # ---- COOKED (sarı bant + orta V,S) ----
            cooked_scores = []
            for i, c in enumerate(C):
                if i in chosen: continue
                Lc, Ac, Bc, Sc, Vc, Hc, in_y = metrics(c)
                s = (100 if in_y else -60) - 1.0*abs(Vc-185) - 0.8*abs(Sc-130) + 0.4*Bc
                cooked_scores.append((s, i))
            if cooked_scores:
                cooked_idx = max(cooked_scores)[1]
                km_to_class[cooked_idx] = 1; chosen.add(cooked_idx)

            # ---- Kalan kümeler için daha sıkı fallback ----
            for i, c in enumerate(C):
                if i in km_to_class: continue
                Lc, Ac, Bc, Sc, Vc, Hc, in_y = metrics(c)
                if (Vc <= 120) or ((Ac >= 155) and (Bc <= 140) and (Vc <= 190)) or ((not in_y) and (Sc >= 100) and (Vc <= 190)):
                    km_to_class[i] = 2  # burnt
                elif (Vc >= 210) and (Sc <= 80) and in_y:
                    km_to_class[i] = 0  # raw
                else:
                    km_to_class[i] = 1  # cooked

            # ---- Piksel etiketleri ----
            cls_vals = labels.flatten()
            rrcc = np.argwhere(m)
            for (r, c), lab in zip(rrcc, cls_vals):
                cls[r, c] = km_to_class[int(lab)]

            # =========================
            #  SON KAPI (pixel-level)
            # =========================
            hsv2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            H2 = hsv2[...,0].astype(np.float32)
            S2 = hsv2[...,1].astype(np.float32)
            V2 = hsv2[...,2].astype(np.float32)
            Hr = (H2/180.0) * (2*np.pi)
            in_yellow = ( (np.degrees(np.arctan2(np.sin(Hr), np.cos(Hr))) % 360.0)/2.0 >= 15 ) & \
                        ( (np.degrees(np.arctan2(np.sin(Hr), np.cos(Hr))) % 360.0)/2.0 <= 35 )

            # ÇİĞ için sıkı kapı: çok parlak + düşük S + sarı bant ZORUNLU
            bad_raw = (cls == 0) & ( (V2 < 215) | (S2 > 90) | (~in_yellow) )
            cls[bad_raw] = 1  # pişmişe çevir

            # YANIK için minimum koyuluk / (sarı dışı + kızarmış) şartı
            #bad_burnt = (cls == 2) & ~( (V2 <= 100) | ((S2 >= 100) & (~in_yellow) & (V2 <= 190)) )
            #cls[bad_burnt] = 1  # pişmişe çevir

            return cls
        # ---------- Heatmap ----------
        def create_heatmap(bgr, image_path):
            mask, L, A, B = chroma_mask_simple(bgr)
            cls = classify_simple(L, A, B, mask, bgr)
            heat_bgr = bgr.copy()
            mo = (mask > 0)
            heat_bgr[(cls == 0) & mo] = COLORS_BGR["dough"]
            heat_bgr[(cls == 1) & mo] = COLORS_BGR["cooked"]
            heat_bgr[(cls == 2) & mo] = COLORS_BGR["burnt"]
            return mask, cls, heat_bgr
        st.markdown(
            """
            <style>
            .block-container h1 {
                margin-top: -80px;   /* başlığın üst boşluğunu azaltır */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        # ---------- Streamlit ----------
        st.title("Patates Kızartması Analizi")

        uploads = st.file_uploader("Görselleri yükle (tek/çoklu)", type=["jpg","jpeg","png"], accept_multiple_files=True)

        if uploads:
            for up in uploads:
                file_bytes = np.frombuffer(up.getvalue(), np.uint8)
                bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                assert bgr is not None, f"Görsel okunamadı: {up.name}"

                mask, cls, heat_bgr = create_heatmap(bgr, up.name)

                c1, c2 = st.columns(2, gap="small")
                with c1:
                    st.subheader("Orijinal")
                    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                with c2:
                    st.subheader("Isı Haritası")
                    st.image(cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

                # ---- Pie chart bu noktada ----
                counts = [
                    int(np.count_nonzero(cls == 0)),
                    int(np.count_nonzero(cls == 1)),
                    int(np.count_nonzero(cls == 2)),
                ]
                total = max(sum(counts), 1)
                perc = [100.0 * c / total for c in counts]

                labels = ["Undercooked", "Cooked", "Overcooked"]
                colors = [CLASS_COLORS["dough"], CLASS_COLORS["cooked"], CLASS_COLORS["burnt"]]

                fig, ax = plt.subplots(figsize=(2, 2), dpi=110)
                ax.pie(counts, labels=labels, colors=colors, startangle=90,
                    counterclock=False, wedgeprops=dict(edgecolor="white", linewidth=1),
                    autopct=lambda p: f"{p:.1f}%" if p > 0 else "")
                ax.axis("equal")
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

                st.markdown(
                    f"**Undercooked:** {perc[0]:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**Cooked:** {perc[1]:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**Overcooked:** {perc[2]:.1f}%"
                )

                st.divider()
        else:
            st.info("Başlamak için görsel/ler yükle.")



def run_pizza():
    # CSS to reduce top margin
    st.markdown(
        """
        <style>
        .block-container h1 { margin-top: -80px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Pizza Analysis")

    # ==========================================
    # 1. UNIVERSAL COLOR CENTERS (Fixed Model)
    # ==========================================
    # These coordinates represent the ideal "center" for each class in LAB color space.
    # Format: [L (Lightness), A (Green-Red), B (Blue-Yellow)]
    UNIVERSAL_CENTERS = np.array([
        [ 30.0,   2.0,   2.0],  # BURNT: Very dark, low saturation.
        [ 65.0,  18.0,  25.0],  # DARK BROWN: Dark but has reddish/orange tint.
        [145.0,  25.0,  50.0],  # BROWN: Golden/Cooked cheese color.
        [215.0,  10.0,  40.0],  # LIGHT BROWN: Pale yellow/white transition.
        [245.0,   1.0,   5.0]   # DOUGH: Raw dough, nearly white/grey.
    ], dtype=np.float32)

    CLASS_NAMES = ["Burnt", "Dark Brown", "Brown", "Light Brown", "Dough"]

    # ==========================================
    # 2. MASKING THRESHOLDS
    # ==========================================
    # -- HSV Thresholds (For Background/Tray Removal) --
    # Pixels darker than this V value are considered background/tray.
    # Increase if the tray is being detected as burnt pizza.
    HSV_V_BLACK_MAX = 40 
    
    # Glare detection: High Value (V) but Low Saturation (S) = White Glare.
    HSV_V_GLARE_MIN = 235
    HSV_S_GLARE_MAX = 30

    # -- LAB Thresholds (For Pizza Inclusion) --
    # L_MIN is kept low (1) to ensure dark burnt edges are included in the initial mask.
    LAB_L_MIN, LAB_L_MAX =   1, 255
    LAB_A_MIN, LAB_A_MAX =   0, 100
    LAB_B_MIN, LAB_B_MAX =   0, 130

    # -- Dough Specific Thresholds --
    DOUGH_L_MIN, DOUGH_L_MAX = 100, 300
    DOUGH_A_MIN, DOUGH_A_MAX = -5,  10
    DOUGH_B_MIN, DOUGH_B_MAX =  0,  60

    # ==========================================
    # COLOR PALETTE
    # ==========================================
    # Hex codes for visualization: Dough -> Burnt
    CUSTOM_HEX = ["#FF99FF", "#FFFFB5", "#FFFF66", "#CCCC00", "#FF0000"] 
    
    def make_custom_cmap():
        cmap = ListedColormap(CUSTOM_HEX, name="pizza5")
        bins = np.linspace(0.0, 1.0, 6)
        norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)
        return cmap, norm, bins

    CMAP5, NORM5, BINS5 = make_custom_cmap()

    # ==========================================
    # 3. MASKING LOGIC (PRUNING METHOD)
    # ==========================================
    def hsv_exclusion_mask(img):
        """Creates a mask of pixels to EXCLUDE (Background & Glare)."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        S, V = hsv[...,1], hsv[...,2]
        return (V <= HSV_V_BLACK_MAX) | ((V >= HSV_V_GLARE_MIN) & (S <= HSV_S_GLARE_MAX))

    def lab_inclusion_mask(img):
        """Creates a mask of pixels to INCLUDE based on broad color ranges."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
        L, A0, B0 = lab[...,0], lab[...,1]-128, lab[...,2]-128
        brown = (L >= LAB_L_MIN) & (L <= LAB_L_MAX) & \
                (A0 >= LAB_A_MIN) & (A0 <= LAB_A_MAX) & \
                (B0 >= LAB_B_MIN) & (B0 <= LAB_B_MAX)
        dough = (L >= DOUGH_L_MIN) & (L <= DOUGH_L_MAX) & \
                (A0 >= DOUGH_A_MIN) & (A0 <= DOUGH_A_MAX) & \
                (B0 >= DOUGH_B_MIN) & (B0 <= DOUGH_B_MAX)
        return (brown | dough)

    def keep_largest_component(mask_u8):
        """Removes small noise blobs, keeping only the largest object (the pizza)."""
        m = (mask_u8 > 0).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1: return m * 255
        # Index 0 is background, start from 1
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return ((lab == k).astype(np.uint8) * 255)

    def build_pizza_mask_pruned(img, keep_only_largest=True):
        """
        Generates the final binary mask using 'Pruning' logic.
        1. Creates a loose mask.
        2. Aggressively erodes (opens) it to sever connections between the pizza and background noise.
        3. Keeps the largest component.
        4. Dilates (closes) it back to restore the original shape.
        """
        # 1. Base Mask
        inc = lab_inclusion_mask(img)
        exc = hsv_exclusion_mask(img)
        m0  = (inc & (~exc)).astype(np.uint8) * 255
        
        # 2. Pruning (Morphological Open)
        # Using a large kernel (9x9) and high iterations (5) to strip away edge noise.
        # Decreasing iterations preserves more edge detail but may keep background noise.
        pruning_kernel = np.ones((1,1), np.uint8) 
        m_pruned = cv2.morphologyEx(m0, cv2.MORPH_OPEN, pruning_kernel, iterations=1)

        # 3. Component Selection
        if keep_only_largest:
            m_main = keep_largest_component(m_pruned)
        else:
            m_main = m_pruned

        # 4. Restoration (Morphological Close)
        # Fills in holes created by the pruning process.
        closing_kernel = np.ones((1,1), np.uint8)
        m_final = cv2.morphologyEx(m_main, cv2.MORPH_CLOSE, closing_kernel, iterations=20)
        
        return m_final
    
    def outline_on(img, mask_u8):
        """Draws a green contour around the detected mask."""
        out = img.copy()
        if (mask_u8 > 0).any():
            cnts, _ = cv2.findContours((mask_u8 > 0).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Thickness 20 for visibility
            cv2.drawContours(out, cnts, -1, (0, 255, 0), 20, lineType=cv2.LINE_AA)
        return out

    # ==========================================
    # 4. ANALYSIS ENGINE (CLASSIFICATION)
    # ==========================================
    def predict_universal_map(img_bgr, mask_u8, centers, class_order):
        H, W = mask_u8.shape
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
        L, A0, B0 = lab[...,0], lab[...,1]-128, lab[...,2]-128
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        S, V = hsv[...,1].astype(np.int16), hsv[...,2].astype(np.int16)
        m = (mask_u8 > 0)

        class_idx = np.full((H,W), -1, dtype=np.int16)
        if not np.any(m):
            counts = {c:0 for c in class_order}; perc = {c:0.0 for c in class_order}; dom = None
            return class_idx, counts, perc, dom

        # --- Step 1: KNN Classification (Euclidean Distance) ---
        X = np.stack([L[m], A0[m], B0[m]], axis=1).astype(np.float32)
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        class_idx[m] = labels

        burnt_i = 0 
        dark_i  = 1 
        brown_i = 2 
        light_i = 3 
        dough_i = 4 

        # --- Step 2: Hard Rules & Logic Corrections ---

        # Rule A: Charcoal/Ash Detection
        # L < 85: Pixels this dark are burnt regardless of color.
        is_charcoal = (L < 85) & m  
        
        # Matte Black Detection: Dark pixels (up to L=115) with very low color (A&B < 20).
        is_ash = (L >= 85) & (L < 115) & (A0 < 20) & (B0 < 20) & m

        # Shiny Burnt Detection: High Value (V) but low Saturation (S).
        is_shiny_burnt = (V < 190) & (S < 40) & (L < 180) & m

        # Apply Burnt overrides
        true_burnt_mask = (is_charcoal | is_ash | is_shiny_burnt)
        class_idx[true_burnt_mask] = burnt_i

        # Rule B: Shadow/Crack Protection
        # If labeled Burnt, but has significant Red/Yellow (A or B > 20), it's just a dark shadow.
        shadow_in_burnt = (class_idx == burnt_i) & ((A0 > 20) | (B0 > 20)) & m
        class_idx[shadow_in_burnt] = dark_i

        # Rule C: Brown Expansion
        # If labeled Light Brown but is darker than L=185, force it to Brown.
        should_be_brown = (class_idx == light_i) & (L < 185) & m
        class_idx[should_be_brown] = brown_i

        # Rule D: Dough Tolerance
        # If labeled Dough but has some color (A>5 or B>25), it's likely Light Brown.
        fake_dough = (class_idx == dough_i) & ((A0 > 5) | (B0 > 25)) & m
        class_idx[fake_dough] = light_i

        # --- Statistics ---
        counts = {c: int(np.count_nonzero(class_idx == i)) for i,c in enumerate(class_order)}
        tot = sum(counts.values())
        perc = {c: (counts[c]/tot if tot>0 else 0.0) for c in class_order}
        dominant = max(class_order, key=lambda c: counts[c]) if tot>0 else None
        
        return class_idx, counts, perc, dominant

    # ==========================================
    # 5. VISUALIZATION HELPERS
    # ==========================================
    def pct_line(perc: dict, order: list[str]) -> str:
        """Returns a single line string of percentages."""
        return " | ".join(f"{k}: {perc[k]*100:.1f}%" for k in order)

    def heatmap_overlay(img_bgr, class_idx, class_order, alpha=0.6, cmap=CMAP5, norm=NORM5):
        """Overlays the analyzed colors onto the original image."""
        n = len(class_order)
        score = np.zeros(class_idx.shape, dtype=np.float32)
        valid = (class_idx >= 0)
        # Map indices to score (0..1) for colormap
        score[valid] = 1.0 - (class_idx[valid].astype(np.float32) / (n-1))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        colored = (sm.to_rgba(score)[...,:3] * 255).astype(np.uint8)
        colored_bgr = colored[..., ::-1]
        out = img_bgr.copy().astype(np.float32)
        out[valid] = (1-alpha)*out[valid] + alpha*colored_bgr[valid]
        return np.clip(out,0,255).astype(np.uint8)

    def show_heatmap_figure(img_bgr, overlay_bgr, cmap=CMAP5, norm=NORM5, bins=BINS5):
        """Creates the Matplotlib figure with original image, heatmap, and colorbar."""
        fig, axes = plt.subplots(1,3, figsize=(14,5), gridspec_kw={"width_ratios":[1,1,0.06]}, dpi=100)
        axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); axes[0].set_title(""); axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)); axes[1].set_title(""); axes[1].axis("off")
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=axes[2])
        tick_pos = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        cbar.set_ticks(tick_pos); cbar.set_ticklabels(["Dough","Light Brown","Brown","Dark Brown","Burnt"])
        plt.tight_layout()
        return fig

    # ==========================================
    # 6. MAIN UI EXECUTION
    # ==========================================
    
    # Sidebar for upload only
    with st.sidebar:
        st.markdown("""<style>section[data-testid="stSidebar"] > div:first-child {padding-top: 0rem; margin-top: -3rem;}</style>""", unsafe_allow_html=True)
        st.sidebar.title("Settings")
        # Regional analysis sliders removed as requested.

    files = st.file_uploader("Upload Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if not files:
        st.info("Please upload images to begin analysis.")
        return

    decoded = []
    for uf in files:
        data = np.frombuffer(uf.getvalue(), np.uint8)
        decoded.append((uf.name, cv2.imdecode(data, cv2.IMREAD_COLOR)))

    grid = st.columns(2)
    
    for i, (name, img) in enumerate(decoded):
        # 1. Generate Mask
        mask = build_pizza_mask_pruned(img)
        
        # 2. Analyze Colors
        class_idx, counts, perc, dominant = predict_universal_map(
            img, mask, UNIVERSAL_CENTERS, CLASS_NAMES
        )
        
        # 3. Create Heatmap
        heat_over = heatmap_overlay(img, class_idx, CLASS_NAMES, alpha=0.6, cmap=CMAP5, norm=NORM5)

        col = grid[i % 2]
        with col:
            st.subheader(name)
            
            # Show original with green contour outline
            base = outline_on(img, mask)
            fig = show_heatmap_figure(base, heat_over, cmap=CMAP5, norm=NORM5, bins=BINS5)
            st.pyplot(fig, clear_figure=True); plt.close(fig)

            # Percentage Line
            st.markdown(
                f"<div style='text-align:center; margin-top:-8px; margin-bottom:10px'>"
                f"<b>Yüzdeler:</b> {pct_line(perc, CLASS_NAMES)}</div>",
                unsafe_allow_html=True
            )

            # Data prep for Pie Chart
            display_order = ["Dough", "Light Brown", "Brown", "Dark Brown", "Burnt"]
            display_colors = ["#FF99FF", "#FFFFB5", "#FFFF66", "#CCCC00", "#FF0000"]
            counts_list = [counts.get(k, 0) for k in display_order]
            total = sum(counts_list) if sum(counts_list) > 0 else 1
            perc_list = [100.0 * c / total for c in counts_list]

            # ==================================================
            # SMART PIE CHART (ÇAKIŞMA ÖNLEYİCİ + TAŞMA KORUMALI)
            # ==================================================
            # 1. Figür oluştur (DPI artırıldı, boyut ayarlandı)
            fig_pie, ax_pie = plt.subplots(figsize=(6, 4), dpi=120)
            
            # Pasta dilimlerini çiz
            wedges, _ = ax_pie.pie(perc_list, colors=display_colors, startangle=90, counterclock=False)
            
            # Etiket kutusu stili
            bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=0.5)
            kw = dict(arrowprops=dict(arrowstyle="-", lw=0.5), bbox=bbox_props, zorder=0, va="center")
            
            # 2. Etiket verilerini topla
            labels_to_draw = []
            for j, p in enumerate(wedges):
                val = perc_list[j]
                
                # Eğer %0 olanları da görmek istiyorsan alttaki satırı sil veya yorum yap:
                if val <= 0.5: continue 
                
                # Açıyı hesapla (Dilimin tam ortası)
                ang = (p.theta2 - p.theta1)/2. + p.theta1
                
                # Koordinatları bul
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                
                # Etiket hangi tarafta? (1: Sağ, -1: Sol)
                side = 1 if x >= 0 else -1
                
                labels_to_draw.append({
                    "text": f"{display_order[j]}\n%{val:.1f}",
                    "x": x,
                    "y": y,
                    "ang": ang,
                    "side": side,
                    "val": val
                })

            # 3. Sağ ve Sol taraftaki etiketleri ayır ve Yüksekliklerine (Y) göre sırala
            # Bu sıralama, yukarıdan aşağıya doğru yerleştirme yapmamızı sağlar.
            right_labels = sorted([l for l in labels_to_draw if l["side"] == 1], key=lambda k: k["y"], reverse=True)
            left_labels  = sorted([l for l in labels_to_draw if l["side"] == -1], key=lambda k: k["y"], reverse=True)

            # 4. Akıllı Yerleştirme Fonksiyonu
            def draw_side_labels(label_group, is_left=False):
                # Başlangıç tavan noktası (Grafiğin en tepesinden biraz yukarı)
                last_y = 1.5 
                min_dist = 0.30 # Etiketler arası minimum dikey mesafe (Çakışmayı önler)

                for lbl in label_group:
                    # İdeal Y pozisyonu (Dilimin kendi hizası)
                    ideal_y = lbl["y"] * 1.15
                    
                    # Eğer ideal pozisyon, bir önceki etikete çok yakınsa, onu aşağı it.
                    if last_y - ideal_y < min_dist:
                        target_y = last_y - min_dist
                    else:
                        target_y = ideal_y

                    # Çok aşağı gitmemesi için taban sınırı koy (-1.5'in altına inmesin)
                    target_y = max(target_y, -1.5)
                    
                    # Bir sonraki etiket için referans noktasını güncelle
                    last_y = target_y

                    # Çizim ayarları
                    align = "right" if is_left else "left"
                    connection_style = f"angle,angleA=0,angleB={lbl['ang']}"
                    kw["arrowprops"].update({"connectionstyle": connection_style})
                    
                    # Etiketi bas
                    ax_pie.annotate(lbl["text"], 
                                    xy=(lbl["x"], lbl["y"]), 
                                    # X ekseninde biraz dışarı aç (1.4 katı), Y ekseninde hesaplanan yere koy
                                    xytext=(1.4 * lbl["side"], target_y),
                                    horizontalalignment=align, 
                                    fontsize=9, **kw)

            # Fonksiyonu her iki taraf için çalıştır
            draw_side_labels(right_labels, is_left=False)
            draw_side_labels(left_labels, is_left=True)

            # Grafiği çiz (bbox_inches='tight' ile kesilmeyi önle)
            st.pyplot(fig_pie, clear_figure=True, use_container_width=False, bbox_inches='tight', pad_inches=0.2)
            plt.close(fig_pie)
            # ==================================================

            
            # Text Summary
            burnt_pct = perc["Burnt"] * 100
            dough_pct = perc["Dough"] * 100
            cooked_pct = (perc["Dark Brown"] + perc["Brown"] + perc["Light Brown"]) * 100
            
            st.markdown(
                f"<div style='text-align:center; margin-top:-10px; margin-bottom:15px'>"
                f" <b>Undercooked:</b> {dough_pct:.1f}%   |   "
                f" <b>Cooked:</b> {cooked_pct:.1f}%   |   "
                f" <b>Overcooked:</b> {burnt_pct:.1f}%"
                f"</div>",
                unsafe_allow_html=True
            )
            st.divider()
                
def run_borek():
    from streamlit_drawable_canvas import st_canvas
    from PIL import Image


    st.markdown(
        """
        <style>
        .block-container h1 { margin-top: -60px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # =========================================================
    # Yardımcılar
    # =========================================================
    def hex_to_bgr(hex_code):
        h = hex_code.lstrip("#")
        return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

    def simple_mask_white_bg(bgr, V_thresh=230, min_area=20000):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Beyaz arka plan dışarıda kalacak şekilde threshold
        m = (hsv[..., 2] < V_thresh).astype(np.uint8) * 255

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]

        mask = np.zeros_like(m)

        if cnts:
            biggest = max(cnts, key=cv2.contourArea)
            cv2.drawContours(mask, [biggest], -1, 255, -1)

        return mask

    def cnt_color_group(flat_bgr_pixels, target_colors):
        total_count = 0
        for c in target_colors:
            total_count += np.count_nonzero(np.all(flat_bgr_pixels == c, axis=1))
        return total_count

    # =========================================================
    # Renk Eşleme
    # =========================================================
    COLOR_MAP = {
        "#FFFEB5": "#818100",
        "#FEFF94": "#666732",
        "#FEFE7A": "#01FEFF",
        "#FFD15C": "#32CDFF",

        "#EDB256": "#FF99FF",
        "#E4A741": "#FF00FF",

        "#C38F49": "#FFFE67",
        "#B89057": "#CDCC01",
        "#996F3C": "#0167CC",
        "#916533": "#0101CC",
        "#845A37": "#01FF01",
        "#6C5033": "#01CC01",

        "#68533E": "#FE0001",
        "#404032": "#C10100"
    }

    SRC_BGR = np.array([hex_to_bgr(h) for h in COLOR_MAP.keys()], dtype=np.uint8)
    DST_BGR = np.array([hex_to_bgr(h) for h in COLOR_MAP.values()], dtype=np.uint8)
    SRC_LAB = cv2.cvtColor(SRC_BGR[np.newaxis, :, :], cv2.COLOR_BGR2LAB)[0]

    def make_exclusion_mask_with_canvas(bgr, key_prefix):
        H, W = bgr.shape[:2]

        with st.expander("Ispanak / taşan bölge hariç tutma", expanded=False):
            enabled = st.checkbox(
                "Bu görselde analizden çıkarılacak bölge seç",
                key=f"{key_prefix}_exclude_enabled"
            )

            if not enabled:
                return None

            st.caption("Taşan ıspanak veya analiz dışı kalacak bölgeyi dikdörtgen ile işaretle.")

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            max_display_width = 700
            scale = min(max_display_width / W, 1.0)

            display_w = int(W * scale)
            display_h = int(H * scale)

            pil_resized = pil_img.resize((display_w, display_h))

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.25)",
                stroke_width=3,
                stroke_color="rgba(255, 0, 0, 1)",
                background_image=pil_resized,
                update_streamlit=True,
                height=display_h,
                width=display_w,
                drawing_mode="rect",
                key=f"{key_prefix}_canvas"
            )

            exclusion_mask = np.zeros((H, W), dtype=np.uint8)

            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])

                for obj in objects:
                    if obj.get("type") != "rect":
                        continue

                    left = obj.get("left", 0)
                    top = obj.get("top", 0)
                    width = obj.get("width", 0) * obj.get("scaleX", 1)
                    height = obj.get("height", 0) * obj.get("scaleY", 1)

                    x1 = int(left / scale)
                    y1 = int(top / scale)
                    x2 = int((left + width) / scale)
                    y2 = int((top + height) / scale)

                    x1 = max(0, min(W - 1, x1))
                    x2 = max(0, min(W - 1, x2))
                    y1 = max(0, min(H - 1, y1))
                    y2 = max(0, min(H - 1, y2))

                    if x2 > x1 and y2 > y1:
                        cv2.rectangle(
                            exclusion_mask,
                            (x1, y1),
                            (x2, y2),
                            255,
                            -1
                        )

            if np.count_nonzero(exclusion_mask) == 0:
                return None

            ok, encoded = cv2.imencode(".png", exclusion_mask)

            if not ok:
                return None

            st.caption("Seçilen bölge analizden çıkarılacak.")

            return encoded.tobytes()
    
    def recolor_by_lab(img_bgr, mask):
        H, W = img_bgr.shape[:2]

        lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab_flat = lab_img.reshape(-1, 3).astype(np.int32)

        src32 = SRC_LAB.astype(np.int32)

        recolored = np.full((lab_flat.shape[0], 3), 255, dtype=np.uint8)

        mask_flat = mask.reshape(-1) > 0
        selected_pixels = lab_flat[mask_flat]

        if selected_pixels.size == 0:
            return recolored.reshape(H, W, 3)

        # Her pikselin tüm referans renklere uzaklığını aynı anda hesapla
        diff = selected_pixels[:, None, :] - src32[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)

        nearest_idx = np.argmin(dist2, axis=1)

        recolored[mask_flat] = DST_BGR[nearest_idx]

        return recolored.reshape(H, W, 3)
    
    # =========================================================
    # Ana Pişmişlik Sınıfları
    # =========================================================
    RAW_BGR = np.array(
        [hex_to_bgr(c) for c in [
            "#818100",
            "#666732",
            "#01FEFF",
            "#32CDFF"
        ]],
        dtype=np.uint8
    )

    COOKED_BGR = np.array(
        [hex_to_bgr(c) for c in [
            "#FF99FF",
            "#FF00FF",
            "#FFFE67",
            "#CDCC01",
            "#0167CC",
            "#0101CC",
            "#01FF01",
            "#01CC01"
        ]],
        dtype=np.uint8
    )

    BURNT_BGR = np.array(
        [hex_to_bgr(c) for c in [
            "#FE0001",
            "#C10100"
        ]],
        dtype=np.uint8
    )

    MAIN_LABELS = ["Undercooked", "Cooked", "Overcooked"]
    MAIN_COLORS = ["#FFFF66", "#FF9900", "#C10100"]

    # =========================================================
    # Kızarma Grupları (3 ana grup)
    # =========================================================
    BROWNING_GROUPS = [
    {
        "name": "Light Browning",
        "dst_hex_list": ["#FFFE67", "#CDCC01"],
        "plot_color": "#E5D33F"
    },
    {
        "name": "Medium Browning",
        "dst_hex_list": ["#0167CC", "#0101CC"],
        "plot_color": "#2D7BD8"
    },
    {
        "name": "Dark Browning",
        "dst_hex_list": ["#01FF01", "#01CC01"],
        "plot_color": "#31C85A"
    }
]

    BROWNING_WEIGHTS_BY_SURFACE = {
    # Sıra: Açık Kızarma, Orta Kızarma, Çok Kızarma
    "ust": [20, 55, 90],
    "alt": [45, 75, 95]
}

    BROWNING_GROUP_BGR = [
        np.array([hex_to_bgr(c) for c in group["dst_hex_list"]], dtype=np.uint8)
        for group in BROWNING_GROUPS
    ]

    BROWNING_GROUP_COLORS = [
        group["plot_color"] for group in BROWNING_GROUPS
    ]

    # =========================================================
    # Session State Başlat
    # =========================================================
    for key in [
        "ust_borek_files", "alt_borek_files",
        "ust_borek_uploader_key", "alt_borek_uploader_key"
    ]:
        if key not in st.session_state:
            if "files" in key:
                st.session_state[key] = {}
            else:
                st.session_state[key] = 0

    # =========================================================
    # Grafik Fonksiyonları
    # =========================================================
    def draw_waffle_chart(main_perc):
        total_squares = 100

        exact = np.array(main_perc) / 100.0 * total_squares
        base = np.floor(exact).astype(int)

        remainder = total_squares - base.sum()
        fractional_order = np.argsort(exact - base)[::-1]

        for i in fractional_order[:remainder]:
            base[i] += 1

        waffle_grid = []
        for class_id, square_count in enumerate(base):
            waffle_grid.extend([class_id] * square_count)

        waffle_grid = waffle_grid[:100]
        while len(waffle_grid) < 100:
            waffle_grid.append(int(np.argmax(base)))

        waffle_arr = np.array(waffle_grid).reshape((10, 10))
        cmap_waffle = plt.cm.colors.ListedColormap(MAIN_COLORS)

        fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=100)
        ax.matshow(waffle_arr, cmap=cmap_waffle, vmin=0, vmax=2)

        ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=2)

        ax.set_xticks([])
        ax.set_yticks([])

        legend_elements = [
            Patch(
                facecolor=MAIN_COLORS[i],
                edgecolor="w",
                label=f"{MAIN_LABELS[i]} (%{main_perc[i]:.1f})"
            )
            for i in range(3)
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=3,
            frameon=False,
            fontsize=9
        )

        ax.set_title("Pişmişlik Dağılımı", fontsize=11, pad=10)

        return fig

    def draw_browning_pie_chart(main_counts, browning_counts):
        fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=100)

        pie_labels = (
            ["Undercooked"] +
            [group["name"] for group in BROWNING_GROUPS] +
            ["Overcooked"]
        )

        pie_counts = (
            [main_counts[0]] +
            browning_counts +
            [main_counts[2]]
        )

        # İstersen undercooked rengini MAIN_COLORS[0] bırakabilirsin.
        # Ama Açık Kızarma ile karışmasın diye burada biraz daha farklı verdim.
        pie_colors = (
            ["#FF99FF"] +              # Undercooked
            BROWNING_GROUP_COLORS +    # Açık / Orta / Çok Kızarma
            [MAIN_COLORS[2]]           # Overcooked
        )

        total = max(sum(pie_counts), 1)

        filtered_counts = []
        filtered_colors = []
        legend_labels = []

        for label, count, color in zip(pie_labels, pie_counts, pie_colors):
            if count > 0:
                pct = 100.0 * count / total
                filtered_counts.append(count)
                filtered_colors.append(color)
                legend_labels.append(f"{label}  %{pct:.1f}")

        if len(filtered_counts) == 0:
            ax.text(
                0.5,
                0.5,
                "Veri yok",
                ha="center",
                va="center",
                fontsize=11
            )
            ax.axis("off")
            return fig

        def autopct_func(pct):
            # Küçük dilimlerin içine yüzde basma, yoksa yine sıkışır.
            if pct >= 5:
                return f"%{pct:.1f}"
            return ""

        wedges, texts, autotexts = ax.pie(
            filtered_counts,
            labels=None,  # DIŞ LABEL YOK
            colors=filtered_colors,
            startangle=90,
            counterclock=False,
            autopct=autopct_func,
            pctdistance=0.72,
            wedgeprops=dict(
                edgecolor="white",
                linewidth=1.5
            ),
            textprops=dict(
                fontsize=9,
                color="black"
            )
        )

        for t in autotexts:
            t.set_fontsize(9)
            t.set_fontweight("bold")

        ax.set_title("Kızarma + Pişmişlik Dağılımı", fontsize=11, pad=8)
        ax.axis("equal")

        ax.legend(
            wedges,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=2,
            frameon=False,
            fontsize=8
        )

        # Legend aşağıda kesilmesin diye boşluk bırakıyoruz.
        fig.subplots_adjust(bottom=0.28, top=0.88)

        return fig
    # =========================================================
    # Tek Görsel Analizi
    # =========================================================
    @st.cache_data(show_spinner=False)
    def analyze_image(file_bytes, surface_prefix, exclusion_mask_png_bytes=None):
        arr = np.frombuffer(file_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if bgr is None:
            return None

        
        mask = simple_mask_white_bg(bgr)

        if exclusion_mask_png_bytes is not None:
            exclusion_arr = np.frombuffer(exclusion_mask_png_bytes, np.uint8)
            exclusion_mask = cv2.imdecode(exclusion_arr, cv2.IMREAD_GRAYSCALE)

            if exclusion_mask is not None:
                if exclusion_mask.shape != mask.shape:
                    exclusion_mask = cv2.resize(
                        exclusion_mask,
                        (mask.shape[1], mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                mask[exclusion_mask > 0] = 0

        heat_bgr = recolor_by_lab(bgr, mask)

        flat = heat_bgr[mask > 0]

        if flat.size == 0:
            return None

        raw_count = cnt_color_group(flat, RAW_BGR)
        cooked_count = cnt_color_group(flat, COOKED_BGR)
        burnt_count = cnt_color_group(flat, BURNT_BGR)

        main_counts = [raw_count, cooked_count, burnt_count]
        total = max(sum(main_counts), 1)

        main_perc = [100.0 * c / total for c in main_counts]

        # Kızarma grupları
        browning_counts = []

        for group_colors in BROWNING_GROUP_BGR:
            group_count = cnt_color_group(flat, group_colors)
            browning_counts.append(group_count)

        # COOKED içinde olup browning gruplarına girmeyen renkler:
        # #FF99FF ve #FF00FF gibi erken/açık cooked tonları.
        # Bunları Açık Kızarma grubuna ekliyoruz.
        other_cooked_count = max(cooked_count - sum(browning_counts), 0)

        if len(browning_counts) > 0:
            browning_counts[0] += other_cooked_count

        browning_total = sum(browning_counts)

        if browning_total > 0:
            browning_perc_inside_cooked = [
                100.0 * c / browning_total for c in browning_counts
            ]

            surface_weights = BROWNING_WEIGHTS_BY_SURFACE.get(
                surface_prefix,
                BROWNING_WEIGHTS_BY_SURFACE["ust"]
            )
            
            weighted_score = sum(
                c * surface_weights[i]
                for i, c in enumerate(browning_counts)
            )

browning_score_0_to_100 = weighted_score / browning_total
            
            dominant_browning_index = int(np.argmax(browning_counts))
            dominant_browning_name = BROWNING_GROUPS[dominant_browning_index]["name"]
        else:
            browning_perc_inside_cooked = [0.0 for _ in BROWNING_GROUPS]
            browning_score_0_to_100 = 0.0
            dominant_browning_name = "Yok"

        return {
            "bgr": bgr,
            "heat_bgr": heat_bgr,
            "main_counts": main_counts,
            "main_perc": main_perc,
            "browning_counts": browning_counts,
            "browning_perc_inside_cooked": browning_perc_inside_cooked,
            "browning_score_0_to_100": browning_score_0_to_100,
            "dominant_browning_name": dominant_browning_name
        }

    # =========================================================
    # Bölüm Render Fonksiyonu
    # =========================================================
    def render_surface_section(section_title, prefix):
        files_key = f"{prefix}_borek_files"
        uploader_key_name = f"{prefix}_borek_uploader_key"

        st.header(section_title)

        upload_col1, upload_col2 = st.columns([4, 1])

        with upload_col1:
            uploads = st.file_uploader(
                f"{section_title} için görsel yükle",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"{prefix}_uploader_{st.session_state[uploader_key_name]}"
            )

        with upload_col2:
            st.write("")
            st.write("")
            if st.button("Listeyi Temizle", key=f"{prefix}_clear_all"):
                st.session_state[files_key] = {}
                st.session_state[uploader_key_name] += 1
                st.rerun()

        if uploads:
            for up in uploads:
                file_bytes = up.getvalue()
                file_hash = hashlib.md5(file_bytes).hexdigest()
                file_id = f"{up.name}_{len(file_bytes)}_{file_hash}"

                if file_id not in st.session_state[files_key]:
                    st.session_state[files_key][file_id] = {
                        "name": up.name,
                        "bytes": file_bytes
                    }

        files = list(st.session_state[files_key].items())

        if not files:
            st.info(f"{section_title} için henüz görsel yüklenmedi.")
            return

        st.success(f"{len(files)} adet görsel analiz listesinde.")

        analyzed_items = []

        for file_id, item in files:

            arr_preview = np.frombuffer(item["bytes"], np.uint8)
            bgr_preview = cv2.imdecode(arr_preview, cv2.IMREAD_COLOR)

            exclusion_mask_png_bytes = None

            if bgr_preview is not None:
                exclusion_mask_png_bytes = make_exclusion_mask_with_canvas(
                    bgr_preview,
                    key_prefix=f"{prefix}_{file_id}"
                )

            result = analyze_image(
                item["bytes"],
                prefix,
                exclusion_mask_png_bytes
            )

            if result is None:
                st.warning(f"{item['name']} okunamadı veya maske oluşturulamadı, atlandı.")
                continue

            analyzed_items.append({
                "file_id": file_id,
                "name": item["name"],
                "result": result
            })

        if not analyzed_items:
            st.warning("Analiz edilebilir görsel bulunamadı.")
            return

        for idx, item in enumerate(analyzed_items, start=1):
            file_id = item["file_id"]
            name = item["name"]
            r = item["result"]

            bgr = r["bgr"]
            heat_bgr = r["heat_bgr"]
            main_perc = r["main_perc"]
            main_counts = r["main_counts"]
            browning_counts = r["browning_counts"]
            browning_perc_inside_cooked = r["browning_perc_inside_cooked"]
            browning_score_0_to_100 = r["browning_score_0_to_100"]
            dominant_browning_name = r["dominant_browning_name"]

            title_col, remove_col = st.columns([5, 1])

            with title_col:
                st.markdown(f"## {idx}. Analiz: `{name}`")

            with remove_col:
                if st.button("Kaldır", key=f"{prefix}_remove_{file_id}"):
                    if file_id in st.session_state[files_key]:
                        del st.session_state[files_key][file_id]
                    st.rerun()

            # =====================================================
            # 1. SATIR -> Orijinal | Analiz
            # =====================================================
            row1_col1, row1_col2 = st.columns(2, gap="medium")

            with row1_col1:
                st.subheader("Orijinal")
                st.image(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )

            with row1_col2:
                st.subheader("Analiz")
                st.image(
                    cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )

            # =====================================================
            # 2. SATIR -> Waffle | Browning Pie
            # =====================================================
            row2_col1, row2_col2 = st.columns(2, gap="medium")

            with row2_col1:
                st.subheader("Pişmişlik Dağılımı")
                fig_waffle = draw_waffle_chart(main_perc)
                st.pyplot(fig_waffle, clear_figure=True, use_container_width=True)
                plt.close(fig_waffle)

            with row2_col2:
                st.subheader("Kızarma Dağılımı")
                fig_browning = draw_browning_pie_chart(
                main_counts,
                browning_counts
                )
                st.pyplot(fig_browning, clear_figure=True, use_container_width=True)
                plt.close(fig_browning)

            # =====================================================
            # 3. SATIR -> Özet Tablo
            # =====================================================
            st.subheader("Özet")

            summary_df = pd.DataFrame([{
                "Undercooked (%)": round(main_perc[0], 2),
                "Cooked (%)": round(main_perc[1], 2),
                "Overcooked (%)": round(main_perc[2], 2),
                "Kızarma Skoru / 100": round(browning_score_0_to_100, 1),
                "Baskın Kızarma Tipi": dominant_browning_name
            }])

            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.divider()

    # =========================================================
    # Sayfa
    # =========================================================
    st.title("Börek Analizi")

    render_surface_section("Üst Yüzey Analizi", "ust")
    st.markdown("---")
    render_surface_section("Alt Yüzey Analizi", "alt")




# ==========================================
# 1. SMALL CAKE ÖZEL SABİTLER
# ==========================================

# Ry Değeri Eşikleri
SHADE_THRESHOLDS = [
    (7.2, 17), (9.3, 16), (12.2, 15), (16.4, 14), (20.1, 13),
    (22.9, 12), (26.5, 11), (31.7, 10), (38.5, 9), (46.9, 8),
    (54.2, 7), (64.3, 6), (75.2, 5)
]

# Renk Kodu -> BGR
SHADE_COLOR_MAP_BGR = {
    4:  (73,  74,  38),   # Çok Açık
    5:  (45,  64,  54),
    6:  (0,  255,  255),  # Sarı
    7:  (0,  192,  255),
    8:  (128, 128, 255),  # Kırmızımsı
    9:  (255,   0, 255),  # Mor
    10: (128, 255, 255),
    11: (0, 128, 128),
    12: (255, 128, 128),
    13: (255,   0, 128),
    14: (0, 255,   0),    # Yeşil
    15: (0, 128,   0),
    16: (255,   0,   0),  # Mavi 
    17: (128,   0,   0)   # Koyu
}

SHADE_COLOR_MAP_RGB = {k: (v[2]/255.0, v[1]/255.0, v[0]/255.0) for k, v in SHADE_COLOR_MAP_BGR.items()}

# ==========================================
# 2. YARDIMCI FONKSİYONLAR
# ==========================================

def apply_clahe_lab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)

def get_shade_number(ry_val):
    for limit, shade in SHADE_THRESHOLDS:
        if ry_val < limit:
            return shade
    return 4

def robust_cake_mask(img_bgr):
    alpha = 1.2
    beta = 30
    image_for_masking = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    gray = cv2.cvtColor(image_for_masking, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    cake_mask = cv2.bitwise_not(white_mask)
    return cake_mask

def get_inscribed_circle(mask_u8):
    """
    Bir maskenin içine sığabilecek EN BÜYÜK daireyi (Inscribed Circle) bulur.
    Bunun için Distance Transform kullanır.
    """
    # Her pikselin en yakın sıfıra (siyaha) olan uzaklığını hesapla
    dist_transform = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    
    # En büyük uzaklık değeri = Yarıçap
    # En büyük uzaklığın olduğu yer = Merkez
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
    
    radius = max_val
    center = max_loc # (x, y)
    
    return center, radius

def get_13_zones(mask_shape, center, r_max):
    H, W = mask_shape[:2]
    cx, cy = int(center[0]), int(center[1])
    r1 = int(r_max)
    r2 = int(0.6 * r1)
    r3 = int(0.3 * r1)
    
    zones = []
    
    # 1. Merkez (C)
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r3, 255, -1)
    zones.append(("C", m))

    # 2. İç Halka (M1-M4)
    for i in range(4):
        m = np.zeros((H, W), dtype=np.uint8)
        angle_start = i * 90
        angle_end = (i + 1) * 90
        cv2.ellipse(m, (cx, cy), (r2, r2), 0, angle_start, angle_end, 255, -1)
        cv2.circle(m, (cx, cy), r3, 0, -1)
        zones.append((f"M{i+1}", m))

    # 3. Dış Halka (O1-O8)
    for i in range(8):
        m = np.zeros((H, W), dtype=np.uint8)
        angle_start = i * 45
        angle_end = (i + 1) * 45
        cv2.ellipse(m, (cx, cy), (r1, r1), 0, angle_start, angle_end, 255, -1)
        cv2.circle(m, (cx, cy), r2, 0, -1)
        zones.append((f"O{i+1}", m))
        
    return zones, r1, r2, r3

def create_pixel_heatmap(img_bgr, circle_mask_u8):
    """
    Bölge ortalaması almadan, her pikseli kendi Ry değerine göre boyar.
    Sadece circle_mask_u8 içindeki alanı işler.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L_channel = lab[:, :, 0]
    
    # Ry haritası hesapla (Vektörize işlem - Hızlı)
    # Ry = (L / 255) * 100
    ry_map = (L_channel.astype(np.float32) / 255.0) * 100.0
    
    # Çıktı resmi (Boş)
    heatmap = np.zeros_like(img_bgr)
    
    # Maskenin dolu olduğu yerlerdeki koordinatlar
    y_idxs, x_idxs = np.where(circle_mask_u8 > 0)
    
    if len(y_idxs) == 0:
        return heatmap

    # İlgili piksellerin Ry değerleri
    target_rys = ry_map[y_idxs, x_idxs]
    
    # Her piksel için tek tek renk bulmak yavaş olabilir ama en doğrusu bu.
    # Hızlandırmak için np.digitize kullanılabilir ama senin eşikler non-linear.
    # Basit bir map ile yapalım:
    
    # Piksel piksel boyama (Görselleştirme amacıyla)
    for y, x, ry in zip(y_idxs, x_idxs, target_rys):
        shade = get_shade_number(ry)
        color = SHADE_COLOR_MAP_BGR.get(shade, (128,128,128))
        heatmap[y, x] = color
        
    return heatmap

def analyze_single_cake(img_bgr, mask_bool):
    processed = apply_clahe_lab(img_bgr)
    mask_u8 = mask_bool.astype(np.uint8) * 255
    
    # 1. GEOMETRİ: İçeri sığan en büyük daire (Inscribed Circle)
    (cx, cy), radius = get_inscribed_circle(mask_u8)
    
    # Eğer radius çok küçükse (gürültü) atla
    if radius < 5: return None, None, None, None
    
    # 13 Bölgeyi oluştur
    zones, r1, r2, r3 = get_13_zones(img_bgr.shape, (cx, cy), radius)
    
    vis_layer_zones = np.zeros_like(img_bgr)
    line_layer = np.zeros_like(mask_u8)
    
    zone_results = []
    
    # Temiz Daire Maskesi (Analizin sınırları artık bu mükemmel daire)
    clean_circle_mask = np.zeros_like(mask_u8)
    cv2.circle(clean_circle_mask, (int(cx), int(cy)), int(radius), 255, -1)

    # --- A) BÖLGESEL ANALİZ (13 ZONE) ---
    for z_name, z_mask in zones:
        valid_zone = (z_mask > 0)
        
        # Çizgiler
        z_cnts, _ = cv2.findContours(valid_zone.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(line_layer, z_cnts, -1, 255, 2)
        
        # Renk Hesabı
        pixels = processed[valid_zone]
        if len(pixels) == 0: continue

        brightness = np.mean(pixels, axis=1)
        if len(brightness) > 0:
            p5, p95 = np.percentile(brightness, [5, 95])
            filtered_pixels = pixels[(brightness >= p5) & (brightness <= p95)]
            if len(filtered_pixels) == 0: filtered_pixels = pixels
        else:
            filtered_pixels = pixels
        
        avg_bgr = np.mean(filtered_pixels, axis=0)
        lab_px = cv2.cvtColor(np.uint8([[avg_bgr]]), cv2.COLOR_BGR2Lab)[0][0]
        L_val = lab_px[0]
        ry = (L_val / 255.0) * 100.0
        
        shade = get_shade_number(ry)
        color = SHADE_COLOR_MAP_BGR.get(shade, (128, 128, 128))
        zone_results.append(shade)
        
        # Boyama (Solid)
        vis_layer_zones[valid_zone] = color
        
        # Yazı
        M = cv2.moments(z_mask)
        if M["m00"] > 0:
            tx, ty = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(vis_layer_zones, str(shade), (tx-6, ty+4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Siyah çizgileri bas
    vis_layer_zones[line_layer > 0] = (0, 0, 0)
    
    # --- B) PİKSEL ANALİZİ (HEATMAP) ---
    # Sadece o temiz dairenin içindeki her pikseli analiz et
    vis_layer_pixel = create_pixel_heatmap(processed, clean_circle_mask)
    # Piksel analizinde de dış çerçeveyi siyah çizelim ki net dursun
    vis_layer_pixel[line_layer > 0] = (0,0,0)

    return vis_layer_zones, vis_layer_pixel, zone_results, clean_circle_mask

# =========================
# 3. ARAYÜZ VE AKIŞ
# =========================

def run_smallcake():
    
    st.markdown("<style>.block-container h1{margin-top:-80px}</style>", unsafe_allow_html=True)
    st.title("Small Cake Analizi")
                
    uploads = st.file_uploader("Kek Görseli Yükle", type=["jpg","jpeg","png"], accept_multiple_files=True, key="uploads_sc")

    if not uploads:
        st.info("Lütfen analiz edilecek kek görsellerini yükleyin.")
        return

    for up in uploads:
        st.divider()
        st.subheader(f"Dosya: {up.name}")
        
        file_bytes = np.frombuffer(up.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None: continue
        
        mask_u8 = robust_cake_mask(img_bgr)
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sonuç Tuvalleri (Beyaz Zemin)
        # EN standardı gibi temiz görünmesi için beyaz zemin kullanıyoruz.
        h, w = img_bgr.shape[:2]
        canvas_zones = np.ones((h, w, 3), dtype=np.uint8) * 255
        canvas_pixels = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        all_file_shades = []

        found_any = False
        for c in cnts:
            if cv2.contourArea(c) < 1000: continue
            
            single_mask = np.zeros_like(mask_u8)
            cv2.drawContours(single_mask, [c], -1, 255, -1)
            
            # Analiz Fonksiyonu (Hem Zone hem Pixel döndürür)
            v_zones, v_pixel, shades, clean_mask = analyze_single_cake(img_bgr, single_mask > 0)
            
            if v_zones is not None:
                found_any = True
                roi = clean_mask > 0
                
                # Zone Canvas'a işle
                canvas_zones[roi] = v_zones[roi]
                # Siyah çizgileri netleştir (Zone için)
                black_px_z = np.all(v_zones == [0,0,0], axis=-1) & roi
                canvas_zones[black_px_z] = [0,0,0]
                
                # Pixel Canvas'a işle
                canvas_pixels[roi] = v_pixel[roi]
                # Siyah çizgileri netleştir (Pixel için - opsiyonel, çerçeve görünsün diye)
                black_px_p = np.all(v_pixel == [0,0,0], axis=-1) & roi
                canvas_pixels[black_px_p] = [0,0,0]

                all_file_shades.extend(shades)

        if not found_any:
            st.warning("Kek tespit edilemedi.")
            continue

        # --- GÖRSELLEŞTİRME (3 KOLON) ---
        c1, c2, c3 = st.columns([1, 1, 1])
        
        with c1:
            st.markdown("##### 1. Orijinal Görüntü")
            st.markdown("*Ham Görüntü*")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            if all_file_shades:
                avg_s = sum(all_file_shades) / len(all_file_shades)
                st.info(f"Ortalama Shade: **{avg_s:.2f}**")
                
        with c2:
            st.markdown("##### 2. Bölgesel (Zone) Analiz")
            st.markdown("*EN Standardı Stili (Solid)*")
            st.image(cv2.cvtColor(canvas_zones, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with c3:
            st.markdown("##### 3. Piksel Bazlı Analiz")
            st.markdown("*Bölge ortalaması yok, ham doku*")
            st.image(cv2.cvtColor(canvas_pixels, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Grafik Alanı
        if all_file_shades:
            shade_counts = {s: all_file_shades.count(s) for s in set(all_file_shades)}
            sorted_shades = sorted(shade_counts.keys())
            counts = [shade_counts[s] for s in sorted_shades]
            bar_colors = [SHADE_COLOR_MAP_RGB.get(s, (0.5,0.5,0.5)) for s in sorted_shades]
            
            fig, ax = plt.subplots(figsize=(10, 3))
            bars = ax.bar(range(len(counts)), counts, color=bar_colors, tick_label=[str(s) for s in sorted_shades])
            ax.set_title("Bölgesel Renk Dağılımı")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 1), textcoords="offset points", ha='center', va='bottom')
            st.pyplot(fig)

# ==========================================
#  ROUTER (YÖNLENDİRME)
# ==========================================

if st.session_state.current_page == "Home":
    show_home_page()

else:
    # İç sayfalarda sidebar geri gelsin
    with st.sidebar:
        st.markdown("""<style>[data-testid="stSidebar"] {display: block;}</style>""", unsafe_allow_html=True)
        
        if st.button("🏠 Ana Sayfa", use_container_width=True, on_click=change_page, args=("Home",)):
            pass # on_click zaten işi yapıyor
        
        st.divider()
        st.caption(f"Mod: {st.session_state.current_page}")

    # Fonksiyonları Çalıştır
    if st.session_state.current_page == "Patates":
        run_potato()
    elif st.session_state.current_page == "Pizza":
        run_pizza()
    elif st.session_state.current_page == "Börek":
        run_borek()
    elif st.session_state.current_page == "Small Cake":
        run_smallcake()

 













