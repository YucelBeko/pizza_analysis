import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import base64
import pandas as pd, os, glob
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
import matplotlib.cm as cm
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None
from matplotlib.patches import Patch
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
    #up_files = st.file_uploader("Görsel yükle", type=["jpg","jpeg","png"], accept_multiple_files=True)
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
                # --- Piksel etiketleri ---
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
        st.set_page_config(page_title="Pişme Analizi", layout="wide")
        st.title("Patates Kızartması Analizi")

        uploads = st.file_uploader("Görselleri yükle (tek/çoklu)", type=["jpg","jpeg","png"], accept_multiple_files=True)

        if uploads:
            for up in uploads:
                file_bytes = np.frombuffer(up.read(), np.uint8)
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
        st.set_page_config(page_title="Pişme Analizi", layout="wide", )

        st.title("Pizza Analizi")

        # =========================
        # MASKE SABİTLERİ
        # =========================
        HSV_V_BLACK_MAX = 40
        HSV_V_GLARE_MIN = 225
        HSV_S_GLARE_MAX = 30

        LAB_L_MIN, LAB_L_MAX = 60, 250
        LAB_A_MIN, LAB_A_MAX =  2,  90
        LAB_B_MIN, LAB_B_MAX =  6, 130

        DOUGH_L_MIN, DOUGH_L_MAX = 100, 250
        DOUGH_A_MIN, DOUGH_A_MAX = -5,  10
        DOUGH_B_MIN, DOUGH_B_MAX =  0,  60
        
        # --- Yanık sınıfını sıkılaştırma (yalnızca siyahımsı pikseller 'burnt')
        # --- Yanık (siyahımsı) için hibrit eşikler (LAB + HSV)
        BURNT_L_MAX = 120        # 0..255  (düşük L = koyu)
        BURNT_CHROMA_MAX = 80     # a/b'den kroma  (düşük kroma = renksiz/siyaha yakın)
        BURNT_V_MAX = 105          # HSV parlaklık (V) eşiği
        BURNT_S_MAX = 125         # HSV satürasyon (S) eşiği

        # === Bölgesel analiz sabitleri ===
        TARGET_REGION_CLASSES = ("brown", "dark_brown")  # yanık HARİÇ
        REGION_BINS = (0, 0.2, 0.4, 0.6, 0.8, 1.0)      # renk skalası dilimleri (0-100%)

        CONTOUR_COLOR = (0,255,0)
        CONTOUR_THICK = 1

        # Sınıf isimleri (L artan: koyudan açığa)
        CLASS_NAMES = ["burnt", "dark_brown", "brown", "light_brown", "dough"]

        # =========================
        # ÖZEL ISI HARİTASI RENK PALETİ (çiğ→yanık)
        # =========================
        # 1: çiğ  → 5: yanık
        CUSTOM_HEX = ["#FF99FF", "#FFFFB5","#FFFF66" , "#CCCC00", "#FF0000"]  # dough, light_brown, brown, dark_brown, burnt

        def make_custom_cmap():
            """Çiğ→yanık 5 renk için adımlı cmap/norm/bins döndürür."""
            cmap = ListedColormap(CUSTOM_HEX, name="pizza5")     # 5 renk
            bins = np.linspace(0.0, 1.0, 6)                      # 0..1 arası 5 dilim
            norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True) # adımlı eşleme
            return cmap, norm, bins

        CMAP5, NORM5, BINS5 = make_custom_cmap()

        # =========================
        # YARDIMCI
        # =========================
        def pct_line(perc: dict, order: list[str]) -> str:
            """Sınıf yüzdelerini 'burnt: 12.3% | ...' formatında tek satır döndürür."""
            return " | ".join(f"{k}: {perc[k]*100:.1f}%" for k in order)

        # =========================
        # MASKELEME
        # =========================
        def hsv_exclusion_mask(img):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S, V = hsv[...,1], hsv[...,2]
            return (V <= HSV_V_BLACK_MAX) | ((V >= HSV_V_GLARE_MIN) & (S <= HSV_S_GLARE_MAX))

        def lab_inclusion_mask(img):
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
            m = (mask_u8 > 0).astype(np.uint8)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
            if n <= 1: return m*255
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            return ((lab == k).astype(np.uint8) * 255)

        def build_pizza_mask_from_ranges(img, keep_only_largest=True):
            inc = lab_inclusion_mask(img)
            exc = hsv_exclusion_mask(img)
            m0  = (inc & (~exc)).astype(np.uint8) * 255
            return keep_largest_component(m0) if keep_only_largest else m0

        def outline_on(img, mask_u8, color=CONTOUR_COLOR, thick=CONTOUR_THICK):
            out = img.copy()
            if (mask_u8 > 0).any():
                cnts,_ = cv2.findContours((mask_u8 > 0).astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, cnts, -1, color, thick, lineType=cv2.LINE_AA)
            return out

        # =========================
        # KMEANS (5 sınıf) — eğit & tahmin
        # =========================
        def _brown_score(centers_lab):
            # centers_lab: shape (k,3)  [L, a0, b0]  (a0,b0: -128..127)
            L = centers_lab[:, 0].astype(np.float32)
            A = centers_lab[:, 1].astype(np.float32)
            B = centers_lab[:, 2].astype(np.float32)
            return 0.6*(255.0 - L) + 0.4*np.maximum(A, 0) + 0.15*np.maximum(B, 0)  # büyük = daha kahverengi

        def _stack_masked_lab(img_bgr, mask_u8, max_pix=30000, seed=42):
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
            L, A0, B0 = lab[...,0], lab[...,1]-128, lab[...,2]-128
            sel = (mask_u8 > 0)
            if not np.any(sel): return np.empty((0,3), np.float32)
            X = np.stack([L[sel], A0[sel], B0[sel]], axis=1).astype(np.float32)
            if X.shape[0] > max_pix:
                rs = np.random.RandomState(seed)
                X = X[rs.choice(X.shape[0], size=max_pix, replace=False)]
            return X

        def fit_kmeans_5(images_bgr, mask_fn, seed=42):
            if KMeans is None:
                raise RuntimeError("scikit-learn gerekli: pip install scikit-learn")
            pools = []
            for img in images_bgr:
                m = mask_fn(img)
                X = _stack_masked_lab(img, m, max_pix=30000, seed=seed)
                if X.size: pools.append(X)
            if not pools:
                raise RuntimeError("Maske içinde örnek piksel bulunamadı.")
            X = np.vstack(pools).astype(np.float32)

            # --- ÖLÇEKLEME (train)
            mean_ = X.mean(axis=0)
            std_  = X.std(axis=0) + 1e-6
            Xs = (X - mean_) / std_

            km = KMeans(n_clusters=5, random_state=seed, n_init=10)
            km.fit(Xs)

            # Merkezleri orijinal LAB uzayına geri çevir (rapor/görselleme için)
            centers = km.cluster_centers_ * std_ + mean_   # (5,3) L,a0,b0

            # --- KÜMELERİ BROWN SCORE’a göre sırala (büyük→küçük)
            score = _brown_score(centers)
            order = np.argsort(-score)                # burnt, dark_brown, brown, light_brown, dough
            inv = np.empty_like(order); inv[order] = np.arange(order.size)

            #class_order = ["burnt", "dark_brown", "brown", "light_brown", "dough"]
            class_order = ["Burnt", "Dark Brown", "Brown", "Light Brown", "Dough"]

            # UI’de görmek için sakla
            st.session_state.km_mean = mean_
            st.session_state.km_std  = std_

            return km, order, inv, class_order, centers[order]

        def predict_kmeans_map(img_bgr, mask_u8, km, inv, class_order):
            H,W = mask_u8.shape
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
            L, A0, B0 = lab[...,0], lab[...,1]-128, lab[...,2]-128
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            S, V = hsv[...,1].astype(np.int16), hsv[...,2].astype(np.int16)
            m = (mask_u8 > 0)

            class_idx = np.full((H,W), -1, dtype=np.int16)
            if not np.any(m):
                counts = {c:0 for c in class_order}; perc = {c:0.0 for c in class_order}; dom = None
                return class_idx, counts, perc, dom

            # --- KMeans tahmini (ölçekleme eğitimle aynı)
            X = np.stack([L[m], A0[m], B0[m]], axis=1).astype(np.float32)
            mean_ = st.session_state.get("km_mean", np.zeros(3, np.float32))
            std_  = st.session_state.get("km_std",  np.ones(3,  np.float32))
            Xs = (X - mean_) / std_
            labels = km.predict(Xs)
            ranked = inv[labels]                 # 0: burnt ... 4: dough
            class_idx[m] = ranked

            burnt_i = class_order.index("Burnt")
            dark_i  = class_order.index("Dark Brown")

            # --- HİBRİT SİYAHLIK: (LAB koyu & renksiz) VEYA (HSV düşük V & düşük S)
            chroma = np.sqrt(A0.astype(np.float32)**2 + B0.astype(np.float32)**2)
            blackish_lab = (L <= BURNT_L_MAX) & (chroma <= BURNT_CHROMA_MAX)
            blackish_hsv = (V <= BURNT_V_MAX) & (S <= BURNT_S_MAX)
            blackish = (blackish_lab | blackish_hsv) & m

            # 1) siyahımsı → YANIK'a TERFİ
            class_idx[blackish] = burnt_i

            # 2) ama siyahımsı değilse ve 'burnt' etiketliyse → DARK_BROWN'a DÜŞÜR
            wrong_burnt = (class_idx == burnt_i) & (~blackish) & m
            dough_i = class_order.index("Dough")
            light_i = class_order.index("Light Brown")
        
            # Dough seçilmiş ama içinde biraz "sarılık/kırmızılık" (A veya B) olanları terfi ettir
            # Bu sayıları düşürürseniz "Light Brown" alanı genişler (Pembe azalır).
            UPGRADE_A_MIN = 5   # Kırmızılık eşiği (Düşürürseniz pembe alan azalır)
            UPGRADE_B_MIN = 10  # Sarılık eşiği    (Düşürürseniz pembe alan azalır)
        
            is_dough = (class_idx == dough_i)
            # Hamur ise VE (kırmızılık > 5 VEYA sarılık > 10) ise -> Light Brown yap
            should_be_cooked = is_dough & ((A0 > UPGRADE_A_MIN) | (B0 > UPGRADE_B_MIN))
            class_idx[should_be_cooked] = light_i

            counts = {c: int(np.count_nonzero(class_idx == i)) for i,c in enumerate(class_order)}
            class_idx[wrong_burnt] = dark_i

            # --- istatistik
            counts = {c: int(np.count_nonzero(class_idx == i)) for i,c in enumerate(class_order)}
            tot = sum(counts.values())
            perc = {c: (counts[c]/tot if tot>0 else 0.0) for c in class_order}
            dominant = max(class_order, key=lambda c: counts[c]) if tot>0 else None
            return class_idx, counts, perc, dominant

        # =========================
        # ISI HARİTASI (özel renk paleti)
        # =========================
        def heatmap_overlay(img_bgr, class_idx, class_order, alpha=0.6, cmap=CMAP5, norm=NORM5):
            """
            class_idx: 0..4 (0=burnt, 4=dough). Skor = burnt(1) → dough(0).
            Özel palet: dough→light_brown→brown→dark_brown→burnt (CUSTOM_HEX).
            """
            n = len(class_order)
            score = np.zeros(class_idx.shape, dtype=np.float32)
            valid = (class_idx >= 0)
            score[valid] = 1.0 - (class_idx[valid].astype(np.float32) / (n-1))  # 1=burnt ... 0=dough

            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            colored = (sm.to_rgba(score)[...,:3] * 255).astype(np.uint8)        # RGB
            colored_bgr = colored[..., ::-1]

            out = img_bgr.copy().astype(np.float32)
            out[valid] = (1-alpha)*out[valid] + alpha*colored_bgr[valid]
            return np.clip(out,0,255).astype(np.uint8)

        def show_heatmap_figure(img_bgr, overlay_bgr, cmap=CMAP5, norm=NORM5, bins=BINS5):
            fig, axes = plt.subplots(1,3, figsize=(14,5), gridspec_kw={"width_ratios":[1,1,0.06]}, dpi=180)
            axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); axes[0].set_title(""); axes[0].axis("off")
            axes[1].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)); axes[1].set_title(""); axes[1].axis("off")

            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=axes[2])
            tick_pos = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
            tick_lbl = ["Dough","Light Brown","Brown","Dark Brown","Burnt"]
            cbar.set_ticks(tick_pos); cbar.set_ticklabels(tick_lbl); cbar.set_label("Browning Level")

            plt.tight_layout()
            return fig
        def class_colors_for_pie(order: list[str]):
            palette = {
                "Dough":       CUSTOM_HEX[0],
                "Ligh Brown": CUSTOM_HEX[1],
                "Brown":       CUSTOM_HEX[2],
                "Dark Brown":  CUSTOM_HEX[3],
                "Burnt":       CUSTOM_HEX[4],
            }
            return [palette[c] for c in order]
        def _build_region_map(mask_u8, n_sectors, n_rings):
            """Maskeyi (halka × dilim) bölgelere böler ve her piksele bölge id'si verir."""
            m = (mask_u8 > 0)
            H, W = m.shape
            M = cv2.moments(m.astype(np.uint8), binaryImage=True)
            cx, cy = (W/2, H/2) if M["m00"] == 0 else (M["m10"]/M["m00"], M["m01"]/M["m00"])

            yy, xx = np.indices((H, W))
            dx, dy = xx - cx, yy - cy
            r = np.sqrt(dx*dx + dy*dy)
            theta = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)

            r_max = np.sqrt(max(cx, W-cx)**2 + max(cy, H-cy)**2) + 1e-6
            r_idx = np.clip((r / r_max) * n_rings, 0, n_rings - 1e-6).astype(int)
            th_idx = (theta / (2*np.pi) * n_sectors).astype(int)

            reg_map = (r_idx * n_sectors + th_idx)   # 0..(n_rings*n_sectors-1)
            return reg_map, (int(cx), int(cy))


        def _region_ratios(class_idx, mask_u8, class_order,
                        target_classes, n_sectors, n_rings):
            """Her bölge için (hedef sınıf pikselleri / tüm maske pikselleri) oranını döndürür."""
            reg_map, _ = _build_region_map(mask_u8, n_sectors, n_rings)
            m = (mask_u8 > 0)
            K = n_sectors * n_rings
            tgt_idx = [class_order.index(c) for c in target_classes if c in class_order]

            ratios = np.full(K, np.nan, dtype=np.float32)
            for k in range(K):
                region = (reg_map == k) & m
                tot = int(np.count_nonzero(region))
                if tot == 0:
                    continue
                num = sum(int(np.count_nonzero((class_idx == ti) & region)) for ti in tgt_idx)
                ratios[k] = num / tot
            return ratios, reg_map


        def _uniformity_score(ratios):
            """0..100: 100 = tüm bölgelerde aynı oran (tam homojen)."""
            v = ratios[~np.isnan(ratios)]
            if v.size < 2:
                return 0.0
            std = float(np.std(v))
            # [0,1] aralığında max std ≈ 0.5 kabul edilip normalize edilir
            score = 1.0 - (std / 0.5)
            return float(np.clip(score, 0.0, 1.0) * 100.0)


        def region_overlay_figure_kmeans(img_bgr, class_idx, mask_u8, class_order,
                                        n_sectors=8, n_rings=2, alpha=0.55):
            """
            Bölgesel analiz (dilim × halka):
            - Her bölgenin class_idx ortalaması alınır (0: burnt ... 4: dough).
            - Ortalama en yakın tam sayıya yuvarlanır ve tüm bölge o sınıfın rengine boyanır.
            - Homojenlik skoru: bölge ortalamalarının (0..4) normalize edilmiş std'süne göre (0..100).
            """
            # --- sınıf indeksine karşılık gelen HEX rengi (index 0=burnt → CUSTOM_HEX[4], 4=dough → CUSTOM_HEX[0])
            idx2hex = [CUSTOM_HEX[4 - i] for i in range(len(class_order))]  # ["#FF0000", "#00FF00", "#FFFF66", "#33CCFF", "#666633"]

            # Bölge haritası
            reg_map, (cx, cy) = _build_region_map(mask_u8, n_sectors, n_rings)
            m = (mask_u8 > 0)

            overlay = img_bgr.copy().astype(np.float32)
            K = n_sectors * n_rings
            region_means = np.full(K, np.nan, dtype=np.float32)

            for k in range(K):
                region = (reg_map == k) & m
                if not np.any(region):
                    continue
                vals = class_idx[region]           # 0..4 (burnt..dough), -1 yok çünkü m ile kesiyoruz
                mean_idx = float(np.mean(vals))
                region_means[k] = mean_idx

                idx_round = int(np.clip(np.rint(mean_idx), 0, len(class_order)-1))
                # HEX -> BGR
                h = idx2hex[idx_round].lstrip("#")
                rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                bgr = np.array(rgb[::-1], dtype=np.float32)

                overlay[region] = (1 - alpha) * overlay[region] + alpha * bgr

            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            # kılavuz çizgileri
            # kılavuz çizgileri sadece maskenin içinde
            ys, xs = np.where(m)
            if xs.size > 0:
                rM = int(np.sqrt(((xs - cx)**2 + (ys - cy)**2).max()))
            
            # boş çizgi katmanı
                lines = np.zeros_like(overlay, dtype=np.uint8)
            
            # çizgileri çiz
                for s in range(n_sectors):
                    th = 2*np.pi*s/n_sectors
                    x2 = int(cx + rM*np.cos(th)); y2 = int(cy + rM*np.sin(th))
                    cv2.line(lines, (int(cx), int(cy)), (x2, y2), (150,200,100), 15, cv2.LINE_AA)
                for r_i in range(1, n_rings):
                    cv2.circle(lines, (int(cx), int(cy)), int(rM*r_i/n_rings), (150,200,100), 15, cv2.LINE_AA)
        #150,100,100
                # maskeyi 3 kanala genişlet
                mask3 = (mask_u8 > 0).astype(np.uint8) 
                mask3 = np.repeat(mask3[:, :, None], 6, axis=2)

                # sadece maskenin içindeki çizgileri bırak
                lines = cv2.bitwise_and(lines, lines, mask=mask3[:,:,0])

                # overlay ile birleştir
                overlay = cv2.addWeighted(overlay, 1.0, lines, 1.0, 0)

            # FIGÜR (kompakt: Orijinal | Bölgesel)

            fig = plt.figure(figsize=(5.5, 2.9), dpi=150)  # 6x3 yerine daha kompakt
            gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.06)

            ax0 = fig.add_subplot(gs[0,0])
            ax0.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            ax0.set_title("Orijinal", fontsize=10, pad=2); ax0.axis("off")

            ax1 = fig.add_subplot(gs[0,1])
            ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax1.set_title("Bölgesel (ortalama sınıf rengi)", fontsize=10, pad=2); ax1.axis("off")

            fig.tight_layout(pad=0.6)

            # Homojenlik skoru: bölge ortalamalarını 0..1'e normalize edip (1 - std/0.5)*100
            # (0=dough .. 4=burnt  ⇒  score = 1 - mean_idx/4)
            with np.errstate(invalid="ignore"):
                browning_norm = 1.0 - (region_means / max(1, (len(class_order)-1)))  # 0..1
            uni = _uniformity_score(browning_norm)  # 0..100
            return fig, uni
        # =========================
        # SIDEBAR (sadece kontur seçeneği)
        # =========================
        #with st.sidebar:
            #st.subheader("Ayarlar")
            #show_mask_outline = st.checkbox("Maskeyi konturla göster", True)
            #st.caption("Tam çözünürlükte işlenir; eğitimde örnekleme yapılır.")

        
        with st.sidebar:
            # --- DEFAULTS ---
            DEFAULT_SECTORS = 8
            DEFAULT_RINGS   = 1
            DEFAULT_SHOW_REGION = True   # veya False, senin ihtiyacına göre
            st.markdown(
            """
            <style>
                /* Sidebar'ın en üst boşluğunu kaldır */
                section[data-testid="stSidebar"] > div:first-child {
                    padding-top: 0rem;
                    margin-top: -3rem;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
            st.sidebar.title("Ayarlar")
            # --- SESSION STATE INIT ---
            if "ui_n_sectors" not in st.session_state:
                st.session_state.ui_n_sectors = DEFAULT_SECTORS

            if "ui_n_rings" not in st.session_state:
                st.session_state.ui_n_rings = DEFAULT_RINGS

            if "ui_show_region" not in st.session_state:
                st.session_state.ui_show_region = DEFAULT_SHOW_REGION

            # --- RESET BUTTON ---
            def reset_params():
                st.session_state.ui_n_sectors = DEFAULT_SECTORS
                st.session_state.ui_n_rings   = DEFAULT_RINGS
                st.session_state.ui_show_region = DEFAULT_SHOW_REGION

            # --- WIDGETS ---
            n_sectors = st.slider("Dilim", 1, 10, st.session_state.ui_n_sectors, key="ui_n_sectors")
            n_rings   = st.slider("Halka",   1, 5, st.session_state.ui_n_rings,   key="ui_n_rings")
            st.button( "Reset", on_click=reset_params)
            show_region = st.checkbox("Bölge Analizi", st.session_state.ui_show_region, key="ui_show_region")


        # =========================
        # GİRİŞ: Upload / Klasör
        # =========================
        files = st.file_uploader("Görselleri yükle (tek/çoklu)", type=["jpg","jpeg","png"], accept_multiple_files=True)
        #folder = st.text_input("Veya klasör (opsiyonel)")

        paths = []
        #if folder and os.path.isdir(folder):
            #paths = sorted(sum([glob.glob(os.path.join(folder, p)) for p in ("*.jpg","*.jpeg","*.png")], []))

        if not files and not paths:
            st.info("Başlamak için görsel/ler yükle.")
            st.stop()

        decoded = []
        if files:
            for uf in files:
                data = np.frombuffer(uf.read(), np.uint8)
                decoded.append((uf.name, cv2.imdecode(data, cv2.IMREAD_COLOR)))
        if paths:
            for p in paths:
                decoded.append((os.path.basename(p), cv2.imread(p)))

        # =========================
        # MODEL: Eğit (K=5)
        # =========================
        if "km_model" not in st.session_state:
            st.session_state.km_model = None
            st.session_state.km_inv = None
            st.session_state.km_class_order = CLASS_NAMES[:]
            st.session_state.km_centers = None

        #st.markdown("#### 1) KMeans modeli")
        colA, colB = st.columns([1,2])

        with colA:
            if KMeans is None:
                st.error("`scikit-learn` gerekli: `pip install scikit-learn`")
        # train_btn = st.button("KMeans (k=5) eğit.")
            try:
                imgs_only = [im for _, im in decoded]
                km, order, inv, class_order, centers_sorted = fit_kmeans_5(imgs_only, build_pizza_mask_from_ranges, seed=42)
                st.session_state.km_model = km
                st.session_state.km_inv   = inv
                st.session_state.km_class_order = class_order
                st.session_state.km_centers = centers_sorted
                #st.success("Model eğitildi.")
            except Exception as e:
                    st.error("Eğitim hatası:"); st.exception(e)

        #with colB:
            #if st.session_state.km_centers is not None:
                #dfc = pd.DataFrame(st.session_state.km_centers, columns=["L","A0","B0"])
                #dfc["class"] = st.session_state.km_class_order
                #st.dataframe(dfc[["class","L","A0","B0"]], use_container_width=True)

        if st.session_state.km_model is None:
            st.warning("Önce **KMeans eğit** butonuna bas.")
            st.stop()

        # =========================
        # 2) İŞLEME (tam çöz.) + Isı Haritası
        # =========================
        rows = []
        grid = st.columns(2)


        for i, (name, img) in enumerate(decoded):
            mask = build_pizza_mask_from_ranges(img)
            class_idx, counts, perc, dominant = predict_kmeans_map(
                img, mask, st.session_state.km_model, st.session_state.km_inv, st.session_state.km_class_order
            )

            # Isı haritası (özel HEX palet ile)
            heat_over = heatmap_overlay(img, class_idx, st.session_state.km_class_order, alpha=0.6, cmap=CMAP5, norm=NORM5)

            col = grid[i % 2]
            with col:
                st.subheader(name)
                base = outline_on(img, mask) #if show_mask_outline else img

                fig = show_heatmap_figure(base, heat_over, cmap=CMAP5, norm=NORM5, bins=BINS5)
                st.pyplot(fig, clear_figure=True); plt.close(fig)

                # ısı haritası altına yüzdeleri tek satır yaz
                order = st.session_state.km_class_order
                st.markdown(
                    f"<div style='text-align:center; margin-top:-8px; margin-bottom:10px'>"
                    f"<b>Yüzdeler:</b> {pct_line(perc, order)}</div>",
                    unsafe_allow_html=True
                )

                # Yüzde barı (burnt dahil)
                labels = st.session_state.km_class_order
                vals = [perc[k]*100 for k in labels]
                colors =["#FF0000","#CCCC00","#FFFF66","#FFFFAD","#FF99FF"]  # dough, light_brown, brown, dark_brown, burnt
                # Sıralama: Dough -> Light Brown -> Brown -> Dark Brown -> Burnt
                display_order = ["Dough", "Light Brown", "Brown", "Dark Brown", "Burnt"]
                # Senin tanımladığın özel renk paleti
                display_colors = ["#FF99FF", "#FFFFB5", "#FFFF66", "#CCCC00", "#FF0000"]
                
                # Sözlükten (counts) listeye çevir
                counts_list = [counts.get(k, 0) for k in display_order]
                total = sum(counts_list) if sum(counts_list) > 0 else 1
                perc_list = [100.0 * c / total for c in counts_list]

                # --- GRAFİK: DONUT CHART (EXCEL TARZI OKLU) ---
                # figsize=(5, 3) ile kompakt tutuyoruz
                fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
                
                wedges, texts = ax.pie(
                    perc_list, 
                    colors=display_colors, 
                    startangle=90, 
                    counterclock=False, # Saat yönünde (Dough'dan Burnt'a)
                    #wedgeprops=dict(width=0.4, edgecolor="white", linewidth=1) # Donut halkası
                )
                
                # Etiket Kutusu ve Ok Ayarları
                bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=0.5)
                kw = dict(arrowprops=dict(arrowstyle="-", lw=0.5), bbox=bbox_props, zorder=0, va="center")

                for i, p in enumerate(wedges):
                    val = perc_list[i]
                    # %1.0'dan küçük dilimleri etiketleyip kalabalık yapmayalım
                    if val < 1.0: 
                        continue

                    # Açıyı hesapla
                    ang = (p.theta2 - p.theta1)/2. + p.theta1
                    y = np.sin(np.deg2rad(ang))
                    x = np.cos(np.deg2rad(ang))
                    
                    # Ok yönü
                    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                    kw["arrowprops"].update({"connectionstyle": f"angle,angleA=0,angleB={ang}"})
                    
                    # Etiket Metni (Sınıf Adı ve Yüzde)
                    # Dough gibi uzun isimler için gerekirse kısaltma yapılabilir ama şimdilik tam basıyoruz
                    ax.annotate(f"{display_order[i]}\n%{val:.1f}", 
                                xy=(x, y), 
                                xytext=(1.25*np.sign(x), 1.3*y), # Oku biraz dışarı aç
                                horizontalalignment=horizontalalignment, 
                                fontsize=8, # Yazı boyutu sabit
                                **kw)
                
                # use_container_width=False diyerek figsize'a sadık kalmasını sağlıyoruz
                st.pyplot(fig, clear_figure=True, use_container_width=False)
                plt.close(fig)

                # Altına sade bir metin özeti (okuması zor olanlar için)
                st.caption(
                    f"Hamur: %{perc_list[0]:.1f} | "
                    f"İdeal: %{perc_list[1]+perc_list[2]+perc_list[3]:.1f} | "
                    f"Yanık: %{perc_list[4]:.1f}"
                )
                st.divider()

                

                
                # --- Ek toplam yüzdeler ---
                burnt       = perc["Burnt"] * 100
                dough       = perc["Dough"] * 100
                pis = (perc["Dark Brown"] + perc["Brown"] + perc["Light Brown"]) * 100

                st.markdown(
                    f"<div style='text-align:center; margin-top:-10px; margin-bottom:15px'>"
                    f" <b>Undercooked:</b> {dough:.1f}%   |   "
                    f" <b>Cooked:</b> {pis:.1f}%   |   "
                    f" <b>Overcooked:</b> {burnt:.1f}%"
                    f"</div>",
                    unsafe_allow_html=True
                )

                row = {"file": name, "dominant": dominant}
                row.update({f"pct_{k}": round(perc[k]*100, 2) for k in st.session_state.km_class_order})
                rows.append(row)
                if st.session_state.ui_show_region:
                    fig_reg, uni = region_overlay_figure_kmeans(
                        img, class_idx, mask, st.session_state.km_class_order,
                        n_sectors=int(st.session_state.ui_n_sectors),
                        n_rings=int(st.session_state.ui_n_rings),
                        alpha=0.55
                    )
                    st.pyplot(fig_reg, clear_figure=True, bbox_inches = "tight"); plt.close(fig_reg)
                    st.caption(f"Homojenlik skoru: **{uni:.1f}/100**")
 
def run_borek():
    st.markdown(
        """
        <style>
        .block-container h1 { margin-top: -80px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ============ Yardımcılar ============
    def hex_to_bgr(hex_code):
        h = hex_code.lstrip("#")
        return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

    COLOR_MAP = {
        "#FFFEB5": "#818100", "#FEFF94": "#666732", "#FEFE7A": "#01FEFF", "#FFD15C": "#32CDFF",
        "#EDB256": "#FF99FF", "#E4A741": "#FF00FF", "#C38F49": "#FFFE67", "#B89057": "#CDCC01",
        "#996F3C": "#0167CC", "#916533": "#0101CC", "#845A37": "#01FF01", "#6C5033": "#01CC01",
        "#68533E": "#FE0001", "#404032": "#C10100"
    }
    SRC_BGR = np.array([hex_to_bgr(h) for h in COLOR_MAP.keys()], dtype=np.uint8)
    DST_BGR = np.array([hex_to_bgr(h) for h in COLOR_MAP.values()], dtype=np.uint8)
    SRC_LAB = cv2.cvtColor(SRC_BGR[np.newaxis,:,:], cv2.COLOR_BGR2LAB)[0]

    def simple_mask_white_bg(bgr, V_thresh=230, min_area=20000):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = (hsv[...,2] < V_thresh).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
        mask = np.zeros_like(m)
        if cnts: cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
        return mask

    def recolor_by_lab(img_bgr, mask):
        H, W = img_bgr.shape[:2]
        lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.int16)
        src16 = SRC_LAB.astype(np.int16)
        recolored = np.full((lab_img.shape[0], 3), 255, dtype=np.uint8)
        idx = np.where(mask.reshape(-1) > 0)[0]
        for i in idx:
            dist = np.linalg.norm(src16 - lab_img[i], axis=1)
            recolored[i] = DST_BGR[np.argmin(dist)]
        return recolored.reshape(H, W, 3)

    # --- Renk Tanımları ---
    RAW_BGR    = np.array([hex_to_bgr(c) for c in ["#818100", "#666732", "#01FEFF", "#32CDFF"]], dtype=np.uint8)
    COOKED_BGR = np.array([hex_to_bgr(c) for c in ["#FF99FF", "#FF00FF", "#FFFE67", "#CDCC01", "#01FF01", "#01CC01","#0167CC", "#0101CC"]], dtype=np.uint8)
    BURNT_BGR  = np.array([hex_to_bgr(c) for c in ["#FE0001", "#C10100"]], dtype=np.uint8)

    # --- UI Başlangıcı ---
    st.set_page_config(page_title="Pişme Analizi", layout="wide")
    st.title("Börek Analizi")

    uploads = st.file_uploader("Görsel Yükle", type=["jpg","png","jpeg"], accept_multiple_files=True)
    if uploads:
        for up in uploads:
            file_bytes = np.frombuffer(up.read(), np.uint8)
            bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            mask = simple_mask_white_bg(bgr)
            heat_bgr = recolor_by_lab(bgr, mask)

            c1, c2 = st.columns(2, gap="small")
            with c1: st.subheader("Orijinal"); st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with c2: st.subheader("Analiz"); st.image(cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Sayım
            flat = heat_bgr[mask > 0]
            def cnt(tgt):
                tot = 0
                for c in tgt: tot += np.count_nonzero(np.all(flat == c, axis=1))
                return tot
            
            counts = [cnt(RAW_BGR), cnt(COOKED_BGR), cnt(BURNT_BGR)]
            total = max(sum(counts), 1)
            perc = [100.0 * c / total for c in counts]
            
            labels = ["Undercooked", "Cooked", "Overcooked"]
            colors = ["#FFFF66", "#FF9900", "#C10100"] 

            # ==========================================
            #   EXCEL TARZI OKLU & DIŞARI TAŞAN GRAFİK
            # ==========================================
            # 1. FIGSIZE: (4, 2.5) yaparak fiziksel boyutu küçültüyoruz.
            # DPI: 100 standarttır, artırırsanız yazılar yine devleşebilir.
            # ==========================================
            #   WAFFLE CHART (KARE KARE ANALİZ)
            # ==========================================
            # Toplam 100 karelik bir tepsi hayal edelim (10x10 veya 5x20)
            # Kullanıcıya seçim şansı veriyoruz
            chart_type = st.radio(
                "Grafik Türünü Seç:",
                ["Pasta (Donut)", "Waffle (Tepsi Görünümü)", "Yatay Çubuk (Bar)"],
                horizontal=True,
                index=0
            )

            fig = None # Figure placeholder

            # -------------------------------------------------------
            # SEÇENEK 1: PASTA (DONUT) - Excel Tarzı
            # -------------------------------------------------------
            if chart_type == "Pasta (Donut)":
                fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100)
                wedges, texts = ax.pie(
                    counts, 
                    colors=colors, 
                    startangle=90, 
                    counterclock=False,
                    wedgeprops=dict(width=0.4, edgecolor="white", linewidth=1)
                )
                
                # Excel tarzı oklu etiketleme
                bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=0.5)
                kw = dict(arrowprops=dict(arrowstyle="-", lw=0.5), bbox=bbox_props, zorder=0, va="center")

                for i, p in enumerate(wedges):
                    val = perc[i]
                    if val < 1.0: continue # %1 altını etiketleme

                    ang = (p.theta2 - p.theta1)/2. + p.theta1
                    y = np.sin(np.deg2rad(ang))
                    x = np.cos(np.deg2rad(ang))
                    
                    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                    kw["arrowprops"].update({"connectionstyle": f"angle,angleA=0,angleB={ang}"})
                    
                    ax.annotate(f"{labels[i]}\n%{val:.1f}", xy=(x, y), xytext=(1.2*np.sign(x), 1.25*y),
                                horizontalalignment=horizontalalignment, fontsize=8, **kw)

            # -------------------------------------------------------
            # SEÇENEK 2: WAFFLE CHART - Kare Kare Analiz
            # -------------------------------------------------------
            elif chart_type == "Waffle (Tepsi Görünümü)":
                # Görsel olarak tam sayıya yuvarlanır (kare boyamak için)
                # Ama lejantta gerçek 'perc' değerini yazarız.
                total_squares = 100
                counts_per_class = [int(p) for p in perc]
                
                # Toplam 100 etmezse farkı en büyük sınıfa ekle
                diff = total_squares - sum(counts_per_class)
                counts_per_class[np.argmax(counts_per_class)] += diff
                
                # Matrisi oluştur
                waffle_grid = []
                for class_id, count in enumerate(counts_per_class):
                    waffle_grid.extend([class_id] * count)
                waffle_arr = np.array(waffle_grid).reshape((10, 10))
                
                cmap_waffle = plt.cm.colors.ListedColormap(colors)
                
                fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
                ax.matshow(waffle_arr, cmap=cmap_waffle, vmin=0, vmax=2)
                
                # Izgara görünümü
                ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
                ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
                ax.set_xticks([]); ax.set_yticks([])
                
                # Lejant (Burada gerçek ondalıklı değerleri kullanıyoruz)
                
                legend_elements = [
                    Patch(facecolor=colors[i], edgecolor='w', label=f'{labels[i]} (%{perc[i]:.1f})')
                    for i in range(3)
                ]
                ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                          ncol=3, frameon=False, fontsize=9)
            
            # -------------------------------------------------------
            # SEÇENEK 3: YATAY ÇUBUK (BAR) - Minimalist
            # -------------------------------------------------------
            else:
                fig, ax = plt.subplots(figsize=(6, 1.5), dpi=100)
                # Yığılmış çubuk
                left_pos = 0
                for i in range(3):
                    ax.barh(0, perc[i], left=left_pos, color=colors[i], edgecolor="white", height=0.6, label=labels[i])
                    
                    # Dilim büyükse içine yaz
                    if perc[i] > 5:
                        ax.text(left_pos + perc[i]/2, 0, f"%{perc[i]:.1f}", 
                                ha='center', va='center', color='black', fontsize=9, fontweight='bold')
                    left_pos += perc[i]

                ax.axis('off')
                ax.set_xlim(0, 100)
                ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=3, frameon=False, fontsize=9)

            # --- Çizim ve Temizlik ---
            st.pyplot(fig, clear_figure=True, use_container_width=False)
            plt.close(fig)
            
            # Ekstra metin özeti (Her zaman görünür)
            st.caption(f"Detaylı Oranlar: Çiğ %{perc[0]:.2f} | Pişmiş %{perc[1]:.2f} | Yanık %{perc[2]:.2f}")
            st.divider()

    else:
        st.info("Başlamak için görsel/ler yükle.")


def run_smallcake():

    

    # ---------- Sayfa ----------
    st.set_page_config(page_title="Pişme Analizi", layout="wide")
    st.markdown(
        "<style>.block-container h1{margin-top:-80px}</style>",
        unsafe_allow_html=True
    )
    st.title("Small Cake Analizi")

    # ---------- Yardımcılar ----------
    def hex_to_bgr(h):
        h = h.lstrip("#")
        return np.array([int(h[4:6],16), int(h[2:4],16), int(h[0:2],16)], dtype=np.uint8)

    def simple_mask_white_bg_multi(bgr, V_thresh=232, min_area=5000):
        """Beyaz fonda tüm keklerin kaba maskesi + kontürler."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        V = hsv[...,2]
        m = (V < V_thresh).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(m); kept=[]
        for c in cnts:
            if cv2.contourArea(c) >= min_area:
                kept.append(c)
                cv2.drawContours(mask, [c], -1, 255, -1)
        return mask, kept

    def remove_white_and_shadow_rim(bgr, mask, band=8, V_white=212, S_shadow=33, C_shadow=10):
        """Kenar bandındaki beyaz (V yüksek) VEYA gri gölge (S ve C_ab düşük) pikselleri maskeden çıkar."""
        mask = (mask>0).astype(np.uint8)*255
        if band<=0: return mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band*2+1, band*2+1))
        inner = cv2.erode(mask, k, 1)
        ring  = cv2.subtract(mask, inner)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        S, V = hsv[...,1], hsv[...,2]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        A = lab[...,1].astype(np.float32); B = lab[...,2].astype(np.float32)
        C = np.sqrt((A-128.0)**2 + (B-128.0)**2)
        cut = (ring>0) & ( (V>=V_white) | ((S<=S_shadow)&(C<=C_shadow)) )
        out = mask.copy(); out[cut]=0
        return out

    # ---------- Renk skalası (verdiğin eşleme) ----------
    COLOR_MAP = {
        "#FFFEB5": "#818100", "#FEFF94": "#666732", "#FEFE7A": "#01FEFF", "#FFD15C": "#32CDFF",
        "#EDB256": "#FF99FF", "#E4A741": "#FF00FF", "#C38F49": "#FFFE67", "#B89057": "#CDCC01",
        "#996F3C": "#0167CC", "#916533": "#0101CC", "#845A37": "#01FF01", "#6C5033": "#01CC01",
        "#68533E": "#FE0001", "#404032": "#C10100"
    }
    SRC_BGR = np.array([hex_to_bgr(h) for h in COLOR_MAP.keys()], dtype=np.uint8)
    DST_BGR = np.array([hex_to_bgr(h) for h in COLOR_MAP.values()], dtype=np.uint8)
    SRC_LAB = cv2.cvtColor(SRC_BGR[np.newaxis,:,:], cv2.COLOR_BGR2LAB)[0]

    def recolor_by_lab(img_bgr, mask):
        """LAB ΔE ile hedef skalaya boyama (arka plan beyaz)."""
        H, W = img_bgr.shape[:2]
        lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.int16)
        mask_flat = mask.reshape(-1)
        out = np.full((H*W, 3), 255, dtype=np.uint8)
        src16 = SRC_LAB.astype(np.int16)
        for i in np.where(mask_flat>0)[0]:
            pix = lab_img[i]
            j = np.argmin(np.linalg.norm(src16 - pix, axis=1))
            out[i] = DST_BGR[j]
        return out.reshape(H, W, 3)

    # ---------- IEC 13 bölge maskeleri ----------
    def smallcake_region_masks(img_shape, center, r1, r2_ratio=0.6, r3_ratio=0.3):
        H,W = img_shape[:2]
        cx, cy = map(int, center)
        r1 = int(r1); r2 = int(r2_ratio*r1); r3 = int(r3_ratio*r1)
        regs=[]; blank=lambda: np.zeros((H,W),np.uint8)
        mC=blank(); cv2.circle(mC,(cx,cy),r3,255,-1); regs.append(("C",mC))
        for k in range(4):
            m=blank(); cv2.ellipse(m,(cx,cy),(r2,r2),0,k*90,(k+1)*90,255,-1); cv2.circle(m,(cx,cy),r3,0,-1)
            regs.append((f"M{k+1}",m))
        for k in range(8):
            m=blank(); cv2.ellipse(m,(cx,cy),(r1,r1),0,k*45,(k+1)*45,255,-1); cv2.circle(m,(cx,cy),r2,0,-1)
            regs.append((f"O{k+1}",m))
        return regs, r1, r2, r3

    # ---------- (YENİ) Bölgeyi COLOR_MAP hedef rengine göre boyama ----------
    def region_dominant_dst_color(heat_bgr, idx, dst_palette):
        """Bu parçada heatmap'te en çok görünen hedef rengi (DST_BGR) ve yüzdesini döndür."""
        if not np.any(idx):
            return (255,255,255), 0.0
        flat = heat_bgr[idx]
        counts = [np.count_nonzero(np.all(flat==c, axis=1)) for c in dst_palette]
        counts = np.asarray(counts, np.int32)
        j = int(np.argmax(counts))
        tot = int(np.sum(counts)) or 1
        pct = 100.0 * counts[j] / tot
        return tuple(int(v) for v in dst_palette[j]), pct

    def paint_all_cakes_overlay(bgr, mask, heat_bgr,
                                r2_ratio=0.6, r3_ratio=0.3,
                                alpha=0.45,
                                line_color=(0,0,0), line_thick=1):
        """
        Orijinal görüntü üzerine (sadece mask içinde) 13 parçayı,
        COLOR_MAP hedef rengine göre boyar ve yüzdeleri yazar.
        """
        H, W = bgr.shape[:2]
        base = bgr.copy()
        overlay = base.copy()  # bütün kekler bu katmanda çizilecek

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            (cx, cy), r1 = cv2.minEnclosingCircle(c)
            regions, r1, r2, r3 = smallcake_region_masks(bgr.shape, (cx,cy), r1, r2_ratio, r3_ratio)

            # Her bölge: sadece kek içinde, baskın DST rengine göre boya
            for name, m in regions:
                idx = (m > 0) & (mask > 0)
                color_bgr, pct = region_dominant_dst_color(heat_bgr, idx, DST_BGR)
                overlay[idx] = color_bgr

                # yüzde yazısı (kek içinde ise)
                M = cv2.moments(m, binaryImage=True)
                if M["m00"] > 0:
                    x = int(M["m10"]/M["m00"]); y = int(M["m01"]/M["m00"])
                    if mask[y, x] > 0:
                        # okunabilirlik için ince beyaz gölge + siyah metin
                        cv2.putText(overlay, f"{pct:.0f}%", (x-12, y+4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255,255,255), 3, cv2.LINE_AA)
                        cv2.putText(overlay, f"{pct:.0f}%", (x-12, y+4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0,0,0), 1, cv2.LINE_AA)

            # ince şablon çizgileri
            cv2.circle(overlay,(int(cx),int(cy)),int(r1),line_color,line_thick)
            cv2.circle(overlay,(int(cx),int(cy)),int(r2),line_color,line_thick)
            cv2.circle(overlay,(int(cx),int(cy)),int(r3),line_color,line_thick)
            for ang in range(0,360,90):
                a = np.deg2rad(ang)
                p0 = (int(cx+r3*np.cos(a)), int(cy+r3*np.sin(a)))
                p1 = (int(cx+r2*np.cos(a)), int(cy+r2*np.sin(a)))
                cv2.line(overlay, p0, p1, line_color, line_thick)
            for ang in range(0,360,45):
                a = np.deg2rad(ang)
                p0 = (int(cx+r2*np.cos(a)), int(cy+r2*np.sin(a)))
                p1 = (int(cx+r1*np.cos(a)), int(cy+r1*np.sin(a)))
                cv2.line(overlay, p0, p1, line_color, line_thick)

        # Tek seferde alpha blend ve sadece mask içinde uygula
        blended = cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0)
        idx = (mask > 0)         # 2B maske
        out = base.copy()
        out[idx] = blended[idx]
        return out

    # ---------- Upload ---------

        # 1) Maske (kalıp & gölge kenarını çıkar)
        coarse, _ = simple_mask_white_bg_multi(bgr, V_thresh=232, min_area=5000)
        mask = remove_white_and_shadow_rim(bgr, coarse, band=8, V_white=212, S_shadow=33, C_shadow=10)

        # 2) Heatmap (COLOR_MAP hedef renklerine yeniden renklendirme)
        heat = recolor_by_lab(bgr, mask)

        # 3) Tüm kekler tek canvas: overlay + yüzdeler (renkler COLOR_MAP'ten)
        region_vis = paint_all_cakes_overlay(bgr, mask, heat, r2_ratio=0.6, r3_ratio=0.3, alpha=0.45)

        # ---- Küçük görseller ----
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.markdown("Orijinal")
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), width=72)
        with c2:
            st.markdown("Pişme Analizi")
            st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), width=720)
        with c3:
            st.markdown("Bölgesel Analiz ")
            st.image(cv2.cvtColor(region_vis, cv2.COLOR_BGR2RGB), width=720)

        st.divider()





    # ---------- Upload ----------
    uploads = st.file_uploader("Görselleri yükle (tek/çoklu)", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if not uploads:
        st.info("Başlamak için görsel/ler yükle."); return

    for up in uploads:
        st.subheader(f" {up.name}")
        bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None: st.error("Görsel okunamadı."); continue

        # 1) Maske (kalıp & gölge kenarını çıkar)
        coarse, _ = simple_mask_white_bg_multi(bgr, V_thresh=232, min_area=5000)
        mask = remove_white_and_shadow_rim(bgr, coarse, band=8, V_white=212, S_shadow=33, C_shadow=10)

        # 2) Heatmap
        heat = recolor_by_lab(bgr, mask)

        # 3) Tek canvas üzerinde bölgesel boyama
        region_vis = paint_all_cakes_overlay(bgr, mask, heat, r2_ratio=0.6, r3_ratio=0.3, alpha=0.45)

        # ---- Küçük görseller (ekranı kaplamasın) ----
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.markdown("Orijinal")
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), width=1080)
        with c2:
            st.markdown("Pişme Analizi")
            st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), width=1080)
        with c3:
            st.markdown("Bölgesel Analiz")
            st.image(cv2.cvtColor(region_vis, cv2.COLOR_BGR2RGB), width=1080)
        # --- Sidebar Reset Button ---
        if st.sidebar.button("Reset"):
            # file_uploader için verdiğin key neyse onu kullan
            st.session_state["uploads"] = None


        st.divider()


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

 














