# ============================================================
# 🏍️ MOTORBIKE RECOMMENDATION DASHBOARD (Pandora Blue – Dark Mode + SEO Ready)
# ============================================================
# Author: Hai Nguyen & Chau Le
# Version: v9-SEO-Full – JSON-LD, OpenGraph, Cache Optimized, Full Tabs
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os, joblib, pickle, chardet
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
from recommender import load_clean_data, recommend_hybrid
import random, re
from datetime import datetime, timedelta

# ============================================================
# 🧭 PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🏍️ Ứng dụng Gợi ý & Định giá Xe Máy Cũ",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 🌐 SEO & SOCIAL META TAGS
# ============================================================
st.markdown("""
<title>Motorbike Recommender – Gợi ý & Định giá Xe Máy Cũ thông minh</title>
<meta name="description" content="Công cụ gợi ý & định giá xe máy cũ bằng AI. Tìm xe tương tự, xem giá thị trường, và phân tích xu hướng xe máy 2025.">
<meta name="keywords" content="xe máy cũ, định giá xe máy, mua bán xe, Honda Vision, Yamaha, Air Blade, xe ga, xe số, giá xe máy 2025">
<meta name="robots" content="index, follow">

<!-- Open Graph -->
<meta property="og:title" content="Motorbike Recommender – Gợi ý & Định giá Xe Máy Cũ thông minh">
<meta property="og:description" content="Xem giá thị trường thực tế, gợi ý xe tương tự bằng AI, cập nhật giá xe 2025.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://your-app-domain/">
<meta property="og:image" content="https://your-app-domain/preview.png">

<!-- Twitter Card -->
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Motorbike Recommender – Định giá Xe Máy Cũ thông minh">
<meta name="twitter:description" content="So sánh giá xe cũ, gợi ý mua xe thông minh với AI.">
<meta name="twitter:image" content="https://your-app-domain/preview.png">
""", unsafe_allow_html=True)

# ============================================================
# 🎨 DARK MODE STYLE
# ============================================================
st.markdown("""
<style>
html, body, .stApp, .main, .block-container {
    background-color: #0F172A !important;
    color: #E2E8F0 !important;
    font-family: 'Segoe UI', sans-serif;
}
aside[data-testid="stSidebar"] {
    background-color: #1E293B !important;
    color: #E2E8F0 !important;
    border-right: 1px solid #334155 !important;
}
h1, h2, h3, h4 { color: #93C5FD !important; font-family: 'Segoe UI Semibold', sans-serif; }
.stButton>button { background-color: #2563EB !important; color: #F8FAFC !important; border-radius: 8px !important; }
.stTextInput>div>div>input { background-color: #1E293B !important; color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
h3, .stPlotlyChart {
    margin-top: 10px !important;
    margin-bottom: 40px !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🔧 LOAD MODELS & CACHE
# ============================================================
@st.cache_resource
def load_all_models():
    data_path = "data/motorbike_final_dataset_clean.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu: {data_path}")

    with open(data_path, 'rb') as f:
        enc = chardet.detect(f.read(200000))['encoding']
    df = pd.read_csv(data_path, encoding=enc)

    dictionary = corpora.Dictionary.load("model/dictionary.dict")
    tfidf_model = models.TfidfModel.load("model/tfidf_gensim.model")
    index = similarities.MatrixSimilarity.load("model/tfidf_index.index")
    texts = joblib.load("model/texts.pkl")

    if os.path.exists("model/w2v_model.model"):
        model_w2v = Word2Vec.load("model/w2v_model.model")
    elif os.path.exists("model/w2v_model.pkl"):
        with open("model/w2v_model.pkl", "rb") as f:
            model_w2v = pickle.load(f)
    else:
        raise FileNotFoundError("❌ Không tìm thấy mô hình Word2Vec (.model hoặc .pkl)")

    return df, dictionary, tfidf_model, index, texts, model_w2v

# ============================================================
# LOAD CLUSTERING RESULT
# ============================================================

@st.cache_data
def load_clustered_data():
    df_clustered = pd.read_excel("output_cluster/meta_gmm_full.xlsx")
    return df_clustered

df_clustered = load_clustered_data()

# ============================================================
# 🧠 AUTO INSIGHT GENERATOR 
# ============================================================

def generate_auto_insight(model_name, usage_status, user_mode, user_id, df_clustered,
                          price_mean, price_min, price_max, good_price,
                          as_card=True):
    """
    Sinh insight tự động (có style đẹp, đồng bộ giá cụm & đổi màu nền theo theme)
    """

    import streamlit as st
    import pandas as pd

    # --- 1️⃣ MAPPING PHÂN KHÚC CƠ BẢN ---
    cluster_segments = {
        0: {"name": "phổ thông – giá thấp (10–15 triệu)",
            "traits": "xe tiết kiệm, chi phí thấp, dễ bảo dưỡng, phù hợp người mới đi làm hoặc sinh viên",
            "emoji": "🟢", "color": "#22C55E"},
        1: {"name": "tầm trung – giá 15–25 triệu",
            "traits": "xe phổ biến, thương hiệu mạnh, giữ giá tốt, được ưa chuộng khi mua lại",
            "emoji": "🔵", "color": "#3B82F6"},
        2: {"name": "cao cấp – trên 25 triệu",
            "traits": "xe đời mới, ít sử dụng, có trang bị tiện ích và độ bền cao",
            "emoji": "🟣", "color": "#8B5CF6"},
        3: {"name": "cao cấp đặc biệt – trên 40 triệu",
            "traits": "dòng xe sang, hướng đến người dùng yêu cầu chất lượng và thương hiệu",
            "emoji": "🟡", "color": "#FACC15"}
    }

    # --- 2️⃣ XÁC ĐỊNH CỘT CỤM & CỤM NGƯỜI DÙNG ---
    for col in ["meta_gmm", "meta_cluster", "cluster_gmm", "cluster_kmeans"]:
        if col in df_clustered.columns:
            cluster_col = col
            break
    else:
        raise KeyError("Không tìm thấy cột cụm trong df_clustered.")

    user_cluster = int(df_clustered.loc[df_clustered["id"] == user_id, cluster_col].values[0])

    # --- 3️⃣ XÁC ĐỊNH PHÂN KHÚC THEO GIÁ TRUNG BÌNH ---
    # Ưu tiên dùng cột giá đã quy đổi (Giá_tb_số / Giá_clean)
    price_col = None
    for c in ["Giá_tb_số", "Giá_clean"]:
        if c in df_clustered.columns:
            price_col = c
            break

    if price_col:
        cluster_price_map = (
            df_clustered.groupby(cluster_col)[price_col]
            .mean()
            .sort_values()
            .reset_index()
        )
        cluster_price_map["rank"] = range(len(cluster_price_map))
        price_to_segment = dict(zip(cluster_price_map[cluster_col], cluster_price_map["rank"]))
        user_cluster_rank = price_to_segment.get(user_cluster, 1)
        segment_info = cluster_segments.get(user_cluster_rank, cluster_segments[1])
    else:
        # fallback nếu không có giá
        segment_info = cluster_segments.get(user_cluster, cluster_segments[1])

    # --- 4️⃣ LÀM TRÒN GIÁ ---
    def smart_round(x): return round(x, 1) if x < 10 else round(x)
    avg_price, min_price, max_price, good_price = map(
        smart_round, [price_mean, price_min, price_max, good_price]
    )

    # --- 5️⃣ NỘI DUNG INSIGHT ---
    intro = (
        f"💡 Xe tương tự <b>{model_name} ({usage_status})</b> – "
        f"giá trung bình <b>{avg_price} triệu</b>, dao động <b>{min_price}–{max_price} triệu</b>.<br><br>"
    )
    deal = (
        f"✨ Nếu bạn {'tìm được' if user_mode=='buyer' else 'rao bán ở'} "
        f"mức <b>{good_price} triệu</b>, đó là mức <b>rất tốt!</b><br><br>"
    )
    segment = (
        f"🚗 Xe {'bạn đang xem' if user_mode=='buyer' else 'của bạn'} "
        f"thuộc <b>{segment_info['emoji']} phân khúc {segment_info['name']}</b>, "
        f"thường được người mua chọn vì <b>{segment_info['traits']}</b>."
    )

    # --- 6️⃣ STYLE HIỂN THỊ (TỰ ĐỔI MÀU THEO THEME) ---
    theme_base = st.get_option("theme.base")
    is_dark = theme_base == "dark"

    bg_color = "#1E1E1E" if is_dark else "#F9FAFB"
    text_color = "#F9FAFB" if is_dark else "#1B1E23"
    border_color = segment_info["color"]

    html_block = f"""
    <div style="
        background-color:{bg_color};
        color:{text_color};
        border-left:6px solid {border_color};
        border-radius:14px;
        padding:18px 22px;
        margin-top:12px;
        box-shadow:0 2px 8px rgba(0,0,0,0.15);
        line-height:1.7;
        font-size:16px;
    ">
    {intro}{deal}{segment}
    </div>
    """

    if as_card:
        st.markdown(html_block, unsafe_allow_html=True)
    else:
        return intro + "\n\n" + deal + "\n\n" + segment



# ============================================================
# 🧭 SIDEBAR MENU
# ============================================================
menu = st.sidebar.radio(
    "Chọn chức năng:",
    ["📘 Giới thiệu ứng dụng", "💰 Gợi ý & định giá xe", "🧠 Phân tích mô hình"],
    index=0
)




# ============================================================
# 1️⃣ GIỚI THIỆU ỨNG DỤNG
# ============================================================
if menu == "📘 Giới thiệu ứng dụng":
    st.header("🏍️ Ứng dụng Gợi ý & Phân tích xe máy cũ")
    st.markdown("""
    ### 🎯 Mục tiêu
    - Gợi ý xe tương tự giúp người mua tham khảo dễ dàng.
    - Đưa ra mức giá hợp lý giúp người bán điều chỉnh chính xác hơn.
    - Hỗ trợ nghiên cứu xu hướng thị trường bằng phân tích cụm.

    ### ⚙️ Công nghệ
    - **TF-IDF + Word2Vec (Hybrid)**
    - **KMeans / UMAP / PCA / Silhouette**
    - **Streamlit – Pandora Blue Dark Mode**
    """)
    # st.markdown("<br><small><i>Designed by Hai Nguyen & Chau Le – 29/11/2025</i></small>", unsafe_allow_html=True)

# ============================================================
# 2️⃣ GỢI Ý & ĐỊNH GIÁ XE – SEO READY
# ============================================================
elif menu == "💰 Gợi ý & định giá xe":
    st.header("💰 Gợi ý & Định giá xe")

    role = st.radio("Chọn vai trò của bạn:", ["Tôi muốn mua xe", "Tôi muốn bán xe"], horizontal=True)
    query = st.text_input("Nhập mô tả xe (vd: Honda Vision 2019 màu đỏ 22 triệu):")
    k = st.slider("Số lượng xe hiển thị", 6, 18, 9)

    if st.button("🚀 Tìm xe tương tự"):
        st.session_state.setdefault("model_loaded", False)
        if not st.session_state["model_loaded"]:
            df, dictionary, tfidf_model, index, texts, model_w2v = load_all_models()
            st.session_state["model_loaded"] = True
        else:
            df, dictionary, tfidf_model, index, texts, model_w2v = load_all_models()

        results = recommend_hybrid(query, df, dictionary, tfidf_model, index, model_w2v, texts, final_k=k)

        # --- Join lại với data gốc ---
        df_raw = pd.read_excel("data/data_motorbikes.xlsx")
        if "id" not in df_raw.columns:
            df_raw.insert(0, "id", range(1, len(df_raw) + 1))

        results_full = pd.merge(results, df_raw, on="id", how="left", suffixes=("_rec", "_raw"))
        for col in ["Tiêu đề", "Giá", "Thương hiệu", "Dòng xe", "Loại xe", "Dung tích xe"]:
            raw_col, rec_col = f"{col}_raw", f"{col}_rec"
            if raw_col in results_full.columns:
                results_full[col] = results_full[raw_col]
            elif rec_col in results_full.columns:
                results_full[col] = results_full[rec_col]
        results_full = results_full[[c for c in results_full.columns if not c.endswith(("_rec", "_raw"))]]
        results_full = results_full.loc[:, ~results_full.columns.duplicated()]
        st.success(f"✅ Đã tìm thấy {len(results_full)} xe tương tự !")

        # ========== CARD SEO LAYOUT ==========
        st.markdown("""
        <style>
        .bike-card {
            background-color: #1E293B;
            border-radius: 18px;
            padding: 22px;
            margin-bottom: 35px;
            box-shadow: 0 0 20px rgba(37,99,235,0.25);
            transition: all 0.25s ease-in-out;
            min-height: 260px;
        }
        .bike-card:hover { transform: translateY(-6px); box-shadow: 0 0 35px rgba(37,99,235,0.5); }
        .bike-header { color:#BFDBFE; font-weight:700; font-size:19px; line-height:1.4em; margin-bottom:8px; text-transform:capitalize; }
        .bike-price { color:#FACC15; font-size:17px; font-weight:600; margin-bottom:4px; }
        .bike-meta { color:#CBD5E1; font-size:14px; margin-bottom:6px; }
        .bike-desc { color:#94A3B8; font-size:13px; margin-top:8px; }
        .bike-link { color:#60A5FA; text-decoration:none; font-weight:500; font-size:13px; }
        </style>
        """, unsafe_allow_html=True)

        num_cols = 2
        for i in range(0, len(results_full), num_cols):
            cols = st.columns(num_cols, gap="large")
            for j, col in enumerate(cols):
                if i + j < len(results_full):
                    r = results_full.iloc[i + j]
                    href = r.get("Href", "#")
                    desc = r.get("Mô tả chi tiết", "Không có mô tả chi tiết")
                    days_ago = random.randint(1, 15)
                    date_str = f"{days_ago} ngày trước"

                    col.markdown(f"""
                    <div class="bike-card">
                        <div style="font-size:13px;color:#CBD5E1;text-align:right;">📅 {date_str}</div>
                        <h3 class="bike-header">{r.get('Tiêu đề', 'Không có tiêu đề')}</h3>
                        <meta name="description" content="{desc[:150]}">
                        <script type="application/ld+json">
                        {{
                          "@context": "https://schema.org/",
                          "@type": "Product",
                          "name": "{r.get('Tiêu đề','')}",
                          "brand": "{r.get('Thương hiệu','')}",
                          "model": "{r.get('Dòng xe','')}",
                          "description": "{desc[:150]}",
                          "offers": {{
                              "@type": "Offer",
                              "price": "{r.get('Giá','')}",
                              "priceCurrency": "VND",
                              "availability": "https://schema.org/InStock",
                              "url": "{href}"
                          }}
                        }}
                        </script>
                        <div class="bike-price">💰 {r.get("Giá","Đang cập nhật")}</div>
                        <div class="bike-meta">{r.get('Thương hiệu','')} – {r.get('Dòng xe','')} | {r.get('Loại xe','')} | {r.get('Năm đăng ký','')}</div>
                        <a href="{href}" target="_blank" class="bike-link">🔗 Xem bài đăng</a>
                        <div class="bike-desc">{desc[:160]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom:80px;'></div>", unsafe_allow_html=True)

        # ============================================================
        # 📋 BẢNG CHI TIẾT & INSIGHT GIÁ
        # ============================================================
        with st.expander("📋 Xem bảng chi tiết"):
            cols_display = [
                "id", "Tiêu đề", "Giá", "Khoảng giá min", "Khoảng giá max", "Địa chỉ",
                "Mô tả chi tiết", "Thương hiệu", "Dòng xe", "Năm đăng ký", "Số Km đã đi",
                "Tình trạng", "Loại xe", "Dung tích xe", "Xuất xứ",
                "Chính sách bảo hành", "Trọng lượng", "Href"
            ]
            st.dataframe(results_full[[c for c in cols_display if c in results_full.columns]].astype(str),width='stretch')
        def clean_price_vnd(val):
            if pd.isnull(val): return np.nan
            val = str(val).replace(",", ".").lower()
            try:
                if "tr" in val or "triệu" in val:
                    num = re.findall(r"[\d.]+", val)
                    return float(num[0]) * 1_000_000 if num else np.nan
                elif "đ" in val:
                    num = re.findall(r"[\d.]+", val)
                    return float(num[0])
                else:
                    return float(val)
            except:
                return np.nan

        if {"Khoảng giá min", "Khoảng giá max"}.issubset(results_full.columns):
            results_full["Giá_min_số"] = results_full["Khoảng giá min"].apply(clean_price_vnd)
            results_full["Giá_max_số"] = results_full["Khoảng giá max"].apply(clean_price_vnd)
            results_full["Giá_tb_số"] = results_full[["Giá_min_số", "Giá_max_số"]].mean(axis=1)
        else:
            results_full["Giá_tb_số"] = results_full["Giá"].apply(clean_price_vnd)

        valid_prices = results_full["Giá_tb_số"].dropna()
        if len(valid_prices) > 0:
            avg_price = valid_prices.mean() / 1_000_000
            min_price = valid_prices.min() / 1_000_000
            max_price = valid_prices.max() / 1_000_000
            brand = results_full["Thương hiệu"].mode()[0]
            model = results_full["Dòng xe"].mode()[0]
            seg = results_full["Tình trạng"].mode()[0]

            model_name = f"{brand.title()} {model.title()}"
            usage_status = seg.lower()

            user_mode = "seller" if role == "Tôi muốn bán xe" else "buyer"

            # Tìm id tương ứng trong file phân cụm
            try:
                user_id = int(
                    df_clustered.loc[
                        (df_clustered["Thương hiệu"].str.lower() == brand.lower()) &
                        (df_clustered["Dòng xe"].str.lower() == model.lower())
                    ]["id"].values[0]
                )
            except:
                user_id = 0  # fallback nếu không tìm thấy

            generate_auto_insight(
                model_name=model_name,
                usage_status=usage_status,
                user_mode=user_mode,
                user_id=user_id,
                df_clustered=df_clustered,
                price_mean=avg_price,
                price_min=min_price,
                price_max=max_price,
                good_price=avg_price*0.9,
                as_card=True  # 👈 quan trọng
            )

    # st.markdown("<br><small><i>Designed by Hai Nguyen & Chau Le – 29/11/2025</i></small>", unsafe_allow_html=True)
# ============================================================
# 3️⃣ PHÂN TÍCH MÔ HÌNH (FULL GIỮ NGUYÊN)
# ============================================================
else:
    st.header("🧠 Phân tích mô hình")

    st.subheader("📈 Phân tích Meta Segmentation (GMM)")

    # Hiển thị Scatter + Silhouette song song
    col1, col2 = st.columns(2)
    with col1:
        st.image("output_cluster/meta_gmm_scatter.png", caption="Meta Segmentation – PCA 2D (GMM)")
    with col2:
        st.image("output_cluster/meta_gmm_silhouette.png", caption="Silhouette Plot – Meta GMM")

    # Biểu đồ phụ
    st.image("output_cluster/meta_gmm_cluster_size.png", caption="Phân bố số lượng mẫu theo cụm Meta GMM")
    st.image("output_cluster/meta_gmm_boxplot_price.png", caption="Phân bố giá xe theo cụm Meta GMM (triệu VND)")

    import plotly.express as px
    import plotly.graph_objects as go

    # ============================================================
    # 🏷️ Thống kê theo Thương hiệu
    # ============================================================
    st.subheader("🏷️ Thống kê theo Thương hiệu")

    df_brand = pd.read_excel("output_cluster/meta_gmm_brand_summary.xlsx")

    # Lấy top 15 thương hiệu có giá TB cao nhất
    df_brand = df_brand.sort_values("Giá TB (triệu VND)", ascending=False).head(15)

    fig_brand = px.bar(
        df_brand,
        x="Giá TB (triệu VND)",
        y="Thương hiệu",
        orientation="h",
        text="Giá TB (triệu VND)",
        color="Giá TB (triệu VND)",
        color_continuous_scale="Blues",
        title="Giá trung bình theo Thương hiệu (Top 15)",
    )
    fig_brand.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_brand.update_layout(
        showlegend=False,
        height=500,
        margin=dict(l=80, r=40, t=60, b=40),
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
    )
    st.plotly_chart(fig_brand, use_container_width=True)

    # ============================================================
    # ⚙️ Thống kê theo Loại xe
    # ============================================================
    st.subheader("⚙️ Thống kê theo Loại xe")

    df_type = pd.read_excel("output_cluster/meta_gmm_type_summary.xlsx")
    df_type = df_type.sort_values("Giá TB (triệu VND)", ascending=False)

    fig_type = px.bar(
        df_type,
        x="Loại xe",
        y="Giá TB (triệu VND)",
        color="Giá TB (triệu VND)",
        color_continuous_scale="Viridis",
        text="Giá TB (triệu VND)",
        title="Giá trung bình theo Loại xe",
    )
    fig_type.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_type.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=40, r=40, t=60, b=60),
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
    )
    st.plotly_chart(fig_type, use_container_width=True)

    # ============================================================
    # 💨 Thống kê theo Phân khúc dung tích
    # ============================================================
    st.subheader("💨 Thống kê theo Phân khúc dung tích")

    df_cc = pd.read_excel("output_cluster/meta_gmm_cc_summary.xlsx")
    df_cc = df_cc.sort_values("Giá TB (triệu VND)", ascending=False)

    # Biểu đồ tròn + màu pastel dễ nhìn
    fig_cc = px.pie(
        df_cc,
        values="Số lượng",
        names="Phan_khuc_dung_tich" if "Phan_khuc_dung_tich" in df_cc.columns else df_cc.index,
        color_discrete_sequence=px.colors.sequential.Tealgrn,
        title="Phân bố số lượng xe theo Phân khúc dung tích",
    )
    fig_cc.update_traces(textinfo="percent+label", pull=[0.05]*len(df_cc))
    fig_cc.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
    )
    st.plotly_chart(fig_cc, use_container_width=True)


    # st.markdown("<br><small><i>Designed by Hai Nguyen & Chau Le – 29/11/2025</i></small>", unsafe_allow_html=True)




from datetime import datetime
today = datetime.now().strftime("%d/%m/%Y %H:%M")

footer_html = f"""
<hr style="margin-top:25px; margin-bottom:8px; border:0; border-top:1px solid rgba(255,255,255,0.15);">

<div style="
    text-align:center;
    font-size:13px;
    font-weight:400;
    line-height:1.8;
    letter-spacing:0.3px;
    color:#FFFFFF;
    text-transform:none;
    font-family:'Segoe UI', Arial, sans-serif;
">
    © {datetime.now().year} <span style="font-weight:500;">Hai Nguyen</span> & 
    <span style="font-weight:500;">Chau Le</span>. All rights reserved.<br>
    <small>Version 1.0 – Prototype for research & demo use</small><br>
    <small style="font-size:11px; opacity:0.9;">Last updated: {today}</small>
</div>
"""

with st.container():
    st.markdown(footer_html, unsafe_allow_html=True)
