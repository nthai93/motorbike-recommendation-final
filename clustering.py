# ============================================================
# clustering.py – Full Pipeline with Colored Logging, Cached Models & Meta GMM Charts
# ============================================================

import os
import re
import time
import joblib
import unicodedata
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from colorama import Fore, Style, init
from underthesea import word_tokenize
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# ============================================================
# INIT & SETTINGS
# ============================================================
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
init(autoreset=True)

os.makedirs("model", exist_ok=True)
os.makedirs("output_cluster", exist_ok=True)

def log(msg, color=Fore.CYAN):
    t = time.strftime("[%H:%M:%S]")
    print(color + f"{t} {msg}" + Style.RESET_ALL)

# ============================================================
# 1️⃣ LOAD RAW DATA
# ============================================================
log("📂 Loading data_motorbikes.xlsx ...", Fore.GREEN)
df = pd.read_excel("data/data_motorbikes.xlsx")
df = df.reset_index(drop=True)
df["id"] = df.index
log(f"✅ Data loaded: {df.shape[0]} rows", Fore.GREEN)

# ============================================================
# 2️⃣ BEHAVIOR SEGMENTATION
# ============================================================
log("🚀 Starting BEHAVIOR segmentation...", Fore.CYAN)
text_cols = ["Tiêu đề", "Mô tả chi tiết", "Thương hiệu", "Dòng xe", "Loại xe", "Dung tích xe"]

def remove_accents(text):
    text = unicodedata.normalize("NFD", text)
    return text.encode("ascii", "ignore").decode("utf-8")

def clean_text(text):
    if pd.isnull(text): return ""
    text = str(text).lower()
    text = remove_accents(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return word_tokenize(text, format="text").strip()

for col in text_cols:
    df[col] = df[col].astype(str).apply(clean_text)

df["content"] = df[text_cols].agg(" ".join, axis=1)

# --- SBERT Embeddings ---
sbert_path = "model/sbert_embeddings.pkl"
if os.path.exists(sbert_path):
    log("🔁 Loading cached SBERT embeddings ...", Fore.YELLOW)
    embeddings = joblib.load(sbert_path)
else:
    log("🧠 Generating SBERT embeddings (first time) ...", Fore.MAGENTA)
    model = SentenceTransformer("keepitreal/vietnamese-sbert")
    embeddings = model.encode(df["content"].tolist(), show_progress_bar=True)
    joblib.dump(embeddings, sbert_path)
    log("✅ Saved SBERT embeddings cache.", Fore.GREEN)

# --- UMAP ---
umap_path = "model/umap_model.pkl"
if os.path.exists(umap_path):
    log("🔁 Loading cached UMAP model ...", Fore.YELLOW)
    umap_model = joblib.load(umap_path)
else:
    log("🧩 Training UMAP model ...", Fore.MAGENTA)
    umap_model = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=20, random_state=42)
    umap_model.fit(embeddings)
    joblib.dump(umap_model, umap_path)
    log("✅ Saved UMAP model cache.", Fore.GREEN)

X_umap = umap_model.transform(embeddings)

# --- KMeans Behavior ---
km_beh_path = "model/kmeans_behavior.pkl"
if os.path.exists(km_beh_path):
    log("🔁 Loading cached KMeans (Behavior) ...", Fore.YELLOW)
    kmeans_beh = joblib.load(km_beh_path)
else:
    log("⚙️ Training KMeans (Behavior) ...", Fore.MAGENTA)
    best_k = 7
    kmeans_beh = KMeans(n_clusters=best_k, random_state=42, n_init='auto').fit(X_umap)
    joblib.dump(kmeans_beh, km_beh_path)
    log("✅ Saved KMeans (Behavior) cache.", Fore.GREEN)

df["behavior_cluster"] = kmeans_beh.predict(X_umap)

# --- PCA Behavior ---
pca_beh_path = "model/pca_behavior.pkl"
if os.path.exists(pca_beh_path):
    log("🔁 Loading cached PCA (Behavior) ...", Fore.YELLOW)
    pca_beh = joblib.load(pca_beh_path)
else:
    log("⚙️ Training PCA (Behavior) ...", Fore.MAGENTA)
    pca_beh = PCA(n_components=3)
    pca_beh.fit(X_umap)
    joblib.dump(pca_beh, pca_beh_path)
    log("✅ Saved PCA (Behavior) cache.", Fore.GREEN)

beh_factors = pca_beh.transform(X_umap)
df["beh_f1"], df["beh_f2"], df["beh_f3"] = beh_factors[:,0], beh_factors[:,1], beh_factors[:,2]

# ============================================================
# 3️⃣ TECHNICAL SEGMENTATION
# ============================================================
log("🚀 Starting TECHNICAL segmentation...", Fore.CYAN)

def clean_year(x):
    x = str(x).strip().lower()
    if x.isdigit(): return int(x)
    if "200" in x and "x" in x: return 2000
    if "201" in x and "x" in x: return 2010
    if "trước" in x or "truoc" in x: return 1979
    nums = re.findall(r"\d{4}", x)
    if nums: return int(nums[0])
    return np.nan

def clean_price(x):
    if pd.isnull(x): return np.nan
    x = str(x)
    x = re.sub(r"[^0-9]", "", x)
    return float(x) if x != "" else np.nan

df["Year_clean"] = df["Năm đăng ký"].apply(clean_year)
df["Year_clean"] = df["Year_clean"].fillna(df["Year_clean"].median())
df["Tuoi_xe"] = 2025 - df["Year_clean"]

df["Giá_clean"] = df["Giá"].apply(clean_price)
median_price = df["Giá_clean"].median()
df["Giá_clean"] = df["Giá_clean"].fillna(median_price)
log(f"✅ Cleaned price column, median={median_price:,.0f}", Fore.GREEN)

df["Thương hiệu"] = df["Thương hiệu"].fillna("unknown")
df["Loại xe"] = df["Loại xe"].fillna("unknown")
df["Dung tích xe"] = df["Dung tích xe"].fillna("unknown")

df["brand_strength"] = df.groupby("Thương hiệu")["Giá_clean"].transform("median")
df["type_strength"] = df.groupby("Loại xe")["Giá_clean"].transform("median")
df["cc_strength"] = df.groupby("Dung tích xe")["Giá_clean"].transform("median")

tech_features = ["Giá_clean", "Tuoi_xe", "brand_strength", "type_strength", "cc_strength"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[tech_features])

km_tech_path = "model/kmeans_technical.pkl"
if os.path.exists(km_tech_path):
    log("🔁 Loading cached KMeans (Technical) ...", Fore.YELLOW)
    kmeans_tech = joblib.load(km_tech_path)
else:
    log("⚙️ Training KMeans (Technical) ...", Fore.MAGENTA)
    best_k_tech = 5
    kmeans_tech = KMeans(n_clusters=best_k_tech, random_state=42, n_init="auto").fit(X_scaled)
    joblib.dump(kmeans_tech, km_tech_path)
    log("✅ Saved KMeans (Technical) cache.", Fore.GREEN)

df["tech_cluster"] = kmeans_tech.predict(X_scaled)

pca_tech_path = "model/pca_technical.pkl"
if os.path.exists(pca_tech_path):
    log("🔁 Loading cached PCA (Technical) ...", Fore.YELLOW)
    pca_tech = joblib.load(pca_tech_path)
else:
    log("⚙️ Training PCA (Technical) ...", Fore.MAGENTA)
    pca_tech = PCA(n_components=3)
    pca_tech.fit(X_scaled)
    joblib.dump(pca_tech, pca_tech_path)
    log("✅ Saved PCA (Technical) cache.", Fore.GREEN)

tech_factors = pca_tech.transform(X_scaled)
df["tech_f1"], df["tech_f2"], df["tech_f3"] = tech_factors[:,0], tech_factors[:,1], tech_factors[:,2]

# ============================================================
# 4️⃣ META SEGMENTATION (KMeans)
# ============================================================
log("🚀 Starting META segmentation (KMeans) ...", Fore.CYAN)

meta_features = [
    "behavior_cluster", "tech_cluster",
    "beh_f1", "beh_f2", "beh_f3",
    "tech_f1", "tech_f2", "tech_f3"
]

X_meta = df[meta_features].copy()
X_meta_scaled = StandardScaler().fit_transform(X_meta)

km_meta_path = "model/kmeans_meta.pkl"
if os.path.exists(km_meta_path):
    log("🔁 Loading cached KMeans (Meta) ...", Fore.YELLOW)
    kmeans_meta = joblib.load(km_meta_path)
else:
    log("⚙️ Training KMeans (Meta) ...", Fore.MAGENTA)
    best_k_meta = 7
    kmeans_meta = KMeans(n_clusters=best_k_meta, random_state=42, n_init="auto").fit(X_meta_scaled)
    joblib.dump(kmeans_meta, km_meta_path)
    log("✅ Saved KMeans (Meta) cache.", Fore.GREEN)

df["meta_cluster"] = kmeans_meta.predict(X_meta_scaled)

# ============================================================
# 5️⃣ GMM META CLUSTERING
# ============================================================
log("🚀 Running GMM for Meta segmentation ...", Fore.CYAN)
K_meta = len(np.unique(df["meta_cluster"]))
gmm = GaussianMixture(n_components=K_meta, covariance_type='full', random_state=42)
df["meta_gmm"] = gmm.fit_predict(X_meta_scaled)
log("✅ GMM clustering complete.", Fore.GREEN)
# --- PCA 2D cho Meta ---
pca_meta = PCA(n_components=2)
X_meta_pca = pca_meta.fit_transform(X_meta_scaled)
df["pca1"], df["pca2"] = X_meta_pca[:, 0], X_meta_pca[:, 1]

# ============================================================
# 6️⃣ VISUALIZATION (only Meta GMM)
# ============================================================
log("📊 Saving GMM visualizations ...", Fore.CYAN)

# Scatter Plot (Meta PCA Space)
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df["pca1"], y=df["pca2"],
    hue=df["meta_gmm"], palette="tab10", s=25, alpha=0.8
)
plt.title(f"Meta Segmentation – GMM (K={K_meta})", fontweight="bold")
plt.tight_layout()
plt.savefig("output_cluster/meta_gmm_scatter.png", dpi=300)
plt.close()


sample_sil = silhouette_samples(X_meta_scaled, df["meta_gmm"])
sil_avg = silhouette_score(X_meta_scaled, df["meta_gmm"])

plt.figure(figsize=(8,6))
y_lower = 10
for i in range(K_meta):
    ith = sample_sil[df["meta_gmm"] == i]
    ith.sort()
    size_i = ith.shape[0]
    y_upper = y_lower + size_i
    color = cm.tab10(i / K_meta)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith, facecolor=color, alpha=0.7)
    plt.text(-0.03, y_lower + 0.5 * size_i, f"C{i}", fontsize=9)
    y_lower = y_upper + 10

plt.axvline(sil_avg, color="red", linestyle="--", label=f"Avg={sil_avg:.3f}")
plt.title("Silhouette Plot – Meta GMM Clustering", fontweight="bold")
plt.xlabel("Silhouette Value")
plt.ylabel("Clusters")
plt.legend()
plt.tight_layout()
plt.savefig("output_cluster/meta_gmm_silhouette.png", dpi=300)
plt.close()

# Bar Chart: Số lượng mẫu trong mỗi cụm GMM
plt.figure(figsize=(7,4))
df["meta_gmm"].value_counts().sort_index().plot(kind="bar", color="skyblue")
plt.title("Phân bố số lượng mẫu theo cụm Meta GMM", fontweight="bold")
plt.xlabel("Cụm GMM")
plt.ylabel("Số lượng")
plt.tight_layout()
plt.savefig("output_cluster/meta_gmm_cluster_size.png", dpi=300)
plt.close()

# Boxplot: Phân bố giá theo cụm GMM
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="meta_gmm", y=df["Giá_clean"] / 1e6, palette="tab10")
plt.title("Phân bố giá xe theo cụm Meta GMM", fontweight="bold")
plt.xlabel("Cụm GMM")
plt.ylabel("Giá (triệu VND)")
plt.ylim(0, np.percentile(df["Giá_clean"] / 1e6, 99))
plt.tight_layout()
plt.savefig("output_cluster/meta_gmm_boxplot_price.png", dpi=300)
plt.close()



# Bảng mô tả cụm (Cluster Summary Table)
summary = (
    df.groupby("meta_gmm")
      .agg({
          "Giá_clean": "mean",
          "Tuoi_xe": "mean",
          "brand_strength": "mean",
          "type_strength": "mean",
          "cc_strength": "mean",
          "id": "count"
      })
      .rename(columns={"id": "Số lượng"})
      .reset_index()
)
summary.round(2).to_excel("output_cluster/meta_gmm_summary.xlsx", index=False)

# Thống kê theo Thương hiệu (Brand Analysis)
brand_summary = (
    df.groupby("Thương hiệu")
      .agg({
          "Giá_clean": ["mean", "median", "count"],
          "Tuoi_xe": "mean"
      })
      .round(1)
)
brand_summary.columns = ["Giá TB (triệu VND)", "Giá trung vị (triệu VND)", "Số lượng", "Tuổi xe TB (năm)"]
brand_summary["Giá TB (triệu VND)"] = (brand_summary["Giá TB (triệu VND)"] / 1e6).round(1)
brand_summary["Giá trung vị (triệu VND)"] = (brand_summary["Giá trung vị (triệu VND)"] / 1e6).round(1)
brand_summary = brand_summary.sort_values("Giá TB (triệu VND)", ascending=False)
brand_summary.to_excel("output_cluster/meta_gmm_brand_summary.xlsx")

# Thống kê theo Loại xe (Vehicle Type)
type_summary = (
    df.groupby("Loại xe")     # ✅ sửa lại đúng tên cột trong file
      .agg({
          "Giá_clean": ["mean", "median", "count"],
          "Tuoi_xe": "mean"
      })
)

type_summary.columns = ["Giá TB (triệu VND)", "Giá trung vị (triệu VND)", "Số lượng", "Tuổi xe TB (năm)"]
type_summary["Giá TB (triệu VND)"] = (type_summary["Giá TB (triệu VND)"] / 1e6).round(1)
type_summary["Giá trung vị (triệu VND)"] = (type_summary["Giá trung vị (triệu VND)"] / 1e6).round(1)
type_summary = type_summary.sort_values("Giá TB (triệu VND)", ascending=False)
type_summary.to_excel("output_cluster/meta_gmm_type_summary.xlsx")


# Thống kê theo Phân khúc Dung tích
def cc_segment(x):
    try:
        cc = float(re.findall(r"\d+", str(x))[0])
    except:
        return "Khác"
    if cc < 100: return "<100cc"
    elif cc < 125: return "100–124cc"
    elif cc < 150: return "125–149cc"
    elif cc < 175: return "150–174cc"
    elif cc < 250: return "175–249cc"
    else: return "≥250cc"

df["Phan_khuc_dung_tich"] = df["Dung tích xe"].apply(cc_segment)

cc_col = "Phan_khuc_dung_tich" if "Phan_khuc_dung_tich" in df.columns else "Dung tích xe"

cc_summary = (
    df.groupby(cc_col)
      .agg({
          "Giá_clean": ["mean", "median", "count"],
          "Tuoi_xe": "mean"
      })
)

cc_summary.columns = ["Giá TB (triệu VND)", "Giá trung vị (triệu VND)", "Số lượng", "Tuổi xe TB (năm)"]
cc_summary["Giá TB (triệu VND)"] = (cc_summary["Giá TB (triệu VND)"] / 1e6).round(1)
cc_summary["Giá trung vị (triệu VND)"] = (cc_summary["Giá trung vị (triệu VND)"] / 1e6).round(1)
cc_summary = cc_summary.sort_values("Giá TB (triệu VND)", ascending=False)
cc_summary.to_excel("output_cluster/meta_gmm_cc_summary.xlsx")




log("✅ Saved charts → output_cluster/meta_gmm_scatter.png & meta_gmm_silhouette.png", Fore.GREEN)
log("🎯 DONE – Full pipeline executed successfully with caching.", Fore.CYAN)
# Lưu toàn bộ DataFrame đã gán cụm
df.to_excel("output_cluster/meta_gmm_full.xlsx", index=False)
log("💾 Saved full dataset with clusters → output_cluster/meta_gmm_full.xlsx", Fore.GREEN)
