# ============================================================
# 📗 recommender.py – TF-IDF + Word2Vec Hybrid Recommendation
#   (optimized for motorbike_final_dataset_clean.csv)
# ============================================================
import os, re, joblib
import numpy as np
import pandas as pd
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1️⃣ LOAD DATA
# ============================================================
def load_clean_data(path="data/motorbike_final_dataset_clean.csv"):
    df = pd.read_csv(path)
    print(f"✅ Dữ liệu đã nạp: {df.shape[0]} dòng, {df.shape[1]} cột.")
    if "full_description" not in df.columns:
        raise ValueError("❌ Thiếu cột 'full_description' trong dữ liệu đầu vào!")
    return df


# ============================================================
# 2️⃣ TRAIN TF-IDF MODEL
# ============================================================
def build_tfidf(df, save_dir="model"):
    print("⚙️  Đang huấn luyện mô hình Gensim TF-IDF...")
    texts = [simple_preprocess(str(doc)) for doc in df["full_description"]]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]
    tfidf_model = models.TfidfModel(corpus)
    index = similarities.MatrixSimilarity(tfidf_model[corpus], num_features=len(dictionary))

    os.makedirs(save_dir, exist_ok=True)
    dictionary.save(os.path.join(save_dir, "dictionary.dict"))
    tfidf_model.save(os.path.join(save_dir, "tfidf_gensim.model"))
    index.save(os.path.join(save_dir, "tfidf_index.index"))
    joblib.dump(texts, os.path.join(save_dir, "texts.pkl"))

    print("✅ Đã huấn luyện và lưu mô hình TF-IDF.")
    return dictionary, tfidf_model, index, texts


# ============================================================
# 3️⃣ TRAIN WORD2VEC MODEL
# ============================================================
def build_w2v(texts, save_dir="model"):
    print("⚙️  Đang huấn luyện mô hình Word2Vec (150d, sg=1)...")
    model_w2v = Word2Vec(
        sentences=texts, vector_size=150, window=5,
        min_count=2, sg=1, workers=4, epochs=10
    )
    model_path = os.path.join(save_dir, "w2v_model.pkl")
    model_w2v.save(model_path)
    print(f"✅ Đã lưu Word2Vec → {model_path}")
    return model_w2v


# ============================================================
# 4️⃣ UTILS
# ============================================================
def get_vector(words, model):
    vecs = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vecs, axis=0) if len(vecs) > 0 else np.zeros(model.vector_size)



# ============================================================
# 5️⃣ RECOMMENDATION ENGINE – REFACTORED (Data Scientist ver.)
# ============================================================
def recommend_hybrid(
    query, df, dictionary, tfidf_model, index, model_w2v, texts,
    top_n=50, final_k=10, cluster_col=None
):
    """
    Gợi ý xe tương tự – bản 3.2 stable
    -------------------------------------------------
    - TF-IDF + Word2Vec hybrid
    - Re-ranking theo giá trong query (bỏ qua 'cc')
    - Boost theo tuổi, km, thương hiệu, dòng xe, và phân khúc dung tích
    """
    import re
    from sklearn.metrics.pairwise import cosine_similarity
    from gensim.utils import simple_preprocess
    import numpy as np
    import pandas as pd

    print(f"\n🔍 Gợi ý cho: {query}")
    query_tokens = simple_preprocess(query)

    # --- TF-IDF Phase ---
    vec_tfidf = tfidf_model[dictionary.doc2bow(query_tokens)]
    sims_tfidf = index[vec_tfidf]
    top_idx = np.argsort(sims_tfidf)[::-1][:top_n]

    # --- Word2Vec Phase ---
    doc_vecs = np.array([get_vector(texts[i], model_w2v) for i in top_idx])
    query_vec = get_vector(query_tokens, model_w2v).reshape(1, -1)
    sims_w2v = cosine_similarity(query_vec, doc_vecs).flatten()

    sims_mix = 0.6 * sims_w2v + 0.4 * np.array(sims_tfidf[top_idx])

    # --- Detect price (ignore cc/ml) ---
    price_query = None
    match = re.search(
        r"(?<!cc)(?<!ml)(\d{1,3}(?:[.,]\d{1,3})*)(?=\s*(?:tr|triệu|trieu|vnd|đ|dong)\b)",
        query.lower()
    )
    if match:
        try:
            val = match.group(1).replace(",", ".")
            val_f = float(val)
            price_query = val_f * 1_000_000 if val_f < 1000 else val_f
            print(f"💰 Phát hiện giá trong query: {price_query:,.0f} VND")
        except:
            price_query = None

    # --- Detect CC segment (phân khúc dung tích) ---
    cc_segment = None
    cc_match = re.search(r"(\d{2,3})\s*cc", query.lower())
    if cc_match:
        cc_val = int(cc_match.group(1))
        if cc_val < 100:
            cc_segment = "Dưới 100 cc"
        elif 100 <= cc_val <= 175:
            cc_segment = "100 - 175 cc"
        else:
            cc_segment = "Trên 175 cc"
        print(f"⚙️ Phát hiện phân khúc dung tích: {cc_segment}")

    # --- Parse price & compute weight (only for top_idx) ---
    def parse_price(x):
        if pd.isnull(x): return np.nan
        s = str(x).replace(",", ".").lower()
        try:
            if "tr" in s or "triệu" in s:
                return float(re.findall(r"[\d.]+", s)[0]) * 1_000_000
            elif "đ" in s:
                return float(re.findall(r"[\d.]+", s)[0])
            else:
                return float(s)
        except:
            return np.nan

    price_weight = np.ones(len(top_idx))
    if price_query and "Giá" in df.columns:
        temp_price = df.iloc[top_idx]["Giá"].apply(parse_price)
        diff = abs(temp_price - price_query) / price_query
        price_weight = np.clip(1 - diff.fillna(0.5), 0.5, 1.0)

    # --- Boost by age/km ---
    age_boost = np.ones(len(top_idx))
    km_boost = np.ones(len(top_idx))

    if "Tuoi_xe" in df.columns:
        median_age = df["Tuoi_xe"].median()
        age_boost = np.where(df.iloc[top_idx]["Tuoi_xe"] < median_age, 1.1, 1.0)
    if "Số_km_da_đi" in df.columns:
        median_km = df["Số_km_da_đi"].median()
        km_boost = np.where(df.iloc[top_idx]["Số_km_da_đi"] < median_km, 1.1, 1.0)

    # --- Boost theo thương hiệu, dòng xe, phân khúc cc ---
    query_lower = query.lower()
    base_boost = []
    for i in top_idx:
        brand = str(df.iloc[i].get("Thương hiệu", "")).lower()
        model = str(df.iloc[i].get("Dòng xe", "")).lower()
        cc_text = str(df.iloc[i].get("Dung tích xe", "")).strip()

        brand_boost = 1.15 if brand and brand in query_lower else 1.0
        model_boost = 1.25 if model and model in query_lower else 1.0
        cc_boost = 1.2 if cc_segment and cc_segment.lower() in cc_text.lower() else 1.0
        cluster_boost = (
            1.10
            if cluster_col
            and df.iloc[i].get(cluster_col, None)
            == df.iloc[top_idx[0]].get(cluster_col, None)
            else 1.0
        )
        base_boost.append(brand_boost * model_boost * cc_boost * cluster_boost)

    # --- Combine all boosts ---
    sims_final = sims_mix * np.array(base_boost) * price_weight * age_boost * km_boost

    # --- Select top-k ---
    best_idx = np.argsort(sims_final)[::-1][:final_k]
    selected_idx = [top_idx[i] for i in best_idx]

    # --- Giữ lại tiêu đề gốc (không bị rút gọn) ---
    if "Tiêu đề_gốc" not in df.columns and "Tiêu đề" in df.columns:
        df["Tiêu đề_gốc"] = df["Tiêu đề"]

    # Sau khi chọn top kết quả
    results = df.iloc[selected_idx].copy()

    # Nếu tồn tại tiêu đề gốc → khôi phục
    if "Tiêu đề_gốc" in results.columns:
        results["Tiêu đề"] = results["Tiêu đề_gốc"]

    results["id"] = df.iloc[selected_idx]["id"].values
    results["Rank"] = np.arange(1, len(selected_idx) + 1)
    results["Độ tương đồng (%)"] = np.round(np.array(sims_final)[best_idx] * 100, 2)

    print(f"✅ Hoàn tất gợi ý nâng cao ({len(selected_idx)} kết quả).")
    return results.reset_index(drop=True)



# 5️⃣.5️⃣ WRAPPER – cho app.py gọi gọn hơn
# ============================================================
def recommend_unified(query, df, dictionary, tfidf_model, index, model_w2v, texts, final_k=10):
    """Hàm gọn cho app.py"""
    return recommend_hybrid(query, df, dictionary, tfidf_model, index, model_w2v, texts, final_k=final_k)


# ============================================================
# 6️⃣ MAIN TEST
# ============================================================
if __name__ == "__main__":
    df = load_clean_data()
    dictionary, tfidf_model, index, texts = build_tfidf(df)
    model_w2v = build_w2v(texts)

    query = "xe tay ga yamaha grande 125cc xanh"
    results = recommend_hybrid(query, df, dictionary, tfidf_model, index, model_w2v, texts)

    print("\n📋 TOP XE TƯƠNG TỰ:")
    print(results)
