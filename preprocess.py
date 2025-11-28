# ============================================================
# 📘 preprocess.py – Tiền xử lý dữ liệu & văn bản cho hệ thống
# ============================================================

import pandas as pd
import numpy as np
import os, re, unicodedata
from underthesea import word_tokenize

# ============================================================
# 1️⃣ HÀM TIỆN ÍCH
# ============================================================

def remove_accents(text):
    """Bỏ dấu tiếng Việt"""
    if pd.isnull(text):
        return ""
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    return str(text)

def clean_text_light(text):
    """Làm sạch nhẹ (cho Word2Vec)"""
    if pd.isnull(text): return ""
    text = str(text).lower()
    text = word_tokenize(text, format="text")
    text = re.sub(r"[^a-zA-Z0-9áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệ"
                  r"íìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữự"
                  r"ýỳỷỹỵđ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text_full(text):
    """Làm sạch mạnh (cho TF-IDF)"""
    if pd.isnull(text): return ""
    text = str(text).lower()
    text = remove_accents(text)
    text = word_tokenize(text, format="text")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# 2️⃣ LOAD & FEATURE ENGINEERING
# ============================================================

def load_data(path="data/data_motorbikes.xlsx"):
    print("📥 Đang đọc dữ liệu đầu vào...")
    df = pd.read_excel(path)
    df = df.dropna(subset=["Tiêu đề", "Mô tả chi tiết"])

    # 🔑 Giữ ID gốc từ Excel hoặc tạo mới nếu chưa có
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    else:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(method="ffill").astype(int)

    print(f"✅ Đọc thành công {len(df)} dòng từ {path}")
    return df


def feature_engineering(df):
    """Tạo feature kỹ thuật phục vụ clustering"""
    print("⚙️  Đang tạo các feature kỹ thuật...")

    # 1️⃣ Tìm cột có chứa 'năm' và ép kiểu sang số
    year_cols = [c for c in df.columns if "năm" in c.lower()]
    if year_cols:
        year_col = year_cols[0]
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        df["Tuoi_xe"] = 2025 - df[year_col]
    else:
        df["Tuoi_xe"] = np.nan
        print("⚠️  Không tìm thấy cột năm sản xuất, gán NaN.")

    # 2️⃣ Tìm cột chứa thông tin km
    km_cols = [c for c in df.columns if "km" in c.lower()]
    if km_cols:
        df["Số_km_da_đi"] = pd.to_numeric(df[km_cols[0]], errors='coerce')
    else:
        df["Số_km_da_đi"] = np.nan
        print("⚠️  Không tìm thấy cột số km, gán NaN.")

    # 3️⃣ Tìm và xử lý cột giá
    if "Giá" in df.columns:
        df["Giá"] = (
            df["Giá"].astype(str)
            .str.replace("[^0-9]", "", regex=True)
            .replace("", np.nan)
            .astype(float) / 1_000_000
        )
        print("✅ Đã xử lý cột 'Giá' thành số (triệu đồng).")

    elif all(col in df.columns for col in ["Khoảng giá min", "Khoảng giá max"]):
        df["Giá"] = (
            df[["Khoảng giá min", "Khoảng giá max"]]
            .apply(lambda x: np.mean([
                float(re.sub('[^0-9.,]', '', str(v)).replace(',', '.'))
                for v in x if re.sub('[^0-9.,]', '', str(v)).strip() != ""
            ]), axis=1)
        )
        print("✅ Đã tính 'Giá' trung bình từ khoảng giá min/max (triệu đồng).")

    else:
        df["Giá"] = np.nan
        print("⚠️  Không tìm thấy cột giá, gán NaN.")

    # 4️⃣ Tính toán feature mới
    df["Km_moi_nam"] = df["Số_km_da_đi"] / (df["Tuoi_xe"] + 0.1)
    df["Tuoi_xe_x_Km"] = df["Tuoi_xe"] * df["Số_km_da_đi"]
    df["Log_Gia"] = np.log1p(df["Giá"])

    print("✅ Hoàn tất tạo feature kỹ thuật (Tuoi_xe, Km_moi_nam, Log_Gia...).")
    return df


# ============================================================
# 3️⃣ TEXT PIPELINE – Chuẩn hóa mô tả xe
# ============================================================

def text_processing(df, mode="light"):
    """Tiền xử lý text cho TF-IDF / Word2Vec"""
    print("🧹 Đang xử lý văn bản mô tả...")

    func = clean_text_full if mode == "full" else clean_text_light
    text_cols = [
        "Tiêu đề", "Thương hiệu", "Dòng xe", "Loại xe",
        "Dung tích xe", "Mô tả chi tiết"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(func)
        else:
            df[col] = ""

    df["full_description"] = df[text_cols].agg(" ".join, axis=1)
    print(f"✅ Hoàn tất tiền xử lý văn bản ({mode.upper()} mode).")
    return df


# ============================================================
# 4️⃣ PIPELINE TỔNG – CHUẨN HÓA & XUẤT FILE
# ============================================================

def preprocess_pipeline(mode="light"):
    """Pipeline chính cho toàn bộ tiền xử lý"""
    print("🚀 BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU...\n")
    df = load_data()
    print("------------------------------------------------------")
    df = feature_engineering(df)
    print("------------------------------------------------------")
    df = text_processing(df, mode=mode)
    print("------------------------------------------------------")

    # Sắp xếp thứ tự cột quan trọng (giữ ID gốc)
    main_cols = [
        "id",
        "Thương hiệu", "Dòng xe", "Loại xe", "Dung tích xe",
        "Giá", "Số_km_da_đi", "Tuoi_xe", "Km_moi_nam",
        "Tuoi_xe_x_Km", "Log_Gia", "full_description"
    ]
    df = df[[c for c in main_cols if c in df.columns]]

    # Xuất file kết quả
    output_path = "data/motorbike_final_dataset_clean.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"💾 Đã lưu file: {output_path}")
    print("🎯 Dữ liệu sẵn sàng cho Clustering & Recommendation.\n")
    print("=======================================================")
    print(df.head(3))
    print("=======================================================")
    return df


# ============================================================
# 5️⃣ CHẠY THỬ
# ============================================================

if __name__ == "__main__":
    df = preprocess_pipeline(mode="light")
    print("✅ Quy trình tiền xử lý hoàn tất thành công.")
