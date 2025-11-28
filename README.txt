📘 MOTORBIKE RECOMMENDATION & CLUSTERING PROJECT

(Pandora – Motorbike Insight & Recommendation System)

🎯 1️⃣ Mục tiêu

Xây dựng hệ thống gợi ý xe máy cũ dựa trên mô tả, thương hiệu, dòng xe và đặc điểm kỹ thuật.

Kết hợp phân cụm hành vi & kỹ thuật (Behavior + Technical) → Meta Segmentation (GMM) để tạo insight tự động.

Cung cấp giao diện Streamlit thân thiện cho người mua và người bán.

🧩 2️⃣ Cấu trúc thư mục
File / Folder	Mô tả
app.py	Giao diện Streamlit chính – gồm 3 tab: Gợi ý xe, Thông tin chi tiết, Phân tích mô hình
preprocess.py	Làm sạch & chuẩn hóa dữ liệu gốc data_motorbikes.xlsx
recommender.py	Huấn luyện mô hình TF-IDF (Sklearn, Gensim), Word2Vec và lưu model
clustering.py	Pipeline phân cụm (Behavior, Technical, Meta GMM) + xuất biểu đồ và summary
data/data_motorbikes.xlsx	Dữ liệu gốc
output_cluster/	Chứa biểu đồ và bảng kết quả phân cụm
model/	Chứa vectorizer, embedding và model đã huấn luyện
requirements.txt	Danh sách thư viện cần thiết (đúng version local)
assets/	Ảnh, biểu đồ hoặc logo dùng cho giao diện
README.md	Hướng dẫn sử dụng
⚙️ 3️⃣ Quy trình huấn luyện & phân cụm

Bước 1: Làm sạch dữ liệu

python preprocess.py


→ Tạo file motorbike_clean.csv

Bước 2: Huấn luyện mô hình gợi ý

python recommender.py


→ Sinh ra:

model/tfidf_vectorizer.pkl

model/w2v_model.pkl

model/tfidf_matrix.npy

Bước 3: Phân cụm & tạo insight Meta GMM

python clustering.py


→ Sinh ra:

output_cluster/meta_gmm_scatter.png

output_cluster/meta_gmm_boxplot_price.png

output_cluster/meta_gmm_summary.xlsx

💻 4️⃣ Chạy giao diện GUI
streamlit run app.py


Giao diện gồm 3 tab:

🚗 Gợi ý xe tương tự

Người dùng nhập mô tả xe → hiển thị top xe gợi ý kèm giá, ảnh, mức độ tương đồng.

Hiển thị thẻ Insight tự động (phân khúc + mức giá đề xuất).

📊 Thông tin phân khúc

Tóm tắt cụm, biểu đồ giá trung bình, loại xe, dung tích.

🧠 Phân tích mô hình (GMM)

Hiển thị scatter, silhouette plot, boxplot giá và summary theo cụm.

📦 5️⃣ Output mô hình & biểu đồ
Loại output	Đường dẫn	Mô tả
Mô hình TF-IDF, Word2Vec	model/	Vectorizer, embedding & ma trận tương đồng
File phân cụm & insight	output_cluster/meta_gmm_summary.xlsx	Thông tin cụm và phân khúc
Biểu đồ GMM	output_cluster/*.png	Scatter, Silhouette, Boxplot, Cluster Size
🧠 6️⃣ Công nghệ sử dụng
Thành phần	Thư viện chính
GUI	streamlit==1.51.0, plotly==6.3.0
Xử lý dữ liệu	pandas, numpy, openpyxl
NLP	underthesea, sentence-transformers, umap-learn
Machine Learning	scikit-learn, GaussianMixture
Visualization	matplotlib, seaborn, wordcloud
🧾 7️⃣ Phiên bản môi trường khuyến nghị
Python 3.11.x
streamlit==1.51.0
pandas==2.3.2
numpy==2.3.3
scikit-learn==1.7.2
sentence-transformers==5.1.2
underthesea==8.3.0
umap-learn==0.5.9.post2
plotly==6.3.0
matplotlib==3.10.6
seaborn==0.13.2
openpyxl==3.1.5

👨‍💻 8️⃣ Tác giả

Nguyễn Thanh Hải & Châu Lê
“Built with Streamlit · For research & demo purpose – 2025”