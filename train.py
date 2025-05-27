# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import MyMultinomialNB
from preprocess import preprocess_data
import pickle
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Đọc dữ liệu
logger.info("Đang đọc dữ liệu...")
df = pd.read_csv("enron_spam_data.csv", encoding='utf-8', quoting=csv.QUOTE_ALL, on_bad_lines='skip')

# Làm sạch tên cột
df.columns = df.columns.str.strip()
df = df.rename(columns={'Message': 'Message', 'Subject': 'Subject', 'Spam/Ham': 'Spam/Ham'})

# Tiền xử lý dữ liệu
logger.info("Đang tiền xử lý dữ liệu...")
df = preprocess_data(df)

# Kiểm tra phân phối nhãn
logger.info(f"Phân bố nhãn:\n{df['label'].value_counts(normalize=True)}")

# Tạo vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(df['clean_text']).toarray()
y_tfidf = df['label'].map({'spam': 1, 'ham': 0}).values

# Lưu vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Tách dữ liệu huấn luyện / kiểm tra
X_tfidf_train, X_tfidf_test, y_tfidf_train, y_tfidf_test = train_test_split(
    X_tfidf, y_tfidf, test_size=0.2, random_state=42
)

# Huấn luyện mô hình
logger.info("Đang huấn luyện mô hình...")
model = MyMultinomialNB()
model.fit(X_tfidf_train, y_tfidf_train)

# Lưu mô hình
with open("model_naive_bayes.pkl", "wb") as f:
    pickle.dump(model, f)

# Đánh giá mô hình
y_pred = model.predict(X_tfidf_test)
logger.info("Đánh giá mô hình:")
logger.info(f"Accuracy: {accuracy_score(y_tfidf_test, y_pred):.4f}")
logger.info(f"Precision: {precision_score(y_tfidf_test, y_pred):.4f}")
logger.info(f"Recall: {recall_score(y_tfidf_test, y_pred):.4f}")
logger.info(f"F1-score: {f1_score(y_tfidf_test, y_pred):.4f}")

# In confusion matrix
cm = confusion_matrix(y_tfidf_test, y_pred)
logger.info(f"Confusion Matrix:\n{cm}")

logger.info("Đã huấn luyện và lưu mô hình thành công!")