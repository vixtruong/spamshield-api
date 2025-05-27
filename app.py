# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load vectorizer và mô hình
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model_naive_bayes.pkl", "rb") as f:
    model = pickle.load(f)

# Schema nhận dữ liệu từ client
class EmailRequest(BaseModel):
    content: str

# Endpoint dự đoán spam hay không
@app.post("/predict")
async def predict(req: EmailRequest):
    try:
        logger.info(f"Văn bản đầu vào: {req.content[:100]}...")
        cleaned_content = clean_text(req.content)
        logger.info(f"Văn bản sau khi làm sạch: {cleaned_content[:100]}...")

        # Biến đổi văn bản thành ma trận TF-IDF
        X = vectorizer.transform([cleaned_content]).toarray()
        logger.info(f"Kích thước vector TF-IDF: {X.shape}")

        # Dự đoán
        y_pred = model.predict(X)[0]
        y_proba = model.predict_proba(X)[0].tolist()

        # Tính toán tầm quan trọng của các từ
        feature_names = vectorizer.get_feature_names_out()
        feature_contributions = []
        for i, tfidf_score in enumerate(X[0]):
            if tfidf_score > 0:  # Chỉ xét các từ có trong văn bản
                spam_contribution = tfidf_score * model.feature_log_prob_[1][i]  # Đóng góp vào spam
                ham_contribution = tfidf_score * model.feature_log_prob_[0][i]   # Đóng góp vào ham
                feature_contributions.append({
                    "word": feature_names[i],
                    "tfidf_score": float(tfidf_score),
                    "spam_contribution": float(spam_contribution),
                    "ham_contribution": float(ham_contribution)
                })

        # Sắp xếp theo mức độ đóng góp vào lớp được dự đoán
        feature_contributions = sorted(
            feature_contributions,
            key=lambda x: x["spam_contribution" if y_pred == 1 else "ham_contribution"],
            reverse=True
        )[:5]  # Lấy top 5 từ quan trọng

        logger.info(f"Kết quả dự đoán: label={y_pred}, probabilities={y_proba}")

        return {
            "label": "spam" if y_pred == 1 else "ham",
            "probabilities": y_proba,
            "explanation": {
                "key_words": feature_contributions,
                "message": f"Email được phân loại là {'spam' if y_pred == 1 else 'ham'} dựa trên các từ khóa quan trọng."
            }
        }
    except Exception as e:
        logger.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
        return {"error": str(e)}