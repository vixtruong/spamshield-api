from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load mô hình và vectorizer khi khởi động server
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model_naive_bayes.pkl", "rb") as f:
    model = pickle.load(f)

# Schema nhận dữ liệu từ client
class EmailRequest(BaseModel):
    content: str

# Trả về kết quả spam hay không
@app.post("/predict")
def predict(req: EmailRequest):
    X = vectorizer.transform([req.content])
    y_pred = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0].tolist()

    return {
        "label": "spam" if y_pred == 1 else "not spam",
        "probabilities": y_proba  # [P(ham), P(spam)]
    }
