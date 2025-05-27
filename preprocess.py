# preprocess.py
import pandas as pd
import re
import html
from nltk.corpus import stopwords
import nltk

# Tải stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # Giải mã các ký tự HTML
    text = html.unescape(text)
    # Xóa các cụm header thường thấy trong email
    text = re.sub(r'\b(fyi|re|fw)\s*[:\-]?', '', text)
    # Xóa nội dung trong ngoặc vuông
    text = re.sub(r'\[.*?\]', '', text)
    # Xóa các đường kẻ bảng hoặc ký tự trang trí
    text = re.sub(r'[|+]', ' ', text)
    text = re.sub(r'(-\s*|-{2,}){2,}', ' ', text)
    # Xóa ký tự không phải chữ, số, khoảng trắng hoặc các dấu giữ lại (!, ?, /, :)
    text = re.sub(r"[^\w\s\!\?\/:]", '', text)
    # Xóa dấu chấm riêng lẻ
    text = text.replace('.', '')
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    # Loại bỏ stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def preprocess_data(df):
    # Thay thế NaN bằng chuỗi rỗng
    df['Subject'] = df['Subject'].fillna('')
    df['Message'] = df['Message'].fillna('')
    # Gộp Subject và Message thành 'text'
    df['text'] = df['Subject'] + ' ' + df['Message']
    # Làm sạch văn bản
    df['clean_text'] = df['text'].apply(clean_text)
    # Đảm bảo cột label tồn tại và ở dạng lowercase
    df['label'] = df['Spam/Ham'].str.lower()
    # Loại bỏ dòng trùng lặp
    df = df.drop_duplicates()
    return df[['clean_text', 'label']]