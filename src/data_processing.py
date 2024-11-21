import pandas as pd

def load_data(file_path):
    """تحميل البيانات من ملف CSV"""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """معالجة البيانات: حذف القيم المفقودة وتحويل الأعمدة إن لزم الأمر"""
    data = data.dropna()  # حذف القيم المفقودة
    return data
