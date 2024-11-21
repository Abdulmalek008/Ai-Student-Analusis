import pandas as pd
from src.data_processing import load_data, preprocess_data
from src.model_training import train_model

def main():
    # تحميل البيانات
    data = load_data('data/dataset.csv')
    
    # معالجة البيانات
    data = preprocess_data(data)
    
    # تدريب النموذج
    model = train_model(data, target_column='Target')
    
if __name__ == "__main__":
    main()
