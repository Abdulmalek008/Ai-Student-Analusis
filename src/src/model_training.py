from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(data, target_column):
    """تدريب نموذج باستخدام RandomForest"""
    X = data.drop(columns=[target_column])  # المدخلات
    y = data[target_column]  # الهدف
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # تدريب النموذج
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # تقييم النموذج
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    
    # حفظ النموذج
    joblib.dump(model, 'student_performance_model.pkl')
    
    return model
