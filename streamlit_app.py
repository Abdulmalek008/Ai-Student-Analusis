import pdfplumber
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# مسار ملف PDF
pdf_path = 'file:///C:/Users/Al-kc/Desktop/%E2%80%8E%E2%81%A8%D8%B0%D9%83%D8%A7%D8%A1%20%D8%A7%D8%B5%D8%B7%D9%86%D8%A7%D8%B9%D9%8A1%E2%81%A9.pdf'  # استبدل بهذا المسار الفعلي للملف

# فتح ملف PDF واستخراج النصوص
with pdfplumber.open(pdf_path) as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text()  # استخراج النص من جميع الصفحات

# البيانات الوهمية المستخرجة من النص
print("النص المستخرج من PDF:")
print(text[:1000])  # طباعة أول 1000 حرف لمراجعة النص

# --------------------------------------
# تنظيم البيانات المستخرجة باستخدام pandas
# --------------------------------------

# مثال على البيانات المستخرجة من النص
data = {
    "Student_ID": ["123", "124", "125"],
    "Student_Name": ["Ali", "Sara", "Ahmed"],
    "Total_Score": [97, 85, 92],
    "Final_Exam_Score": [15, 14, 13],
    "Practical_Score": [25, 23, 22],
    "Participation": [10, 8, 9],
}

df = pd.DataFrame(data)

# --------------------------------------
# بناء نموذج تعلم الآلة
# --------------------------------------

# تحويل الأعمدة إلى قيم رقمية
df['Total_Score'] = pd.to_numeric(df['Total_Score'])
df['Final_Exam_Score'] = pd.to_numeric(df['Final_Exam_Score'])
df['Practical_Score'] = pd.to_numeric(df['Practical_Score'])
df['Participation'] = pd.to_numeric(df['Participation'])

# تحديد المدخلات والهدف
X = df[['Final_Exam_Score', 'Practical_Score', 'Participation']]  # المدخلات
y = df['Total_Score'].apply(lambda x: 1 if x >= 90 else 0)  # التصنيف: 1 للطلاب المتفوقين، 0 لغير المتفوقين

# تقسيم البيانات إلى مجموعة تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب نموذج RandomForest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# التنبؤ بالنتائج باستخدام مجموعة الاختبار
y_pred = model.predict(X_test)

# تقييم النموذج
accuracy = accuracy_score(y_test, y_pred)
print(f"\nالدقة: {accuracy:.2f}")
print("\nتقرير التصنيف:\n", classification_report(y_test, y_pred))

# حفظ النموذج
joblib.dump(model, 'student_performance_model.pkl')
