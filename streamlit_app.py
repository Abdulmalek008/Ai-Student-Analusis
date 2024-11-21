import streamlit as st
import pdfplumber
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# إعداد واجهة المستخدم باستخدام Streamlit
st.title("تحليل أداء الطلاب باستخدام تعلم الآلة")
st.write("""
    هذا التطبيق يقوم بتحليل أداء الطلاب باستخدام تعلم الآلة.
    يتم استخراج البيانات من ملف PDF وتحليلها لبناء نموذج تصنيف.
""")

# تحميل ملف PDF من قبل المستخدم
uploaded_file = st.file_uploader("اختر ملف PDF لتحليل البيانات", type="pdf")

if uploaded_file is not None:
    # استخراج النصوص من الملف الذي رفعه المستخدم
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()  # استخراج النص من جميع الصفحات

    # عرض جزء من النص المستخرج لمراجعة البيانات
    st.write("النص المستخرج من PDF:")
    st.text(text[:1000])  # طباعة أول 1000 حرف لمراجعة النص

    # --------------------------------------
    # تنظيم البيانات باستخدام pandas
    # --------------------------------------

    # مثال على البيانات المستخرجة من النص (يمكن تخصيصها بناءً على تنسيق بياناتك الفعلي)
    data = {
        "Student_ID": ["123", "124", "125"],
        "Student_Name": ["Ali", "Sara", "Ahmed"],
        "Total_Score": [97, 85, 92],
        "Final_Exam_Score": [15, 14, 13],
        "Practical_Score": [25, 23, 22],
        "Participation": [10, 8, 9],
    }

    df = pd.DataFrame(data)

    # عرض البيانات في Streamlit
    st.write("البيانات المستخرجة:")
    st.dataframe(df)

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

    st.write(f"\nالدقة: {accuracy:.2f}")
    st.write("\nتقرير التصنيف:\n", classification_report(y_test, y_pred))

    # حفظ النموذج
    joblib.dump(model, 'student_performance_model.pkl')
