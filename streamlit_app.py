import streamlit as st
import pdfplumber
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Streamlit app setup
st.title("Student Performance Analysis using Machine Learning")
st.write("""
    This app analyzes student performance using machine learning.
    It extracts data from a PDF file and processes it to build a classification model.
""")

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file to analyze data", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF file
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()  # Extract text from all pages

    # Display part of the extracted text for review
    st.write("Extracted text from PDF:")
    st.text(text[:1000])  # Display first 1000 characters for review

    # --------------------------------------
    # Organizing the extracted data with pandas
    # --------------------------------------

    # Example of extracted data from the text (customize based on actual data format)
    data = {
        "Student_ID": ["123", "124", "125"],
        "Student_Name": ["Ali", "Sara", "Ahmed"],
        "Total_Score": [97, 85, 92],
        "Final_Exam_Score": [15, 14, 13],
        "Practical_Score": [25, 23, 22],
        "Participation": [10, 8, 9],
    }

    df = pd.DataFrame(data)

    # Display the extracted data in Streamlit
    st.write("Extracted Data:")
    st.dataframe(df)

    # --------------------------------------
    # Build Machine Learning Model
    # --------------------------------------

    # Convert columns to numeric values
    df['Total_Score'] = pd.to_numeric(df['Total_Score'])
    df['Final_Exam_Score'] = pd.to_numeric(df['Final_Exam_Score'])
    df['Practical_Score'] = pd.to_numeric(df['Practical_Score'])
    df['Participation'] = pd.to_numeric(df['Participation'])

    # Define features and target
    X = df[['Final_Exam_Score', 'Practical_Score', 'Participation']]  # Features
    y = df['Total_Score'].apply(lambda x: 1 if x >= 90 else 0)  # Target: 1 for excellent, 0 for others

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the RandomForest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict the results using the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"\nAccuracy: {accuracy:.2f}")
    st.write("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, 'student_performance_model.pkl')
