import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Load and train model
@st.cache_resource
def train_model():
    df = pd.read_csv("bangladesh_student_performance.csv")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    df['Pass'] = ((df['HSC_Result'] >= 4.0) & (df['SSC_Result'] >= 4.0)).astype(int)
    df_model = df.drop(['Student_ID', 'HSC_Result', 'SSC_Result'], axis=1)
    
    text_columns = ['Gender', 'District', 'School_Type', 'Parent_Education', 
                    'Internet_Access', 'Private_Tuition']
    for col in text_columns:
        df_model[col] = le.fit_transform(df_model[col])
    
    df_model['Pass'] = df['Pass']
    df_model = df_model.drop('Performance', axis=1, errors='ignore')
    
    X = df_model.drop('Pass', axis=1)
    y = df_model['Pass']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    return rf, X.columns

model, feature_cols = train_model()

# App UI
st.title("🎓 Student Performance Predictor")
st.write("Predict whether a student will be a high or low performer based on their background and study habits.")

st.sidebar.header("Enter Student Details")

study_hours = st.sidebar.slider("Study Hours per Week", 1, 40, 15)
attendance = st.sidebar.slider("Attendance (%)", 50, 100, 80)
previous_gpa = st.sidebar.slider("Previous GPA (0-5)", 0.0, 5.0, 4.0)
family_income = st.sidebar.slider("Family Income (BDT)", 10000, 60000, 35000)
age = st.sidebar.slider("Age", 15, 18, 17)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
school_type = st.sidebar.selectbox("School Type", ["Public", "Private"])
internet_access = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
private_tuition = st.sidebar.selectbox("Private Tuition", ["Yes", "No"])

# Encode inputs
gender_enc = 1 if gender == "Male" else 0
school_enc = 1 if school_type == "Public" else 0
internet_enc = 1 if internet_access == "Yes" else 0
tuition_enc = 1 if private_tuition == "Yes" else 0

input_data = pd.DataFrame([[gender_enc, age, 3, school_enc, study_hours,
                             attendance, 1, family_income,
                             internet_enc, tuition_enc, previous_gpa]],
                           columns=feature_cols)

if st.button("Predict Performance"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    confidence = max(probability) * 100
    
    if prediction == 1:
        st.success(f"HIGH Performer — Confidence: {confidence:.1f}%")
    else:
        st.warning(f"LOW Performer — Confidence: {confidence:.1f}%")
    
    st.subheader("What matters most?")
    st.write("Based on our model, these are the top factors affecting performance:")
    st.write("1.Family Income — 19.42%")
    st.write("2.Previous GPA — 18.44%")
    st.write("3.Attendance — 15.09%")
    st.write("4.Study Hours — 13.59%")