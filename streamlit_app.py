import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model, scaler, and label encoders
model = joblib.load('stroke_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Title
st.title('Aplikasi Prediksi Risiko Stroke')

# Sidebar for inputs
st.sidebar.header('Masukkan Informasi Pasien')

# Input widgets
age = st.sidebar.slider('Usia', 0, 100, 50)
gender = st.sidebar.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan', 'Lainnya'])
hypertension = st.sidebar.selectbox('Hipertensi', ['Tidak', 'Ya'])
heart_disease = st.sidebar.selectbox('Penyakit Jantung', ['Tidak', 'Ya'])
ever_married = st.sidebar.selectbox('Pernah Menikah', ['Tidak', 'Ya'])
work_type = st.sidebar.selectbox('Jenis Pekerjaan', ['Swasta', 'Wiraswasta', 'Pemerintah', 'Anak-anak', 'Tidak pernah bekerja'])
residence_type = st.sidebar.selectbox('Tipe Tempat Tinggal', ['Perkotaan', 'Pedesaan'])
avg_glucose_level = st.sidebar.slider('Rata-rata Kadar Glukosa', 50.0, 300.0, 100.0)
bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
smoking_status = st.sidebar.selectbox('Status Merokok', ['Tidak pernah merokok', 'Pernah merokok', 'Merokok', 'Tidak diketahui'])

# Mapping from Indonesian to English for encoding
gender_map = {'Laki-laki': 'Male', 'Perempuan': 'Female', 'Lainnya': 'Other'}
ever_married_map = {'Tidak': 'No', 'Ya': 'Yes'}
work_type_map = {'Swasta': 'Private', 'Wiraswasta': 'Self-employed', 'Pemerintah': 'Govt_job', 'Anak-anak': 'children', 'Tidak pernah bekerja': 'Never_worked'}
residence_type_map = {'Perkotaan': 'Urban', 'Pedesaan': 'Rural'}
smoking_status_map = {'Tidak pernah merokok': 'never smoked', 'Pernah merokok': 'formerly smoked', 'Merokok': 'smokes', 'Tidak diketahui': 'Unknown'}

# Convert inputs to numerical values
gender_encoded = label_encoders['gender'].transform([gender_map[gender]])[0]
ever_married_encoded = label_encoders['ever_married'].transform([ever_married_map[ever_married]])[0]
work_type_encoded = label_encoders['work_type'].transform([work_type_map[work_type]])[0]
residence_type_encoded = label_encoders['Residence_type'].transform([residence_type_map[residence_type]])[0]
smoking_status_encoded = label_encoders['smoking_status'].transform([smoking_status_map[smoking_status]])[0]

hypertension_num = 1 if hypertension == 'Yes' else 0
heart_disease_num = 1 if heart_disease == 'Yes' else 0

# Create input DataFrame with column names
columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
input_data = pd.DataFrame([[gender_encoded, age, hypertension_num, heart_disease_num, ever_married_encoded,
                            work_type_encoded, residence_type_encoded, avg_glucose_level, bmi, smoking_status_encoded]], columns=columns)

# Scale numerical features
input_data[['age', 'avg_glucose_level', 'bmi']] = scaler.transform(input_data[['age', 'avg_glucose_level', 'bmi']])

# Predict
prediction_proba = model.predict_proba(input_data)[0][1]
prediction = model.predict(input_data)[0]

# Display results
st.header('Hasil Prediksi')
st.write(f'Probabilitas Risiko Stroke: {prediction_proba:.4f}')

if prediction == 1:
    st.error('Risiko Tinggi: Pasien berisiko terkena stroke.')
else:
    st.success('Risiko Rendah: Pasien tidak berisiko tinggi terkena stroke.')

# Additional info
st.info('Prediksi ini berdasarkan model machine learning dan tidak boleh menggantikan nasihat medis profesional.')
