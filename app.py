import streamlit as st
import numpy as np
import joblib

svc_model = joblib.load('svm.pkl')
knn_model = joblib.load('knn.pkl')

st.title("Parkinson's Disease Prediction")

model_option = st.sidebar.selectbox(
    "Select the Model for Prediction",
    ('Support Vector Classifier (SVC)', 'K-Nearest Neighbors (KNN)')
)
    
st.header("Enter the 22 Input Features:")
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
    'spread2', 'D2', 'PPE'
]

user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0, format="%.6f")
    user_input.append(value)

if st.button("Predict"):
    input_data = np.array(user_input).reshape(1, -1)
    
    if model_option == 'Support Vector Classifier (SVC)':
        prediction = svc_model.predict(input_data)
    else:
        prediction = knn_model.predict(input_data)

    if prediction[0] == 1:
        st.success("The model predicts that the person is likely to have Parkinson's Disease.")
    else:
        st.markdown('<p style="color:red; font-size:20px;">The model predicts that the person is unlikely to have Parkinson\'s Disease.</p>', unsafe_allow_html=True)
       # st.success("The model predicts that the person is unlikely to have Parkinson's Disease.")