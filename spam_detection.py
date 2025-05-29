import streamlit as st
import numpy as np
import pickle


model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Spam Detection App")
input_text = st.text_area("Enter a message:")

if st.button("Predict"):
    transformed_text = vectorizer.transform([input_text]).toarry()
    result = model.predict(transformed_text)
    st.write("Prediction:", "Spam" if result[0] == 1 else "Ham")
