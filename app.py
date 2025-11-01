import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("🧠 Cyberbullying Detector")
st.write("Enter a text message to check if it contains cyberbullying content.")

text = st.text_area("Enter text here:")

if st.button("Detect"):
    if text.strip():
        input_data = vectorizer.transform([text])
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error("🚨 Cyberbullying detected!")
        else:
            st.success("✅ No cyberbullying detected.")
    else:
        st.warning("Please enter some text before detecting.")
