import streamlit as st
import pickle

# -----------------------------
# Load model & vectorizer safely
# -----------------------------
try:
    model = pickle.load(open("LinearSVCTuned.pkl", "rb"))
    vectorizer = pickle.load(open("tfidfvectorizer.pkl", "rb"))
except Exception as e:
    st.error("❌ Model or vectorizer file not found or incompatible!")
    st.stop()

# -----------------------------
# Streamlit UI setup
# -----------------------------
st.set_page_config(page_title="Cyberbullying Detector", page_icon="🧠")

st.markdown(
    """
    <style>
    .top-right {
        position: fixed;
        top: 10px;
        right: 25px;
        background-color: #262730;
        color: white;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: bold;
        box-shadow: 1px 1px 6px rgba(0,0,0,0.3);
    }
    </style>
    <div class="top-right">👨‍💻 Ayush Dwivedi</div>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Cyberbullying Detection App")
st.write("Enter a message to check whether it contains cyberbullying content:")

text = st.text_area("✍️ Enter text:", height=150, placeholder="Type something...")

if st.button("🔍 Detect"):
    if text.strip():
        try:
            input_data = vectorizer.transform([text])
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.error("🚨 Cyberbullying Detected!")
            else:
                st.success("✅ No Cyberbullying Detected.")
        except Exception as e:
            st.error("⚠️ Error during prediction. Vectorizer may not match the model.")
    else:
        st.warning("Please enter text before clicking Detect.")
