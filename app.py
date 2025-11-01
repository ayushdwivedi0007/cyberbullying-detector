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

# Add your name at top-right
st.markdown(
    """
    <style>
    .top-right {
        position: fixed;
        top: 10px;
        right: 25px;
        background-color: #1e1e2f;
        color: white;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: bold;
        box-shadow: 1px 1px 6px rgba(0,0,0,0.3);
        z-index: 9999;
    }
    </style>
    <div class="top-right">👨‍💻 Ayush Dwivedi</div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Main title & input box
# -----------------------------
st.title("🧠 Cyberbullying Detection App")

text = st.text_area("✍️ Enter a message to check:", height=150, placeholder="Type something...")

# -----------------------------
# Prediction logic
# -----------------------------
if st.button("🔍 Detect"):
    if text.strip():
        try:
            input_data = vectorizer.transform([text])
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.error("🚨 **Cyberbullying Detected!**")
            else:
                st.success("✅ **No Cyberbullying Detected.**")
        except Exception as e:
            st.error("⚠️ Error during prediction. Vectorizer may not match the model.")
    else:
        st.warning("⚠️ Please enter some text before clicking **Detect**.")
