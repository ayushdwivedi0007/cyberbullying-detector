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
st.set_page_config(page_title="Cyberbullying Detector", page_icon="🕵️‍♂️", layout="centered")

# Force dark top bar & show your name in top-right
st.markdown(
    """
    <style>
        /* Hide default Streamlit header */
        header {visibility: hidden;}

        /* Add your custom name bar */
        .custom-header {
            position: fixed;
            top: 0;
            right: 0;
            left: 0;
            background-color: #262730;
            color: white;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 25px;
            font-weight: bold;
            font-size: 16px;
            z-index: 9999;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
    </style>
    <div class="custom-header">👨‍💻 Ayush Dwivedi</div>
    """,
    unsafe_allow_html=True
)

# Add space below header so content doesn’t overlap
st.markdown("<br><br><br>", unsafe_allow_html=True)

# -----------------------------
# Main title & input box
# -----------------------------
st.title("🕵️‍♂️ Cyberbullying Detection App")

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


