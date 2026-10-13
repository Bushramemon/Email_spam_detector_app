import streamlit as st
from model import predict_spam

st.title("📧 Email Spam Detector")

st.write("Enter an email or message to check whether it is Spam or Ham.")

message = st.text_area("Enter your message:")

if st.button("Check for Spam"):

    if message.strip():

        result = predict_spam(message)

        if result == "Spam":
            st.error("🚨 This message is Spam!")
        else:
            st.success("✅ This message is Not Spam!")

    else:
        st.warning("Please enter a message.")
