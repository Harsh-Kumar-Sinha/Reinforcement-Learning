import openai
import streamlit as st

openai.api_key = "YOUR_API_KEY_HERE"

st.title("GPT-3 Chatbot")

prompt = st.text_input("Enter your message:")

if prompt:
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    ).choices[0].text
    st.write("GPT-3 says:", response)
