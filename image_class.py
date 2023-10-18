import streamlit as st
import requests
import time
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide", page_title="Image Classification", page_icon=":computer:")
st.title("Image classification")
st.markdown(f"**Upload and Predict an image using  Google Vision Transformer**")

# Sidebar content
st.sidebar.subheader("About the app")
st.sidebar.info("This app uses 🤗HuggingFace's [google/vit-base](https://huggingface.co/google/vit-base-patch16-224) model.\
                 \nYou can find the source code [here](https://github.com/giulianavll/ImageClassification-H)")
st.sidebar.write("\n\n")
st.sidebar.markdown("**Get a free API key from HuggingFace:**")
# HuggingFace API KEY input
API_KEY = st.sidebar.text_input("Enter your HuggingFace API key",  type="password")
st.sidebar.markdown("* Create a [free account](https://huggingface.co/join) or [login](https://huggingface.co/login)")
st.sidebar.markdown("* Go to **Settings** and then **Access Tokens**")
st.sidebar.markdown("* Create a new Token (select 'read' role)")
st.sidebar.markdown("* Paste your API key in the text box")
st.sidebar.divider()
st.sidebar.write("Please ensure that your image has one of the following extensions: .png, .jpg, or .jpeg.")
st.sidebar.write("\n\n")
st.sidebar.divider()
st.sidebar.caption("Inspired by [Text Summarization](https://ivan-lee.medium.com/) using [Streamlit](https://streamlit.io/)🎈.")


# Inputs 
# Image
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    if image is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.image(
                image,
                caption="",
                #use_column_width="auto",
                width=200
            )
        with col3:
            st.write(' ')
        bytes_data = img_file_buffer.getvalue()
col11, col21, col31 = st.columns([1.15,1,0.8])
with col11:
    st.write(" ")
with col21:
    submit_button = st.button("Submit")
with col31:
    st.write(" ")

# HuggingFace API inference URL.
API_URL =  "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
#API_URL = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
headers = {"Authorization": f"Bearer {API_KEY}"}



if submit_button:
    def query(image):
        response = requests.post(API_URL, headers=headers, data=image)
        return response.json()

    with st.spinner('Doing some AI magic, please wait...'):
        time.sleep(1)
        # Query the API
        output = query(bytes_data)
        if isinstance(output, list):
            keys_=[]
            values_=[]
            score = 0
            label = ""
            for v in output:
                v_score = v["score"]
                v_label = v["label"]
                keys_.append(v_score)
                values_.append(v_label)
                if score < v_score:
                    score = round(v_score*100,1)
                    label = v_label
            df = pd.DataFrame({"probability": keys_,
                           "label": values_})
            st.divider()
            st.subheader(f"Classified as: {label} with {score}\% of probability ")
            fig = px.bar(df, x='label', y='probability', color = 'label')
            st.plotly_chart(fig, theme=None, use_container_width=True)
        else:
            st.write("Please try again later.")
