import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼 Image to Pencil Sketch")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)

    img_array = np.array(image)
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gray_img)
    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    invertedblur = cv2.bitwise_not(blur)
    sketch = cv2.divide(gray_img, invertedblur, scale=256.0)

    st.image(sketch, caption='Pencil Sketch', use_column_width=True)