import streamlit as st
from PIL import Image
# from classify import predict

from predict import predict_return_ndarray, pred_mask
import matplotlib.pyplot as plt


st.title("UltraSound Nerve Segmentation Web App")

uploaded_file = st.file_uploader("Choose an image...", type="tif")
if uploaded_file is not None:

    print(uploaded_file)
    image = Image.open(uploaded_file)
    print(type(image))
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    file_path = "demo/images/" + uploaded_file.name

    print(file_path)

    image_vis, mask_vis, pr_mask_, pr_mask_er = predict_return_ndarray(file_path)


    mask_path = file_path.split(".")[0] + "_mask.tif"
    mask_ = Image.open(mask_path)

    pr_mask = pred_mask(mask_vis)


    plt.imsave('filename.jpeg', pr_mask)



    st.image(mask_, caption='Ground Truth Mask', use_column_width=True)

    st.image("filename.jpeg", caption="Predicted Mask", use_column_width=True)
