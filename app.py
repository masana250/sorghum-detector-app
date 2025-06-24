
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

st.title("セイバンモロコシ検出アプリ")

# YOLOモデル読み込み
model = YOLO("best.pt")

# 画像アップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_path = "temp.jpg"
    image.save(image_path)

    results = model.predict(image_path, conf=0.25, save=True)
    result_path = os.path.join(results[0].save_dir, "temp.jpg")
    st.image(result_path, caption="検出結果", use_column_width=True)
