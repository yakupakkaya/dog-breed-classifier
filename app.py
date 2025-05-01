import streamlit as st
import numpy as np
import cv2
import os
import random
import glob
from PIL import Image
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import load_model
from extract_bottleneck_features import extract_Xception
import json

os.makedirs("temp_upload", exist_ok=True)

# Load models and cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
ResNet50_dm = ResNet50(weights='imagenet')
xception_model = load_model('saved_models/weights.best.Xception.hdf5')


# Load dog_names from local dog_subset
dog_names = [item.replace("\\", "/") for item in sorted(glob.glob("dog_subset/train/*"))]



def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except IOError:
        return None

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def ResNet50_predict_labels(img_path):
    img = preprocess_input(preprocess_image(img_path))
    return np.argmax(ResNet50_dm.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return 151 <= prediction <= 268

def Xception_predict_breed(img_path):
    bottleneck_feature = extract_Xception(preprocess_image(img_path))
    predicted_vector = xception_model.predict(bottleneck_feature)
    breed_path = dog_names[np.argmax(predicted_vector)]
    breed_file = breed_path.split("/")[-1]
    breed = breed_path.split(".")[-1]
    return breed, breed_file

def run_algorithm(img_path):
    is_human = face_detector(img_path)
    is_dog = dog_detector(img_path)

    if is_dog:
        breed, breed_file = Xception_predict_breed(img_path)
        return f"Dog detected! Predicted breed: {breed}", breed_file
    elif is_human:
        breed, breed_file = Xception_predict_breed(img_path)
        return f"Human detected! You resemble a {breed}.", breed_file
    else:
        return "Error: No dog or human detected.", None

def show_breed_samples(breed_file, input_image):
    base_path = f"dog_subset/train/{breed_file}"
    breed_sample_paths = glob.glob(f"{base_path}/*")

    st.image(input_image, caption="Input Image", width=250)

    if not breed_sample_paths:
        st.warning("No sample images available for this breed.")
        return

    selected_samples = random.sample(breed_sample_paths, min(3, len(breed_sample_paths)))
    st.markdown("### Sample images of predicted breed:")
    cols = st.columns(len(selected_samples))

    for i, path in enumerate(selected_samples):
        with cols[i]:
            st.image(path, caption=os.path.basename(path), width=200)


# Streamlit App
st.title("🐶 Dog Breed Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])# Updated sample image path
default_images = glob.glob("sample_images/*.jpg")

img_path = None

try:
    if uploaded_file is not None:
        img_path = f"temp_upload/{uploaded_file.name}"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    elif st.button("Use Sample Image"):
        if not default_images:
            st.warning("⚠️ No sample images found in 'sample_images' folder.")
            st.stop()
        img_path = random.choice(default_images)

    if img_path is None:
        st.info("Upload an image or use a sample image to start.")
        st.stop()

    result, breed_file = run_algorithm(img_path)
    st.write(result)

    if breed_file:
        show_breed_samples(breed_file, img_path)

except Exception as e:
    st.error(f"🚨 Something went wrong:\n{e}")