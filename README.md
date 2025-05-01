# 🐶 Dog Breed Classifier

A local web app that predicts the breed of a dog (or the breed you resemble!) using a deep learning model based on Xception architecture.

Built as part of the Udacity Dog Breed Classification project.

---

## 🚀 Features

- Upload an image or use sample test images  
- Detect if the image contains a human or a dog  
- Predict the closest dog breed using a fine-tuned CNN (Xception)  
- Show example images of the predicted breed  
- Clean and lightweight Streamlit interface  
- Runs **fully offline** on your local machine  

---

## 🧠 Model Details

- Pretrained Xception model (ImageNet)  
- Custom classifier trained on 133 dog breeds  
- Uses OpenCV for human face detection  
- Uses ResNet50 for dog detection (ImageNet label ranges 151–268)  

---

## 🛠 How to Run

### 1. Clone this repo

```bash
git clone https://github.com/yakupakkaya/dog-breed-classifier.git
cd dog-breed-classifier
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

> ✅ Ensure you're using Python 3.9+ in a virtual environment

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📁 Folder Structure

```
dog-breed-classifier/
├── app.py
├── extract_bottleneck_features.py
├── dog_breed_notebook.ipynb
├── requirements.txt
├── .gitignore
└── saved_models/
    └── weights.best.Xception.hdf5
```

> Image folders like `dog_subset/` and `sample_images/` are excluded to keep the repo lightweight.

---

## 📦 Required Files (External)

You must download the following manually to run the app fully:

| Resource                       | Description              | Location                        |
|--------------------------------|--------------------------|----------------------------------|
| `weights.best.Xception.hdf5`   | Fine-tuned model weights | [Insert your Google Drive link] |
| `dog_subset/` (optional)       | Sample breed images      | [Insert link]                   |
| `sample_images/` (optional)    | Sample test inputs       | [Insert link]                   |

---

## ✅ Example Use

Upload a photo of a dog or a human face, and the app will:

- Classify the dog breed  
- If it's a human, suggest which dog breed you resemble  
- Show reference images of the predicted breed  

---

## 💡 Future Improvements

- Replace ResNet50 dog detector with a custom dog-specific detector  
- Add multiple model options (e.g., ResNet, MobileNet)  
- Deploy to Hugging Face or Streamlit Cloud  

---

## 📜 License

This project is for educational use (Udacity ML Nanodegree) and personal experimentation only.

---

## 🙏 Acknowledgements

- [Udacity](https://www.udacity.com/)  
- [Keras](https://keras.io/)  
- [TensorFlow](https://www.tensorflow.org/)  
- [OpenCV](https://opencv.org/) for face detection
