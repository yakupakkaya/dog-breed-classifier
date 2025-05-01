# 🐶 Dog Breed Classifier

This project implements a complete pipeline to classify dog breeds using deep learning and computer vision. It was developed as part of the [Udacity Machine Learning Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t).

---

## 📌 Project Overview

The goal of the project is to build a system that can:

- Detect whether an image contains a dog or a human
- If a dog is detected, classify its breed
- If a human is detected, predict the dog breed they resemble most

This pipeline integrates:
- Face detection (OpenCV)
- Dog detection (ResNet50 pretrained on ImageNet)
- Fine-tuned CNN model (Xception) trained to classify dog breeds

---

## 🧪 Steps Performed

1. **Face Detection**  
   - Used OpenCV Haar cascades to detect human faces  
2. **Dog Detection**  
   - Used ResNet50 to classify images and detect dog class labels (151–268 on ImageNet)
3. **Model Development**  
   - Extracted bottleneck features from pretrained Xception model
   - Trained a custom fully connected layer on top to classify 133 dog breeds
4. **Model Evaluation**  
   - Achieved high accuracy (~83–85%) on validation and test sets
   - Explored alternative models (ResNet, VGG, Inception) before finalizing Xception
5. **Web App Integration**  
   - Created a Streamlit interface to upload images or choose samples
   - Visualizes predicted breed with sample images of that breed

---

## ✅ Results Summary

### Final Test Accuracy Results for All Models

| Model                      | Test Accuracy |
|---------------------------|----------------|
| Custom CNN (from scratch) | **5.77%**       |
| VGG16                     | 69.46%         |
| VGG19                     | 63.11%         |
| ResNet50                  | 76.89%         |
| InceptionV3               | 77.37%         |
| Xception                  | 81.56%         |
| Xception (augmented)      | **84.91%**     |

> 🧪 For deployment simplicity, the app uses **Xception without augmentation**, but the best model during experimentation was **Xception with augmentation**.

#### 📝 Observations

- The **custom model trained from scratch** performed poorly (5.77%), confirming that a simple CNN lacks the capacity to learn this complex task without transfer learning or data augmentation.
- All **pretrained models** significantly outperform the custom model, benefiting from prior learning on ImageNet.
- Among the non-augmented models, **Xception** achieves the highest accuracy (81.56%), followed closely by InceptionV3 and ResNet50.
- The **Xception model trained with image augmentation** yields the **best overall performance (84.91%)**, demonstrating that data augmentation helps improve generalization by exposing the model to more varied training examples.

These results highlight the strength of transfer learning and show that both model architecture and data strategy (e.g., augmentation) play a crucial role in improving classification performance.

---

## 🛠 How to Run the App

### 1. Clone this repo

```bash
git clone https://github.com/yakupakkaya/dog-breed-classifier.git
cd dog-breed-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> ✅ Python 3.9+ is recommended  
> ✅ Use a virtual environment

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

---

## 🧠 Model Details

- **Base CNN**: Xception pretrained on ImageNet
- **Classifier**: Fully connected layer trained on 133 breeds
- **Training Data**: Subset from Udacity-provided `dogImages/` dataset
- **Evaluation Metrics**: Accuracy, misclassification examples, human-vs-dog analysis

---

## 💡 Future Work

- Replace ResNet dog detector with lightweight YOLOv8 or MobileNet
- Add real-time camera input support
- Deploy app to Hugging Face Spaces or Streamlit Cloud

---

## 🙏 Acknowledgements

- [Udacity](https://www.udacity.com/) for project support
- [Keras](https://keras.io/) + [TensorFlow](https://www.tensorflow.org/) for modeling
- [OpenCV](https://opencv.org/) for face detection
