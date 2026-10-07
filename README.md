# 🤟 Arabic Sign Language Classification

A deep learning-based computer vision application for recognizing and classifying Arabic Sign Language alphabet hand gestures from images.

The project uses **Transfer Learning with MobileNet** to classify Arabic Sign Language alphabet images and provides an interactive **Streamlit web application** for real-time image-based prediction.

## 🚀 Live Demo

**Live Application:** Coming soon

**GitHub Repository:**  
https://github.com/Bhaskar-Tamma/Arabic-Sign-Classification

---

## 📌 Project Overview

Arabic Sign Language recognition can help improve communication accessibility between people who use sign language and those who do not.

This project develops an image classification system capable of identifying Arabic Sign Language alphabet gestures using a trained deep learning model.

The trained model is integrated with a Streamlit interface where users can upload an image of a hand gesture and receive the predicted Arabic sign along with its confidence score.

---

## 🎯 Objectives

- Develop an automated Arabic Sign Language alphabet classification system.
- Apply transfer learning using MobileNet.
- Preprocess and augment hand-sign images for model training.
- Classify Arabic Sign Language alphabet gestures.
- Evaluate model performance using classification metrics.
- Develop an interactive Streamlit web application.
- Provide top prediction results and confidence scores.

---

## 🧠 Technology Stack

### Machine Learning / Deep Learning

- Python
- TensorFlow
- Keras
- MobileNet
- NumPy
- Pandas
- Matplotlib

### Image Processing

- PIL / Pillow
- Image resizing
- Image normalization
- Data augmentation

### Web Application

- Streamlit

### Development Tools

- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

---

## 🏗️ System Architecture

```text
Input Image
     │
     ▼
Image Upload
     │
     ▼
Image Preprocessing
     │
     ├── Resize to 224 × 224
     ├── RGB Conversion
     └── Pixel Normalization
     │
     ▼
MobileNet-Based Model
     │
     ▼
Feature Extraction
     │
     ▼
Classification Layer
     │
     ▼
Predicted Arabic Sign
     │
     ▼
Confidence Score
     │
     ▼
Streamlit Web Interface
```

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Data Augmentation
   ↓
Train / Validation / Test Split
   ↓
MobileNet Transfer Learning
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit Deployment
   ↓
Image Prediction
```

---

## 📊 Dataset

The project uses an Arabic Sign Language image dataset containing images representing different Arabic sign alphabet classes.

The complete dataset is not included in this repository because of its large size.

After downloading the dataset, place it in the appropriate project directory.

Example:

```text
data/
├── train/
├── validation/
└── test/
```

---

## 🔬 Model

The project uses **MobileNet with transfer learning** as the backbone network.

The input images are resized to:

```text
224 × 224 pixels
```

Pixel values are normalized to:

```text
0 – 1
```

The model uses image augmentation techniques such as:

- Rotation
- Zoom
- Width shifting
- Height shifting
- Shearing
- Horizontal flipping

The classification network is trained to identify the corresponding Arabic Sign Language alphabet class.

---

## 🖥️ Streamlit Application

The Streamlit application provides an interactive interface where users can:

1. Upload an Arabic Sign Language image.
2. Select the trained model.
3. Preview the uploaded image.
4. Run classification.
5. View the predicted Arabic sign.
6. View prediction confidence.
7. View the top-5 predictions.

---

## 📁 Project Structure

```text
Arabic-Sign-Classification/
│
├── app.py
│
├── mobilenet-arabic-sign-language-magic.ipynb
│
├── Dataset.txt
│
├── requirements.txt
│
├── README.md
│
└── models/
    └── arabic_sign_language_model.h5
```

> The trained model and complete dataset may be hosted separately if their file sizes are too large for the GitHub repository.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Bhaskar-Tamma/Arabic-Sign-Classification.git
```

### 2. Navigate to the project

```bash
cd Arabic-Sign-Classification
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

---

## 🖼️ How to Use

1. Open the Streamlit application.
2. Upload a JPG, JPEG, PNG, or WebP image.
3. Select the trained model.
4. Click **Classify Sign**.
5. View the predicted Arabic alphabet sign.
6. Check the confidence score and top predictions.

---

## 📈 Evaluation

The model can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics help measure the classification performance across the Arabic Sign Language classes.

---

## 🔮 Future Enhancements

- Improve classification performance with additional training data.
- Add real-time webcam-based sign recognition.
- Support continuous sign-language recognition.
- Develop a mobile application.
- Add Arabic text and speech output.
- Improve model inference speed.
- Expand the system beyond alphabet recognition.
- Deploy the application as a publicly accessible web application.

---

## 👨‍💻 Author

**Bhaskar Tamma**

MCA Graduate | Software Developer | Machine Learning & Web Development Enthusiast

### GitHub

https://github.com/Bhaskar-Tamma

---

## ⭐ Acknowledgements

This project was developed as an academic machine learning and computer vision project focused on Arabic Sign Language recognition.

---

## 📄 License

This project is intended for educational and academic purposes.