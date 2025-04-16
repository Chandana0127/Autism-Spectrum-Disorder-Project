# Autism Spectrum Disorder Detection Using Deep Learning and Machine Learning

This project aims to detect Autism Spectrum Disorder (ASD) in children using facial images. We compare the performance of a Convolutional Neural Network (CNN) and a Naive Bayes classifier to evaluate their effectiveness in ASD classification.

## 📁 Dataset
The dataset consists of facial images categorized as:
- `autistic`
- `non-autistic`

Each category is split into:
- Training set
- Validation set
- Test set

## ⚙️ Technologies Used
- Python
- Keras / TensorFlow
- OpenCV
- Sklearn
- Matplotlib / Seaborn
- NumPy / Pandas

## 🧠 Models Used
### 1. CNN (Convolutional Neural Network)
- Built using Keras Sequential API
- Layers: Conv2D, MaxPooling2D, Flatten, Dense, Dropout
- Achieved ~70% accuracy

### 2. Naive Bayes
- Images resized and flattened to 1D arrays
- Trained using GaussianNB from `sklearn.naive_bayes`
- Achieved ~54% accuracy

## 📊 Evaluation Metrics
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

## 🔍 Observations
- CNN significantly outperforms Naive Bayes in image classification tasks.
- CNN captures spatial hierarchies and patterns in facial features.
- Naive Bayes is not ideal for image classification as it ignores spatial structure.

## 📬 Contact
For any questions or suggestions, feel free to reach out!

