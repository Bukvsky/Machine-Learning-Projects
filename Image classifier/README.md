# CIFAR-10 Image Classification with CNN

A deep learning project implementing Convolutional Neural Networks (CNN) for image classification on the CIFAR-10 dataset using TensorFlow/Keras.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [File Structure](#file-structure)
- [Contributing](#contributing)

## 🔍 Overview

This project demonstrates image classification using deep learning techniques on the CIFAR-10 dataset. The implementation includes data preprocessing, model training, evaluation, and visualization of results with confusion matrices and prediction examples.

## 📸 Dataset

**CIFAR-10** consists of 60,000 32x32 color images in 10 classes:
- ✈️ Airplane
- 🚗 Automobile  
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐎 Horse
- 🚢 Ship
- 🚛 Truck

**Dataset Split:**
- Training: 50,000 images
- Testing: 10,000 images

## ✨ Features

- **Data Visualization**: Interactive grid display of training samples
- **Data Preprocessing**: Normalization and one-hot encoding
- **CNN Architecture**: Multi-layer convolutional neural network
- **Data Augmentation**: Image transformations to improve model generalization
- **Model Evaluation**: Accuracy metrics and confusion matrix visualization
- **Model Persistence**: Save and load trained models
- **Prediction Visualization**: Side-by-side comparison of predictions vs ground truth

## 🛠️ Requirements

```txt
tensorflow>=2.12.0
keras>=2.12.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
Pillow>=8.3.0
```

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd cifar10-cnn-classification
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```



## 🚀 Usage

### Basic Training

```python
# Load and preprocess data
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Train the model
history = cnn_model.fit(X_train, y_train, batch_size=32, epochs=2)
```

### Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)

# Train with augmented data
cnn_model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=4)
```

### Model Evaluation

```python
# Evaluate model performance
evaluation = cnn_model.evaluate(X_test, y_test)
print('Test accuracy: {}'.format(evaluation[1]))

# Generate predictions
predictions = cnn_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
```

## 🏗️ Model Architecture

```
Sequential Model:
├── Conv2D (128 filters, 3x3, ReLU)
├── Conv2D (128 filters, 3x3, ReLU)  
├── MaxPooling2D (2x2)
├── Dropout (0.4)
├── Conv2D (128 filters, 3x3, ReLU)
├── Conv2D (128 filters, 3x3, ReLU)
├── MaxPooling2D (2x2)
├── Dropout (0.4)
├── Flatten
├── Dense (1024, ReLU)
├── Dense (1024, ReLU)
└── Dense (10, Softmax)
```

**Model Parameters:**
- **Optimizer**: RMSprop (learning_rate=0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Epochs**: 2-4 (with data augmentation)

## 📊 Results

### Performance Metrics
- **Training Accuracy**: ~58% (best try)
- **Test Accuracy**: ~59.7%
- **Model Size**: ~36.55  MB


## 🔧 Optimization Tips

For better performance and accuracy:

1. **Increase Training Epochs**: Current model trains for only 2-4 epochs
2. **Add Batch Normalization**: Stabilizes training
3. **Learning Rate Scheduling**: Adaptive learning rates
4. **Early Stopping**: Prevent overfitting
5. **Transfer Learning**: Use pre-trained models like ResNet or EfficientNet

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **CIFAR-10 Dataset**: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009
- **TensorFlow/Keras**: Open-source machine learning framework
- **Python Data Science Stack**: NumPy, Pandas, Matplotlib, Seaborn

## Author

 **Igor Bukowski**

---

