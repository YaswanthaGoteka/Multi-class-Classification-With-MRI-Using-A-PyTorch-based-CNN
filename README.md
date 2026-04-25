# Brain Tumor Classification with PyTorch

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify brain MRI images into multiple tumor categories and non-tumor cases.

## Overview

The model is trained on grayscale MRI images and uses data augmentation to improve generalization. It evaluates performance on a test dataset and supports prediction on custom MRI images.

## Dataset Structure

Organize the dataset as follows:

data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
├── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/

## Features

Device Setup:
- Automatically uses GPU (CUDA) if available, otherwise CPU

Data Preprocessing:
- Converts images to grayscale
- Resizes images to 128x128
- Normalizes pixel values
- Applies data augmentation:
  - Random horizontal flip
  - Random rotation

Class Handling:
- Renames "notumor" to "no tumor"
- Computes class distribution

## Model Architecture

Input (1x128x128)  
→ Conv2D (32 filters) → ReLU → MaxPool  
→ Conv2D (64 filters) → ReLU → MaxPool  
→ Conv2D (128 filters) → ReLU → MaxPool  
→ Flatten  
→ Linear (256) → ReLU → Dropout  
→ Output (4 classes)

## Training

- Optimizer: Adam  
- Learning Rate: 1e-4  
- Loss Function: CrossEntropyLoss  
- Epochs: 50  
- Batch Size: 32  

During training, the model reports:
- Loss per epoch
- Training accuracy

## Evaluation

After training, the model achieved the following performance on the test dataset:

- Test Loss: 1.38  
- Test Accuracy: 93.50%

## Sample Prediction

The script:
- Displays a random test MRI image
- Shows predicted and actual labels
- Prints probability distribution across classes

## Predict Custom MRI Image

Use the following function:

```python
pred_idx, confidence, img, probs = predict_image("test_mri.jpg", model, device)
```

## Output includes:
- Predicted class
- Confidence score
- Probability for each class
- Image visualization

## Install dependencies:

pip install torch torchvision matplotlib pillow

## How to Run:
Place dataset in data/Training and data/Testing
Run the script:
- python main.py

### To test a custom image:
Place the image in the project directory
Update: image_path = "your_image.jpg"

## Notes:
- Model expects grayscale MRI images
- Input size must be 128x128
- Class imbalance is calculated but not applied to the loss function

## Limitations:
- The model is trained on a limited dataset and may not generalize well to all MRI scans
- No validation set is used, which may lead to overfitting
- Class imbalance is not handled during training despite being calculated
- The model architecture is relatively simple compared to modern deep learning models
- Performance may vary significantly on real-world clinical data
- No model saving/loading functionality is implemented

## Possible Improvements:
- Use pretrained models (e.g., ResNet, EfficientNet)
- Apply class weights to the loss function
- Add a validation dataset
- Implement early stopping
- Save and load trained models
