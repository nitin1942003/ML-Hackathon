# CIFake Image Classification - Detecting AI-Generated Images

## Introduction

With the rise of generative AI models such as Stable Diffusion, Adobe Firefly, and Midjourney, differentiating between AI-generated and real images has become increasingly challenging. This project aims to build a deep learning model that classifies images as either AI-generated (fake) or real, even when adversarial perturbations are introduced.

## Problem Statement

The goal is to develop a machine learning model capable of detecting AI-generated images, with a specific focus on images created using diffusion-based techniques. The final model will be evaluated on two datasets:

1. **Test_dataset_1**: Contains real and AI-generated images without adversarial perturbations.
2. **Test_dataset_2**: Contains real and AI-generated images with adversarial perturbations.

## Dataset

The dataset used in this project is the **CIFAKE dataset**, consisting of:
- **Training Data**: 50,000 real images and 50,000 AI-generated images.
- **Testing Data**: 10,000 real images and 10,000 AI-generated images.

**Dataset Source:** [CIFAKE - Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

## Model Architecture

The model is a **Convolutional Neural Network (CNN)** with the following layers:
- Input layer with image normalization
- Multiple convolutional layers with ReLU activation
- Max-pooling layers
- Fully connected (dense) layers with dropout for regularization
- Output layer with sigmoid activation for binary classification (Real vs. Fake)

## Installation and Setup

### Prerequisites
Make sure you have Python 3.x installed along with the required dependencies. Install dependencies using:

```bash
pip install -r requirements.txt
```

### Running the Model

1. **Download the dataset** from Kaggle and extract it.
2. **Train the model** by running:
   ```bash
   python train.py
   ```
3. **Test the model** on unseen images:
   ```bash
   python test.py
   ```

### Generating Predictions and CSV Files
To generate predictions on the test datasets and save the results in CSV format:

```bash
python generate_results.py
```

This will create:
- `Test_1_results.csv` for **Test_dataset_1**
- `Test_2_results.csv` for **Test_dataset_2**

Each CSV file will have the following format:

| Image Name | Predicted Label |
|------------|----------------|
| img1.jpg   | Real           |
| img2.jpg   | Fake           |

## Evaluation Metrics

The model is evaluated based on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

## Adversarial Robustness

To improve robustness, adversarial examples are generated using the **Fast Gradient Sign Method (FGSM)**, and the model is fine-tuned with adversarial training.

## Deliverables

The following items are included in the submission:
- **All code files**
- **Trained model weights**
- **README file (this document)**
- **Test results in CSV format**

## Acknowledgments

- Dataset: CIFAKE (Kaggle)
- Libraries: TensorFlow, NumPy, Pandas, Matplotlib

## License
This project is open-source and free to use under the MIT License.

