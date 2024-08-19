# Finetune a pre-trained model just as you would do in many real-world applications

Follow [this instruction](https://github.com/Rakeshsah0/dl_assignment_2/blob/main/README.md#installation) for instalation.

## Overview

This project involves fine-tuning a pre-trained ResNet50 model for image classification using the [iNaturalist 12K](https://storage.googleapis.com/wandb_datasets/nature_12K.zip
) dataset. The model is adapted to classify images into 10 different classes, with training, validation, and test phases managed through PyTorch. The project also includes logging and visualization of results using Weights & Biases (WandB).

## Important check points
- check  for installtion of libraries. If any library is missing you can run the given code inside "dl_assignment_2" folder.
```bash
    pip install -r requirements.txt
```
- Check if inaturalist dataset is present.
- Check for path of train dataset at line 23 of part2.py
- Check for path of test dataset at line 27 of part2.py

## Model Architecture

### ResNet50 Fine-Tuning
- **Pre-trained Model**: The model is built on ResNet50, a widely-used deep learning architecture pre-trained on the ImageNet dataset.
- **Layer Freezing**: All layers of the ResNet50 model are frozen except for the final fully connected layer.
- **Final Layer Modification**: The final fully connected layer is replaced to match the number of classes in the dataset (10 classes in this case).

## Dataset

The model is trained and evaluated on the [iNaturalist 12K](https://storage.googleapis.com/wandb_datasets/nature_12K.zip
) dataset, which consists of images belonging to 10 different classes. The dataset is divided into training, validation, and test sets. Data transformations include resizing images to 224x224 pixels and normalizing pixel values.
- **Training Set**: 80% of the images for model training.
- **Validation Set**: 20% of the training images for hyperparameter tuning and model validation.
- **Test Set**: A separate set of images used to evaluate the model's performance.

## Hyperparameters

The following hyperparameters are used:
- **Epochs**: 10
- **num_classes**: 10
- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Optimizer**: Adam

## Training and Evaluation

### Training
- The training process involves iterating over the training data for a specified number of epochs.
- Loss is calculated using CrossEntropyLoss, and the optimizer updates the model's parameters based on the gradients.
- Training accuracy and loss are logged to WandB at each epoch.

### Validation
- During each epoch, the model is also validated using the validation dataset.
- Validation accuracy and loss are similarly logged to WandB.

### Testing
- After training, the model's performance is evaluated on the test set.
- Randomly selected images from the test set are visualized with their true and predicted labels.
- A confusion matrix is generated to analyze model performance across different classes.
- The test accuracy and confusion matrix are logged to WandB.

## Visualization and Logging

The following visualizations and metrics are logged to WandB:
- **Training and Validation Accuracy**: To monitor the model's performance during training.
- **Training and Validation Loss**: To ensure the model is converging properly.
- **Test Predictions**: Visualization of 30 randomly selected images from the test set with their true and predicted labels.
- **Confusion Matrix**: A heatmap showing the confusion matrix for the test set.

## How to Run

To run project type:
```bash
python part2.py
```
