# Train a CNN model from scratch and learn how to tune the hyperparameters and visualize filters

Follow [this instruction](https://github.com/Rakeshsah0/dl_assignment_2/blob/main/README.md#installation) for instalation.

## Important check points
- check  for installtion of libraries. If any library is missing you can run the given code inside "dl_assignment_2" folder.
```bash
    pip install -r requirements.txt
```
- Check if inaturalist dataset is present.
- Check for path of train dataset at line 99 of part1.py
- Check for path of test dataset at line 103 of part1.py

## Model Architecture

The CNN model is constructed with the following key components:
- **Convolutional Layers**: Five convolutional layers with configurable filter sizes and activation functions.
- **Batch Normalization**: Optionally applied after each convolutional layer to stabilize and accelerate training.
- **Activation Functions**: Various activation functions such as ReLU, GELU, Mish, and SiLU are supported.
- **Pooling Layers**: Max pooling layers are used after each activation function to downsample the feature maps.
- **Fully Connected Layer**: A dense layer with a configurable number of neurons followed by a dropout layer to prevent overfitting.
- **Output Layer**: The output layer uses a softmax function to classify the input images into one of the predefined classes.

## Dataset

The model is trained and evaluated on the [iNaturalist 12K](https://storage.googleapis.com/wandb_datasets/nature_12K.zip
) dataset, which consists of images belonging to 10 different classes. The dataset is divided into training, validation, and test sets. Data transformations include resizing images to 224x224 pixels and normalizing pixel values.
- **Training Set**: 80% of the images for model training.
- **Validation Set**: 20% of the training images for hyperparameter tuning and model validation.
- **Test Set**: A separate set of images used to evaluate the model's performance.

## Hyperparameter Tuning

Hyperparameter tuning is performed using WandB sweeps, employing a Bayesian optimization approach. The following hyperparameters are tuned:
- Number of filters in each convolutional layer
- Filter sizes
- Activation functions
- Pooling sizes
- Learning rate
- Number of neurons in the dense layer
- Optimizer (Adam or Nadam)
- Batch normalization (enabled or disabled)
- Dropout probability

## Training

The training process involves:
1. Forward propagation through the CNN model.
2. Calculation of the loss using CrossEntropyLoss.
3. Backward propagation and optimization of weights using the selected optimizer.
4. Logging of training and validation metrics (accuracy and loss) to WandB after each epoch.

## Best Hyperparameters

Based on the hyperparameter sweep, the best model configuration is:
- epochs = 10
- input_shape = (3, 224, 224)
- Filters: [32, 64, 128, 256, 512]
- Filter Sizes: [3, 3, 3, 3, 3]
- Activations: [Mish,Mish,Mish,Mish,Mish]
- Pool Sizes: [2, 2, 2, 2, 2]
- Dense Neurons: 256
- Batch Normalization: True
- Dropout: 0.2
- Optimizer: Adam
- Learning Rate: 0.0001

## Evaluation

After training, the model is evaluated on the test set:
- **Accuracy Calculation**: The accuracy on the test set is computed and logged to WandB.
- **Confusion Matrix**: A confusion matrix is generated to visualize the performance across different classes.
- **Predictions Visualization**: Randomly selected images from the test set are displayed with their true and predicted labels.

## Running the Code

1. **Hyperparameter Sweep**: Execute the hyperparameter sweep using the `mainFunction()` mapped to the sweep configuration.
2. **Training and Evaluation**: After determining the best hyperparameters, train the model on the entire training dataset and evaluate it on the test dataset using `training_and_evaluation()`.

## How to Run

To run project type:
```bash
python part1.py
```
