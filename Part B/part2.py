import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import wandb as wb
import gc
import numpy as np
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights



transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 32
# Load iNaturalist12k dataset
dataset = torchvision.datasets.ImageFolder(root='/inaturalist_12K/train',transform=transform)
val_size = int(0.2 * len(dataset))
train_size = len(dataset)-val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
test_dataset = torchvision.datasets.ImageFolder(root='/inaturalist_12K/val',transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('Amphibia', 'Animalia', 'Arachnida', 'Aves','Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia')


# def freeze_up_to_k(k):
#     model = models.resnet50(pretrained=True)
#     layer_num = 0
#     for child in model.children():
#         layer_num += 1
#         if layer_num <= k:
#             for param in child.parameters():
#                 param.requires_grad = False
#     return model


# def freeze_rest_after_k(k):
#     model = models.resnet50(pretrained=True)
#     layer_num = 0
#     for child in model.children():
#         layer_num += 1
#         if layer_num > k:
#             for param in child.parameters():
#                 param.requires_grad = False
#     return model


def fineTuneCNN(num_classes):
    # Load pre-trained ResNet50 model
    model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Freeze all layers except the final classification layer
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final classification layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model



def training_and_evaluation(epochs, model, train_loader, val_loader, test_loader, device, opt, loss_fn, classes):
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch in train_loader:
            X, y = batch
            X, y = X.to(device), y.to(device)
            yHat = model(X)
            loss = loss_fn(yHat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            _, predicted_train = torch.max(yHat.data, 1)
            total_train += y.size(0)
            correct_train += (predicted_train == y).sum().item()

            train_loss += loss.item()

        train_accuracy = correct_train / total_train
        train_loss /= len(train_loader)

        # Log training metrics
        wb.log({"epoch": epoch, "train_accuracy": train_accuracy, "train_loss": train_loss})
        print("epoch: ",epoch," train accuracy: ",train_accuracy," train_loss: ",train_loss)
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                X_val, y_val = batch
                X_val, y_val = X_val.to(device), y_val.to(device)
                yHat_val = model(X_val)
                loss_val = loss_fn(yHat_val, y_val)

                _, predicted_val = torch.max(yHat_val.data, 1)
                total_val += y_val.size(0)
                correct_val += (predicted_val == y_val).sum().item()

                val_loss += loss_val.item()

        val_accuracy = correct_val / total_val
        val_loss /= len(val_loader)
        wb.log({"epoch": epoch, "val_accuracy": val_accuracy, "val_loss": val_loss})
        print("epoch: ",epoch," val accuracy: ",val_accuracy," val_loss: ",val_loss)
    # Evaluation phase on the test set
    model.eval()

    all_images = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu())
            all_predictions.extend(predicted.cpu())

    # Randomly select 30 images
    selected_indices = random.sample(range(len(all_images)), 30)
    selected_images = [all_images[i] for i in selected_indices]
    selected_labels = [all_labels[i] for i in selected_indices]
    selected_predictions = [all_predictions[i] for i in selected_indices]

    # Convert images to 3-channel RGB
    selected_images_rgb = [img.permute(1, 2, 0).numpy() if img.shape[0] == 3 else img.repeat(3, 1, 1).permute(1, 2, 0).numpy() for img in selected_images]

    # Display the selected images with original and predicted labels
    num_rows = 10
    num_cols = 3
    plt.figure(figsize=(15, 30))
    for i in range(30):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(selected_images_rgb[i])
#         plt.title(f"True: {classes[selected_labels[i]]}, Pred: {classes[selected_predictions[i]]}")
        plt.title(f"True: {classes[selected_labels[i].item()]}, Pred: {classes[selected_predictions[i].item()]}")

        plt.axis('off')
    plt.show()

    # Log the selected images to WandB
    wb.log({"Predictions": [wb.Image(selected_images_rgb[i], caption=f"True: {classes[selected_labels[i]]}, Pred: {classes[selected_predictions[i]]}") for i in range(30)]})

    # Calculate accuracy on the test set
    accuracy = sum([1 for i in range(len(all_labels)) if all_predictions[i] == all_labels[i]]) / len(all_labels)
    wb.log({"test_accuracy": accuracy * 100})
    print('Accuracy on the test set: {:.2f}%'.format(100 * accuracy))

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

    # Log confusion matrix heatmap to WandB
    wb.log({"confusion_matrix": wb.Image(plt)})



# finetuning 
epochs = 10
num_classes = 10
learning_rate = 0.0001
optimiser='adam'
wb.init(project="FineTune_CNN_DL2_Test_Evaluation", config={
    "epochs": epochs,
    "num_classes": num_classes,
    "learning_rate": learning_rate,
    "optimiser": optimiser,
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = fineTuneCNN(num_classes).to(device)
optimizers = {'adam': optim.Adam,'nadam': optim.Adam}
opt=optimizers[optimiser.lower()](model.parameters(),lr=learning_rate)
loss_fn= nn.CrossEntropyLoss()
training_and_evaluation(epochs, model, train_loader, val_loader, test_loader, device, opt, loss_fn, num_classes)
# Finish the WandB run
wb.finish()
model.cpu()
del model
gc.collect()
torch.cuda.empty_cache()
