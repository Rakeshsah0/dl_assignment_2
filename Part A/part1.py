import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam
import wandb as wb
import gc
import torch.optim as optim

# structure of CNN
class CNN(nn.Module):
    def __init__(self, input_shape, num_filters, filter_sizes, activation, pool_sizes, dense_neurons, num_classes, use_batch_norm, dropout_prob):
        super(CNN, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.dropout_prob = dropout_prob

        # Defining convolutional layers with optional batch normalization
        self.conv1 = nn.Conv2d(input_shape[0], num_filters[0], filter_sizes[0])
        self.bn1 = nn.BatchNorm2d(num_filters[0]) if use_batch_norm else None
        self.act1 = self.get_activation(activation[0])
        self.pool1 = nn.MaxPool2d(pool_sizes[0])

        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], filter_sizes[1])
        self.bn2 = nn.BatchNorm2d(num_filters[1]) if use_batch_norm else None
        self.act2 = self.get_activation(activation[1])
        self.pool2 = nn.MaxPool2d(pool_sizes[1])

        self.conv3 = nn.Conv2d(num_filters[1], num_filters[2], filter_sizes[2])
        self.bn3 = nn.BatchNorm2d(num_filters[2]) if use_batch_norm else None
        self.act3 = self.get_activation(activation[2])
        self.pool3 = nn.MaxPool2d(pool_sizes[2])

        self.conv4 = nn.Conv2d(num_filters[2], num_filters[3], filter_sizes[3])
        self.bn4 = nn.BatchNorm2d(num_filters[3]) if use_batch_norm else None
        self.act4 = self.get_activation(activation[3])
        self.pool4 = nn.MaxPool2d(pool_sizes[3])

        self.conv5 = nn.Conv2d(num_filters[3], num_filters[4], filter_sizes[4])
        self.bn5 = nn.BatchNorm2d(num_filters[4]) if use_batch_norm else None
        self.act5 = self.get_activation(activation[4])
        self.pool5 = nn.MaxPool2d(pool_sizes[4])

        # Defining fully connected layer with dropout
        self.fc = nn.Linear(self.calculate_fc_input_size(input_shape), dense_neurons)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(dense_neurons, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))) if self.use_batch_norm else self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))) if self.use_batch_norm else self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.bn3(self.conv3(x))) if self.use_batch_norm else self.act3(self.conv3(x)))
        x = self.pool4(self.act4(self.bn4(self.conv4(x))) if self.use_batch_norm else self.act4(self.conv4(x)))
        x = self.pool5(self.act5(self.bn5(self.conv5(x))) if self.use_batch_norm else self.act5(self.conv5(x)))
        
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layer
        x = F.relu(self.fc(x))
        x = self.dropout(x)  # Apply dropout
        x = self.output(x)
        return x

    def get_activation(self, activation):
        if activation == 'ReLU':
            return nn.ReLU()
        elif activation == 'Sigmoid':
            return nn.Sigmoid()
        elif activation == 'Tanh':
            return nn.Tanh()
        elif activation == 'GELU':
            return nn.GELU()
        elif activation == 'SiLU':
            return nn.SiLU()
        elif activation == 'Mish':
            return nn.Mish()
        else:
            raise NotImplementedError("Activation function {} not implemented".format(activation))

    def calculate_fc_input_size(self, input_shape):
        # Create a random image tensor
        temp_image = torch.randn(1, *input_shape)
        # Perform convolutions and max-pooling operations
        x = self.pool1(self.act1(self.bn1(self.conv1(temp_image))) if self.use_batch_norm else self.act1(self.conv1(temp_image)))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))) if self.use_batch_norm else self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.bn3(self.conv3(x))) if self.use_batch_norm else self.act3(self.conv3(x)))
        x = self.pool4(self.act4(self.bn4(self.conv4(x))) if self.use_batch_norm else self.act4(self.conv4(x)))
        x = self.pool5(self.act5(self.bn5(self.conv5(x))) if self.use_batch_norm else self.act5(self.conv5(x)))
        # Return the shape of the output after the fifth max-pooling operation
        return x.size(1) * x.size(2) * x.size(3)


# load dataset 
transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 32
dataset = torchvision.datasets.ImageFolder(root='/inaturalist_12K/train',transform=transform)
val_size = int(0.2 * len(dataset))
train_size = len(dataset)-val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
test_dataset = torchvision.datasets.ImageFolder(root='/inaturalist_12K/val',transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


classes = ('Amphibia', 'Animalia', 'Arachnida', 'Aves','Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia')


#definition of training class 
def training(epochs, model, train_loader, device, opt, loss_fn, val_loader):
    for epoch in range(epochs):
        # Training phase
        model.train()  # Set the model to training mode
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

            # Calculate train accuracy
            _, predicted_train = torch.max(yHat.data, 1)
            total_train += y.size(0)
            correct_train += (predicted_train == y).sum().item()

            train_loss += loss.item()

        train_accuracy = correct_train / total_train
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                X_val, y_val = batch
                X_val, y_val = X_val.to(device), y_val.to(device)
                yHat_val = model(X_val)
                loss_val = loss_fn(yHat_val, y_val)

                # Calculate validation accuracy
                _, predicted_val = torch.max(yHat_val.data, 1)
                total_val += y_val.size(0)
                correct_val += (predicted_val == y_val).sum().item()

                val_loss += loss_val.item()

        val_accuracy = correct_val / total_val
        val_loss /= len(val_loader)
        print("epoch:",epoch)
        # Log to wandb
        wb.log({"epoch": epoch,"train_loss": train_loss,"train_accuracy": train_accuracy,"val_loss": val_loss,"val_accuracy": val_accuracy})


#definition of main function which is called from every sweep
def mainFunction():
    wb.init(project="dl_assignment_2_cnn")
    config = wb.config
    run_name = f'{config.optimizer}{config.dense_neurons}{config.activations}{config.batch_normalisation}{config.dropout}{config.num_filters}_{config.learning_rate}'
    # Set the run name
    wb.run.name = run_name
    wb.run.save()
    
    input_shape = (3, 224, 224)
    num_filters=config.num_filters
    filter_sizes=config.filter_sizes
    activations=config.activations
    pool_sizes=config.pool_sizes
    dense_neurons=config.dense_neurons
    num_classes=config.num_classes
    use_batch_norm=config.batch_normalisation
    dropout_prob=config.dropout
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = CNN(input_shape, num_filters, filter_sizes, activations, pool_sizes, dense_neurons, num_classes,use_batch_norm,dropout_prob).to(device)
    optimizers = {'adam': optim.Adam,'nadam': optim.Adam}
    
    learning_rate=config.learning_rate
    opt=optimizers[config.optimizer.lower()](model.parameters(),lr=learning_rate)
#     opt=config.(model.parameters(),lr=learning_rate)
    loss_fn= nn.CrossEntropyLoss()
    epochs=config.epochs
    training(epochs,model,train_loader,device,opt,loss_fn,val_loader)
    
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()


#configuration of sweep
sweep_config = {
    'method': 'bayes',
    'name' : 'hyperparameter sweep v5',
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'num_filters': {
          'values': [[32, 64, 128, 256, 512],[512,256,128,64,32],[32,32,32,32,32]]
        },
        'filter_sizes': {
          'values': [[3, 3, 3, 3, 3],[5,5,5,5,5]]
        },
        'activations':{
            'values':[['ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU'],['GELU','GELU','GELU','GELU','GELU'],['Mish','Mish','Mish','Mish','Mish'],['SiLU','SiLU','SiLU','SiLU','SiLU']]
        },
        'pool_sizes':{
            'values':[[2, 2, 2, 2, 2]]
        },
        'learning_rate': {
            'values':[1e-3,1e-4]
        },
        'dense_neurons':{
            'values': [128,256,512]
        },
        'optimizer': {
            'values': ['nadam', 'adam']
        },
        'epochs': {
            'values': [10]
        },
        'num_classes': {
            'values': [10]
        },
        'batch_normalisation':{
            'values':[True,False]
        },
        'dropout':{
            'values':[0,0.2,0.3]
        }
    }
}


# execution of code using sweep configuration and mapping the appropriate function to it

sweep_id = wb.sweep(sweep=sweep_config,project='dl_assignment_2_cnn')
wb.agent(sweep_id, function = mainFunction , count = 30)
wb.finish()