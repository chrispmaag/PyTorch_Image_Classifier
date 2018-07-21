# Trainging network and save checkpoint
# python train.py flowers --learning_rate 0.001 --hidden_units 512 --epochs 1 --arch vgg16 --gpu --save_dir holder

'''
Basic usage: python train.py data_directory
Output: Prints out training loss, validation loss, and validation accuracy
as the network trains

Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg16"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 3
Use GPU for training: python train.py data_dir --gpu

TO TEST:
python train.py flowers --gpu --save_dir holder --epochs 5
python train.py flowers --learning_rate 0.001 --hidden_units 1024 --epochs 1 --arch vgg16 --gpu --save_dir holder
'''

# Imports
import numpy as np
from tqdm import tqdm
import argparse

from collections import OrderedDict
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Instantiates parser.')

parser.add_argument('train_dir_var', action='store',
                    help='Where the training data directory is.')

# Assign true if specified. Otherwise False.
parser.add_argument('--gpu', action='store_true',
                    dest='gpu_var',
                    help='Use GPU for training?')

# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
parser.add_argument('--learning_rate', action='store',
                    dest='lr_var',
                    type=float,
                    default=0.001,
                    help='Set learning rate hyperparameter')

parser.add_argument('--hidden_units', action='store',
                    dest='hidden_units_var',
                    type=int,
                    default=1024,
                    help='Set hidden units hyperparameter')

parser.add_argument('--epochs', action='store',
                    dest='epochs_var',
                    type=int,
                    default=1,
                    help='Set epochs hyperparameter')

# Choose architecture: python train.py data_dir --arch "vgg16"
parser.add_argument('--arch', action='store',
                    dest='arch_var',
                    default='vgg16',
                    help='Choose architecture')

# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
parser.add_argument('--save_dir', action='store',
                    dest='save_dir_var',
                    default='holder/',
                    help='Set directory to save checkpoints')

# Collect the inputs from the command line
results = parser.parse_args()

# Load variables into simple names
gpu_var = results.gpu_var
lr_var = results.lr_var
hidden_units_var = results.hidden_units_var
epochs_var = results.epochs_var
arch_var = results.arch_var
save_dir_var = results.save_dir_var

# Load the data
# Directories
data_dir = results.train_dir_var
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# Only have a training set in the command line version, so can get rid of
# validation and test transforms and loaders
# Define transforms for the training, validation, and testing sets
print('Creating data loaders.')

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# ============================================================================

# Build and train your network
# VGG16
print('Downloading model.')

# First option for architecture
if arch_var == 'vgg16':
    model = models.vgg16(pretrained=True)
    # Input size vgg16 expects for its classifier
    input_size = 25088

# Second option for architecture
if arch_var == 'densenet121':
    model = models.densenet121(pretrained=True)
    # Input size densenet expects for its classifier
    input_size = 1024

if arch_var not in ('vgg16', 'densenet121'):
    print('Please pick from either vgg16 or densenet121.')

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

output_size = 102

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units_var)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.1)),
                          ('fc2', nn.Linear(hidden_units_var, 512)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(0.1)),
                          ('fc3', nn.Linear(512, 256)),
                          ('relu3', nn.ReLU()),
                          ('dropout3', nn.Dropout(0.1)),
                          ('fc4', nn.Linear(256, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

# Train model on GPU if available
device = torch.device('cuda:0' if gpu_var else 'cpu')

criterion = nn.NLLLoss()

# Only train the classifier parameters; feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=lr_var)

# Function for implementing validation loss and accuracy
def validation(model, validloader, criterion):
    valid_loss = 0
    valid_accuracy = 0

    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        valid_accuracy += equality.type(torch.FloatTensor).mean().item()

    return valid_loss, valid_accuracy

# ============================================================================
# Training Section
print('Starting to train model.')
epochs = epochs_var
print_every = 40
steps = 0

running_loss = 0
running_train_accuracy = 0

# To train on GPU
model.to(device)

for e in tqdm(range(epochs)):
    model.train()

    for ii, (images, labels) in tqdm(enumerate(trainloader)):
        steps += 1

        images, labels = images.to(device), labels.to(device)

        # Set gradients to zero so it doesn't accumulate across steps
        optimizer.zero_grad()

        # Forward pass
        output = model.forward(images)

        # Calculate loss
        loss = criterion(output, labels)

        # Update the optimizer
        loss.backward()

        # Update the weights
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy of current batch
        _, predicted = torch.max(output.data, 1)
        running_train_accuracy += (predicted == labels).sum().item() / labels.size(0)

        if steps % print_every == 0:

            # Switch network to evaluation mode to turn off dropout
            model.eval()

            # Turn off gradients for validation (saves memory)
            with torch.no_grad():
                valid_loss, valid_accuracy = validation(model, validloader, criterion)

            print('Epoch {}/{}.. Training loss: {:.3f}.. Validation loss: {:.3f}.. Validation accuracy: {:.3f}'.format(
                        e+1, epochs,
                        running_loss/print_every,
                        valid_loss/len(validloader),
                        valid_accuracy/len(validloader)))

            running_loss = 0
            running_train_accuracy = 0

            # Make sure training is back on
            model.train()


# Checkpoint for model
# Could add model name they selected earlier here so
# that I can load the same model on the predict side
savedata = {'model_state': model.state_dict(),
            'criterion_state': criterion.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'class_to_idx': train_data.class_to_idx,
            'classifier': model.classifier,
            'epochs': epochs_var + 1,
            'learning_rate': lr_var
         }

print('Saving.')
torch.save(savedata, save_dir_var + 'trained_model.pth')
print('Finished.')
