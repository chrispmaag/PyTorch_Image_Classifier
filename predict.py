# Created 'holder/' directory to store my checkpoint
# python predict.py /home/workspace/aipnd-project/flowers/test/1/image_06743.jpg /home/workspace/aipnd-project/holder/trained_model.pth --gpu --category_names cat_to_name.json

'''
Basic usage: python predict.py /path/to/image checkpoint --gpu --category_names cat_to_name.json
Options:
Return top K most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu

TO TEST:
python predict.py /home/workspace/aipnd-project/flowers/test/1/image_06743.jpg /home/workspace/aipnd-project/holder/trained_model.pth --gpu --category_names cat_to_name.json
python predict.py /home/workspace/aipnd-project/flowers/test/1/image_06743.jpg /home/workspace/aipnd-project/holder/trained_model.pth --gpu --top_k 3
python predict.py /home/workspace/aipnd-project/flowers/test/1/image_06743.jpg /home/workspace/aipnd-project/holder/trained_model.pth --gpu --category_names cat_to_name.json --top_k 4
python predict.py /home/workspace/aipnd-project/flowers/test/102/image_08004.jpg /home/workspace/aipnd-project/holder/trained_model.pth --gpu --category_names cat_to_name.json --top_k 4
python predict.py /home/workspace/aipnd-project/flowers/test/19/image_06186.jpg /home/workspace/aipnd-project/holder/trained_model.pth --gpu --category_names cat_to_name.json --top_k 4
'''

# Imports
import numpy as np
import json
import re
import argparse

from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision

from utilities import process_image

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Instantiates parser.')

# Required positional argument for image path
parser.add_argument('path_to_image', action='store',
                    help='Location of image.')

# Required positional argument for checkpoint name
parser.add_argument('checkpoint_var', action='store',
                    help='Name of checkpoint to use for inference.')

# Use GPU for inference. Assign true if specified. Otherwise False.
parser.add_argument('--gpu', action='store_true',
                    dest='gpu_var',
                    help='Use GPU')

# Set hyperparameters: python predict.py input checkpoint --top_k 3
parser.add_argument('--top_k', action='store',
                    dest='top_k_var',
                    type=int,
                    default=5,
                    help='Return top K most likely classes')

# python predict.py input checkpoint --category_names cat_to_name.json
parser.add_argument('--category_names', action='store',
                    dest='category_names_var',
                    help='Use a mapping of categories to real names')

# Collect the inputs from the command line
results = parser.parse_args()

# Load variables into simple names
gpu_var = results.gpu_var
top_k_var = results.top_k_var
category_names_var = None

if results.category_names_var is not None:
    category_names_var = results.category_names_var

# Train model on GPU if available
device = torch.device('cuda:0' if gpu_var else 'cpu')

print('Setting model_checkpoint variable.')
model_checkpoint = results.checkpoint_var

# File to map category label to category name
if category_names_var is not None:
    # this was previously 'cat_to_name.json'
    with open(category_names_var, 'r') as f:
        cat_to_name = json.load(f)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    for param in model.parameters():
        param.requires_grad = False
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

print('Reading in model checkpoint.')
model = load_checkpoint(model_checkpoint)

# Move model to appropriate device (GPU or CPU)
model.to(device)


def predict(image_path, model, topk=top_k_var):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Preprocess image
    image = process_image(image_path)

    # Convert to Torch Tensor
    image = torch.from_numpy(image)

    # Move tensor onto device
    image = image.to(device)

    # Convert input image to float to avoid RuntimeError: expected Double Tensor
    image = image.float()

    # Add batch dimension PyTorch is expecting
    image.unsqueeze_(0)

    # model.to(device)
    model.eval()

    with torch.no_grad():
        output = model.forward(image)

    # Our output is log_softmax, so we take the inverse (exp) to get back softmax distribution
    all_probs = torch.exp(output)

    # topk returns a tuple of (values, indices)
    topk_tuple = torch.topk(all_probs, topk)

    probs = topk_tuple[0]
    classes = topk_tuple[1]

    return probs, classes


# Set image_path based on command line positional argument
image_path = results.path_to_image

probs, classes = predict(image_path, model)

# Convert to numpy and grab inner list
# Numpy doesn't support CUDA, so need to copy to CPU first
probs = probs.cpu()
probs2 = probs[0].numpy()
# print('Probs: ', probs2)

classes = classes.cpu()
classes2 = classes[0].numpy()

# Convert indices to the appropriate class indices
lookup_dict = {v: k for k, v in model.class_to_idx.items()}
new_classes = [lookup_dict[i] for i in classes2]

# Print out class labels and probabilities
if category_names_var is None:
    print('Class Indice, Class Probability:')
    print(list(zip(new_classes, probs2)))
else:
    # Convert indices to class names
    class_names = [cat_to_name[i] for i in new_classes]

    # Correct classification name. Finds the first digits (that's the [0] part) in the image path
    num_from_title_img_path = re.findall(r'\d+', image_path)[0]
    correct_name = cat_to_name[num_from_title_img_path]
    print('Correct Class Name: ', correct_name)
    print('Class Name, Class Probability:')
    print(list(zip(class_names, probs2)))
