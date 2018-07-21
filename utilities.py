# Utility functions for loading data and preprocessing images
import numpy as np
from PIL import Image

# Image preprocessing
def process_image(image, size=256):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    # I based shortest side logic off this post:
    # https://stackoverflow.com/questions/4321290/how-do-i-make-pil-take-into-account-the-shortest-side-when-creating-a-thumbnail

    # Load image
    img = Image.open(image)

    # Get dimensions of image
    width, height = img.size

    # Keep aspect ratio using shortest side
    # Resize image. Thumbnail works in place
    if width == height:
        img.thumbnail((size, size))

    # For ratio, we need a number larger than 1 because we are shrinking
    # the short side to 256 and the other side needs to be larger than
    # that when we multiply size * ratio.
    elif height > width:
        ratio = float(height) / float(width)
        new_height = ratio * size
        img = img.resize((size, int(np.floor(new_height))))

    elif width > height:
        ratio = float(width) / float(height)
        new_width = ratio * size
        img = img.resize((int(np.floor(new_width)), size))

    # Get updated image dimensions after shrinking original image
    width, height = img.size

    # Grab cropping locations for img.crop
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2

    # Crop center
    img = img.crop((left, top, right, bottom))

    # Convert PIL image to Numpy array
    np_image = np.array(img)

    # Convert values to floats between 0 and 1
    np_image = np_image / 255.0

    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Move color channel from third position to first position for PyTorch
    # Numpy image: H x W x C
    # Torch image: C X H x W
    np_image = np_image.transpose((2, 0, 1))

    return np_image
