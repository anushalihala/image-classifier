import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)

    w,h = im.size
    if w<h:
        new_w = 256
        new_h = new_w*(h/w)    
    else: 
        new_h = 256
        new_w = new_h*(w/h)          
    im.thumbnail((new_w,new_h))

    im = im.crop((new_w//2 - 224//2, new_h//2 - 224//2, new_w//2 + 224//2, new_h//2 + 224//2))

    np_image = np.array(im)/255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    np_image = np_image.transpose((2,0,1))
    
    return np_image


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax