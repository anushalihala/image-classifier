import torch
from torchvision import models
import time
from model import *

def save_checkpoint(model, optimizer, epoch, filename='checkpoint'):
    checkpoint = {'feature_model':model.feature_model.name,
                  'arch': model.output_model.get_layer_units(),             
                  'state_dict': model.output_model.state_dict(),
                  'class_to_idx': model.output_model.class_to_idx,
                  'epoch': epoch,
                  'optimizer': optimizer.state_dict(),
                  'time':int(time.time())}
    torch.save(checkpoint, filename+'.pth')
    
    
def load_checkpoint(filepath, feature_model_dict=None, optimizer = None, device=None):
    if feature_model_dict is None:
        feature_model_dict = {'resnet18': models.resnet18,
                            'alexnet': models.alexnet,
                            'squeezenet': models.squeezenet1_0,
                            'vgg16': models.vgg16,
                            'densenet': models.densenet121,
                            'inception': models.inception_v3}
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == 'cpu':
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(filepath)
        
    model = FCNetwork(checkpoint['arch'])
    model.load_state_dict(checkpoint['state_dict'])
    
    
    if feature_model_dict is not None:
        model = nn.Sequential(OrderedDict([('feature_model', feature_model_dict[checkpoint['feature_model']](pretrained=True).features ),
                                           ('output_model', model)]))
        model.feature_model.name = checkpoint['feature_model']
        model.output_model.class_to_idx = checkpoint['class_to_idx']
        model.output_model.idx_to_class = {v:k for k,v in checkpoint['class_to_idx'].items()}
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    return model,optimizer,checkpoint['epoch']
