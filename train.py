import argparse
import numpy as np
import torch
from torch import optim
from torchvision import datasets, transforms, models
from model import *
from preprocessing import *
from modelcheckpoint import *


if __name__ == '__main__':
    pretrained_models = {'squeezenet': models.squeezenet1_0,
                        'vgg16': models.vgg16,
                        'densenet': models.densenet121}
    
    #PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description='Train a deep neural network for image classification.')
    parser.add_argument('data_directory', help='directory containing images under train, valid and test directories.')
    parser.add_argument('--save_dir', dest='save_directory',  help='directory for saving checkpoints.')
    parser.add_argument('--arch', choices=pretrained_models.keys(), help='architecture of pretrained network used to generate features.', default='vgg16')
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--hidden_units', nargs='*',default=[512], type=int, help='sequence of integers for number of units in hidden layers')
    parser.add_argument('--gpu', action='store_const', const='cuda', default='cpu')
    args = parser.parse_args()
    print(args)
    
    data_dir = args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_sections = ['training','validation','testing']
    
    # Defining transforms for the training, validation, and testing sets
    data_transforms = {'training': transforms.Compose([transforms.RandomRotation(30),
                                                       transforms.RandomResizedCrop(224),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                                            [0.229, 0.224, 0.225])]),
                       'validation': transforms.Compose([transforms.Resize(255),
                                                          transforms.CenterCrop(224),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                                [0.229, 0.224, 0.225])]),
                       'testing': transforms.Compose([transforms.Resize(255),
                                                      transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                                            [0.229, 0.224, 0.225])])
                      }


    # Loading the datasets with ImageFolder
    image_datasets = {}
    image_datasets['training'] = datasets.ImageFolder(train_dir, transform=data_transforms['training'])
    image_datasets['validation'] = datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    image_datasets['testing'] = datasets.ImageFolder(test_dir, transform=data_transforms['testing'])

    # Using the image datasets and the trainforms, to define the dataloaders
    dataloaders = {}
    for section in data_sections:
        dataloaders[section] = torch.utils.data.DataLoader(image_datasets[section], batch_size=32, shuffle=True)

    #building model
    feature_model = pretrained_models[args.arch](pretrained=True)
    feature_model.features.name = args.arch
    
    if args.arch == 'vgg16':
        input_size = model.classifier[0].in_features
    elif args.arch == 'densenet':
        input_size = model.classifier.in_features
    elif args.arch == 'squeezenet':
        input_size = model.classifier[1].in_channels
    #elif args.arch == 'resnet' or args.arch == 'inception' :
        #input_size = model.fc.in_features
    
    output_size = len(image_datasets['training'].class_to_idx)
    output_model = FCNetwork([input_size] + args.hidden_units + [output_size])
    output_model.class_to_idx = image_datasets['training'].class_to_idx
    output_model.idx_to_class = {v:k for k,v in image_datasets['training'].class_to_idx.items()}
    #defining optimizer
    optimizer = optim.Adam(output_model.parameters(),lr=args.learning_rate)
    
    #training model
    model, scores = train_model((feature_model.features, output_model), dataloaders['training'], optimizer, nn.CrossEntropyLoss(), epochs=args.epochs, device=args.gpu, validationloader=dataloaders['validation'], print_every=1, eval_batches=1)
    
    filename='checkpoint'
    if args.save_directory is not None:
        filepath=args.save_directory+'/'+filename
    else:
        filepath = filename
        
    try:
        save_checkpoint(model, optimizer, args.epochs, filename=filepath)
    except:
        print('Unable to save in '+args.save_directory)
        save_checkpoint(model, optimizer, args.epochs, filename=filename)