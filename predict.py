import argparse
import json
import torch
from torch.nn import functional as F
from preprocessing import *
from modelcheckpoint import *

def predict(image_path, model, topk=5, device=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    np_image = process_image(image_path)
    np_image = np.array([np_image])
    model,_,_ =  load_checkpoint(model, device=device)
    model.eval()
    model.to(device)
    output = model.forward(torch.from_numpy(np_image.astype(np.float32)).to(device) )
    output = F.softmax(output, dim=1)
    top_classes = torch.topk(output, topk)
    probs = top_classes[0].data.numpy()[0]
    indices = top_classes[1].data.numpy()[0]
    classes = np.array([model.output_model.idx_to_class[i] for i in indices])
    return probs,classes

if __name__ == '__main__':
    #PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description='Predict label for an image using given model.')
    parser.add_argument('input', help='image to classify.')
    parser.add_argument('checkpoint', help='saved model.')
    parser.add_argument('--top_k', default=1, type=int, help='output k most probable classes.')
    parser.add_argument('--category_names', help='JSON file mapping categories to class names.')
    parser.add_argument('--gpu', action='store_const', const='cuda', default='cpu')
    args = parser.parse_args()
    
    probs,classes = predict(args.input, args.checkpoint, topk=args.top_k, device=args.gpu)
    
    if args.category_names is not None:
        try:
            with open(args.category_names, 'r') as f:
                cat_to_name = json.load(f)
            classes = [cat_to_name[i] for i in classes]
        except:
            print("unable to parse file with category names")
        
    print('Most likely classes: ',classes)
    print('Probabilities: ',probs)
        
   