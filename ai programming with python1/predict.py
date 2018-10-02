import argparse
import torch
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import json
import numpy as np
from network_prep import load_model
from image_process import process_image

def get_input_args():
    parser = argparse.ArgumentParser(description='Get nn arguments')
    parser.add_argument('input', type=str, help='image to process and predict')
    parser.add_argument('checkpoint', type=str, help='cnn to load')
    parser.add_argument('--top_k', default=1, type=int, help='default top_k results')
    parser.add_argument('--category_names', default='', type=str, help='default category file')
    parser.add_argument('--gpu', default=False, action='store_true', help='use GPU processing')
    
    return parser.parse_args()

def predict(image, model, top_k, gpu, category_names, arch, class_idx):
    image = image.unsqueeze(0).float()
    image = Variable(image)
    
    if gpu and torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
        print('GPU PROCESSING')
    else:
        print('CPU PROCESSING')
    with torch.no_grad():
        out = model.forward(image)
        results = torch.exp(out).data.topk(top_k)
    classes = np.array(results[1][0], dtype=np.int)
    probs = Variable(results[0][0]).data
    
    if len(category_names) > 0:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
            
        mapped_names = {}
        for k in class_idx:
            mapped_names[cat_to_name[k]] = class_idx[k]
            
        mapped_names = {v:k for k,v in mapped_names.items()}
        
        classes = [mapped_names[x] for x in classes]
        probs = list(probs)
    else:
        class_idx = {v:k for k,v in class_idx.items()}
        classes = [class_idx[x] for x in classes]
        probs = list(probs)
    return classes, probs

def print_predict(classes, probs):
    predictions = list(zip(classes, probs))
    for i in range(len(predictions)):
        print('{} : {:.3f}'.format(predictions[i][0], predictions[i][1]))
    pass

def main():
    in_args = get_input_args()
    norm_image = process_image(in_args.input)
    model, arch, class_idx = load_model(in_args.checkpoint)
    classes, probs = predict(norm_image, model, in_args.top_k, in_args.gpu, in_args.category_names, arch, class_idx)
    print_predict(classes, probs)
    pass

if __name__ == '__main__':
    main()
        