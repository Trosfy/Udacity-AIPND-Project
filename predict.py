import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json

def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_path', type = str, default = 'flowers/train/1/image_06734.jpg', help = 'path to image that will be predicted', required=True)
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'path to checkpoint of trained model', required=True)
    parser.add_argument('--top_k', type = int, default = 3, help = 'top k results given out')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'path to file that maps category to real names')
    parser.add_argument('--gpu', action='store_true', help = 'enable gpu')
    
    
    return parser.parse_args()

def process_image(image, device):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)

    # resize, keeping aspect ratio
    image.thumbnail([256, 256])
    
    # center crop
    new_width = 224
    new_height = 224
    width, height = image.size

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    image = image.crop((left, top, right, bottom))
    
    # normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    normalized_img = transform(image)
    return normalized_img

def get_model():
    model = models.densenet161(pretrained=True)
    return model

def get_prediction(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = process_image(image_path, device).to(device)
    image = image[None, :, :, :]
    model.eval()
    with torch.no_grad():
        logps = model.forward(image)

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
    
    # reset to cpu if use gpu previously

    top_class = top_class.cpu().numpy()[0]
    probs = top_p.cpu().numpy()[0]
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    for i in range(len(top_class)):
        top_class[i] = int(idx_to_class[top_class[i]])
       
    return probs, top_class

def predict(image_path, checkpoint, top_k, category_names, gpu_enabled):
    # set device
    if gpu_enabled:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
        
    # load checkpoint
    loaded_checkpoint = torch.load(checkpoint)  
    
    # get pretrained model and set checkpoint 
    arch = loaded_checkpoint['arch']
    input_units = 0
    output_units = 102
    hidden_units = loaded_checkpoint['hidden']
    dropout = loaded_checkpoint['dropout']
    
    if arch == "densenet161":
        model = models.densenet161(pretrained=True)
        input_units = 2208
        classifier = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, output_units),
            nn.LogSoftmax(dim=1)
        )
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        input_units = 25088
        classifier = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, output_units),
            nn.LogSoftmax(dim=1)
        )
    classifier.load_state_dict(loaded_checkpoint['classifier'])
    model.classifier = classifier
    model.class_to_idx = loaded_checkpoint['class_to_idx']
    
    model.to(device)
    probs, temp_class = get_prediction(image_path, model, top_k, device)
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    classes = []
    for i in range(len(temp_class)):
        classes.append(cat_to_name[str(temp_class[i])])
    print("Predictions:")
    for i in range(top_k):
        print("%d. %s with %f%% confidence" % (i+1, classes[i], probs[i]*100))
    
def main():
    input_arg = get_input_args()
    print(input_arg)
    
    # set parameters
    image_path = input_arg.image_path
    checkpoint = input_arg.checkpoint = "checkpoint.pth"
    top_k = input_arg.top_k
    category_names = input_arg.category_names
    gpu_enabled = input_arg.gpu
    
    predict(image_path, checkpoint, top_k, category_names, gpu_enabled)

if __name__ == "__main__":
    main()