from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
from PIL import Image
from models import *

feature_extract = True

def main(args):
    # Initialize the model for this run
    model_ft, input_size = initialize_model("resnet", 6, feature_extract, use_pretrained=False)

    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Detect if we have a GPU available
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Send the model to GPU
    model_ft.load_state_dict(torch.load(args['weight']))
        
    since = time.time()

    model_ft.eval()   # Set model to evaluate mode

    image = Image.open(args['input'], "r")
    image = image.resize((224,224))
    image = transforms.ToTensor()(image).unsqueeze_(0)
    image = image.to(device, dtype=torch.float)

    outputs = model_ft(image)

    _, preds = torch.max(outputs, 1)

    time_elapsed = time.time() - since
    print('process time::',time_elapsed)
    print('predicted number::', preds.item()+3)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input",help="path to input image", default = "input/sample.png")
    ap.add_argument("-w", "--weight",help="path to weight", default = "./weights/best.pth")
    args = vars(ap.parse_args())

    main(args)