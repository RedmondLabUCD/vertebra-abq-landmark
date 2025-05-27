import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import cv2
from numpy import fliplr
import math
from tqdm import tqdm
from PIL import Image,ImageEnhance
from scipy import ndimage
from skimage import io
from utils import datasets
from utils.landmark_prep import prep_landmarks
from torchvision.datasets.utils import list_files


def mean_and_std(index, data_dir, params):
    '''
    Calculates mean and standard deviation of images to be 
    used in image normalization.
    Inspired by: towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
    '''
    Dataset = getattr(datasets,"SpineDataset")
    
    # Define basic transform (resize and make tensor)
    transform = transforms.Compose([transforms.Resize((params.input_size,params.input_size)),
                                    transforms.ToTensor()])
    
    # Set up transforms for targets
    if "Masks" in params.target_dir:
        target_transform = transforms.Compose([transforms.Grayscale(),
                                               transforms.Resize((params.input_size,params.input_size)),
                                               transforms.ToTensor()
                                               ])
    else:
        target_transform = transforms.ToTensor()
        
    # Define validation dataset
    if index+1 > 10:
        val_index = 1
    else:
        val_index = index + 1

    # Define and load training dataset
    train_data = []
    for i in range(1,11):
        if i != index and i != val_index:
            if AUG:
                fold_data = Dataset(data_dir,i,params.image_dir+" AUG",params.target_dir+" AUG",
                                    target_sfx=params.target_sfx,input_tf=transform,output_tf=target_transform)
            else:
                fold_data = Dataset(data_dir,i,params.image_dir,params.target_dir,target_sfx=params.target_sfx,
                                    input_tf=transform,output_tf=target_transform)
            train_data = ConcatDataset([train_data, fold_data])

    loader = DataLoader(train_data,batch_size=params.batch_size,shuffle=False)
    
    # Calculate mean and std for each batch 
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _, _ in loader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    # Get mean and std across the batches
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    mean = mean.cpu().numpy()
    mean2 = [1.*mean[0],1.*mean[1],1.*mean[2]]
    
    std = std.cpu().numpy()
    std2 = [1.*std[0],1.*std[1],1.*std[2]]

    return mean2, std2


def final_mean_and_std(data_dir, params):
    '''
    Calculates mean and standard deviation of images to be 
    used in image normalization.
    Inspired by: towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
    '''
    Dataset = getattr(datasets,"SpineDataset")
    
    # Define basic transform (resize and make tensor)
    transform=transforms.Compose([transforms.Resize(256),
                                  transforms.ToTensor()])

    csv_file = os.path.join(data_dir,'annotations/annotations.csv')
    csv_df = pd.read_csv(csv_file)

    train_over = []
    val_over = []
    test_over = []

    train_id = 0
    val_id = 0

    for index, row in csv_df.iterrows():
        image_name = row['image']

        if index < int(0.8*len(csv_df)):
            train_over.append(image_name)
            train_id = row['id']
        elif index < int(0.9*len(csv_df)):
            if int(row['id']) == int(train_id):
                train_over.append(image_name)
            else:
                val_over.append(image_name)
                val_id = row['id']
        elif index >= int(0.9*len(csv_df)):
            if int(row['id']) == int(val_id):
                val_over.append(image_name)
            else:
                test_over.append(image_name)

    train = []
    val = []
    test = []

    all_files = list_files(os.path.join(data_dir,params.target_dir),params.target_sfx)

    print(all_files)
    for filename in all_files:
        if any(keyword in filename for keyword in train_over):
            filename = filename.split('//')[-1]
            filename = filename[:-4]
            train.append(filename)
        elif any(keyword in filename for keyword in val_over):
            filename = filename.split('//')[-1]
            filename = filename[:-4]
            val.append(filename)
        elif any(keyword in filename for keyword in test_over):
            filename = filename.split('//')[-1]
            filename = filename[:-4]
            test.append(filename)

    # Define and load training dataset
    train_data = Dataset(data_dir,train,params.image_dir,params.target_dir,target_sfx=params.target_sfx,
                                input_tf=transform,output_tf=transform)

    loader = DataLoader(train_data,batch_size=params.batch_size,shuffle=False)
    
    # Calculate mean and std for each batch 
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _, _ in loader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    # Get mean and std across the batches
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    mean = mean.cpu().numpy()
    mean2 = 1.*mean[0]
    
    std = std.cpu().numpy()
    std2 = 1.*std[0]

    return mean2, std2
