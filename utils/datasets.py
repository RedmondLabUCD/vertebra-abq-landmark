import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import list_files
import random
import pydicom
import cv2 as cv
from pydicom import dcmread


class SpineDataset(Dataset):
    def __init__(self,root,filenames,image_dir,target_dir,target_sfx='.png',
                 input_tf=None,
                 output_tf=None,
                 loader=pil_loader):
        self.root = root
        self.input_tf = input_tf
        self.output_tf = output_tf
        self.loader = loader
        self.target_sfx = target_sfx
        self.input_dir = os.path.join(self.root,image_dir) #set image directory
        self.output_dir = os.path.join(self.root,target_dir) 
        self.file_list = filenames
            
    def __getitem__(self, index): #getitem method
        filename = self.file_list[index]
        input_filename = os.path.join(self.input_dir, filename+'.png')
        output_filename = os.path.join(self.output_dir, filename+'.npy')

        # dicom_image = dcmread(input_filename)
        # input = dicom_image.pixel_array

        # images_to_rotate = {"1.3.6.1.4.1.23849.1523372706.11.1634383154459843750.3.1.1","1.3.6.1.4.1.23849.2292492336.11.1634353588374204427.3.1.1",
        #                 "1.3.6.1.4.1.23849.3070783366.11.1634412454096093750.3.1.1","1.3.6.1.4.1.23849.3292118434.11.1634302616143596860.3.1.1",
        #                 "1.3.6.1.4.1.23849.3482538762.11.1634445082479062500.3.1.1","1.3.6.1.4.1.23849.4190597982.11.1634445083927187500.3.1.1",
        #                 "1.3.6.1.4.1.23849.2872276509.11.1634353586206235677.3.1.1","1.3.6.1.4.1.23849.3047549382.11.1634226912973281250.3.1.1",
        #                 "1.3.6.1.4.1.23849.3088759028.11.1634357890243125000.3.1.1","1.3.6.1.4.1.23849.3715794974.11.1634412455716875000.3.1.1",
        #                 "1.3.6.1.4.1.23849.2498714632.11.1634302614592034360.3.1.1"}

        # if str(filename) in images_to_rotate:
        #     input = np.rot90(input,2)
        #     if str(filename) == "1.3.6.1.4.1.23849.3047549382.11.1634226912973281250.3.1.1":
        #         input = np.flip(input,1)
            
        # input = input.astype(float)
        # input = (input-input.min())/(input.max()-input.min())*255.0
        # input = input.astype(np.uint8)
        # input = Image.fromarray(input)
        # input = input.resize((256,256))
        
        input = self.loader(input_filename)
        output = np.load(output_filename) 
        if self.input_tf is not None: 
            input = self.input_tf(input)
        if self.output_tf is not None:
            output = self.output_tf(output)
        return input, output, filename
    
    def __len__(self):
        return len(self.file_list) # returns the length of the data file list
    
    
class HipSegDatasetTEST(Dataset):
    def __init__(self,root,filenames,image_dir,
                 input_tf=None,
                 loader=pil_loader):
        self.root = root
        self.input_tf = input_tf
        self.loader = loader
        self.input_dir = os.path.join(self.root,image_dir) #set image directory
        self.file_list = filenames
            
    def __getitem__(self, index): #getitem method
        filename = self.file_list[index]
        input_filename = os.path.join(self.input_dir, filename+'.png')
        # Load target and image
        input = self.loader(input_filename) # converts to PIL image RGB
        # Apply transforms if given
        if self.input_tf is not None: 
            input = self.input_tf(input)
        return input, filename
    
    def __len__(self):
        return len(self.file_list) # returns the length of the data file list
    
    
