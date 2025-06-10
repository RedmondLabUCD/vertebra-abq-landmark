from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tF
import torch.optim as optim
from tqdm import tqdm
from utils.train_progress_tools import RunningAverage
from PIL import Image, ImageFilter, ImageOps
from utils.feature_extraction import get_contours
import cv2 as cv
from utils.femhead_post_process import dice_post_process, test_dice_post_process
from utils.lm_post_process import lm_post_process, roi_lm_post_process, final_lm_post_process, final_roi_lm_post_process

'''Model edited from PyTorch implementation at: github.com/milesial/Pytorch-UNet/tree/master/unet'''


def double_conv(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )   

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)

    def forward(self, x):
        return self.conv(x)

class net(nn.Module):

    def __init__(self,num_classes):
        super(net, self).__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512) 
        self.dconv_down5 = double_conv(512, 1024 // 2)         

        self.maxpool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True) 

        self.dconv_up4 = double_conv(1024, 512 // 2)
        self.dconv_up3 = double_conv(512, 256 // 2)
        self.dconv_up2 = double_conv(256, 128 // 2)
        self.dconv_up1 = double_conv(128, 64)
        
        self.conv_last = OutConv(64, num_classes)
        
    def forward(self, x):
        # level 1 - encode
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        # level 2 - encode
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        # level 3 - encode
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        # level 4 - encode
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4) 

        # level 5 - encode/decode
        x = self.dconv_down5(x)
        
        # level 4 - decode
        x = self.up(x)        
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        
        # level 3 - decode
        x = self.up(x)        
        x = torch.cat([x, conv3], dim=1)  
        x = self.dconv_up3(x)     

        # level 2 - decode
        x = self.up(x)        
        x = torch.cat([x, conv2], dim=1) 
        x = self.dconv_up2(x)  

        # level 1 - decode
        x = self.up(x)  
        x = torch.cat([x, conv1], dim=1)  
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return torch.sigmoid(out)
    
    
def train(model, loader, optimizer, criterion, params=None):
    n_steps = len(loader)  
    model.train()

    # iterate over batches
    for step, (batch, targets, filenames) in enumerate(loader):
        # batch = batch.to(device)
        # targets = targets.to(device)
        optimizer.zero_grad() # clear previous gradient computation
        predictions = model(batch) # forward propagation  
        loss = criterion(targets, predictions) # calculate the loss
        loss.backward() # backpropagate to compute gradients
        optimizer.step() # update model weights
        yield step, n_steps, float(loss)
        
        
def val(model, loader, criterion, eval_metric, params, checkpoint=None):
    if checkpoint is not None:
#         load_checkpoint(optimizer=None, model, checkpoint)
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state['model']) 
        
    model.eval()
    # use a running average to keep track of the average loss
    valid_loss = RunningAverage(count=len(loader))
    metrics = []
    # Don't need gradients for validation, so wrap in no_grad to save memory
    with torch.no_grad(): # prevent tracking history (and using memory)
        for step, (batch, targets, full_filenames) in enumerate(loader):
            # batch = batch.to(device)
            # targets = targets.to(device)
            predictions = model(batch) # forward propagation
            loss = criterion(predictions, targets) # calculate the loss
            valid_loss.update(loss) # update running loss value
            
            # Get filename
            filenames = full_filenames[0]
            filename = filenames.split("\\")[-1]
            metric_avg = eval_metric(targets,predictions,filename,params)
            metrics.append(metric_avg)
    acc = sum(metrics)/len(metrics)
    return valid_loss.value, acc


def test(model, loader, eval_metric, params, checkpoint=None, name=None, extra=None,
         prediction_dir=None,test_names=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state['model']) 
        
    model.eval()
    metrics = []
    
    # root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    # data_dir = os.path.join(root,"Results","Statistics") 
    
    # csv_name = None
    # count = [0,0,0,0]
    # stats_df = pd.DataFrame(columns=['image','T4x','T4y','T4_val','T5x','T5y','T5_val','T6x','T6y','T6_val','T7x',
    #                                  'T7y','T7_val','T8x','T8y','T8_val','T9x','T9y','T9_val','T10x','T10y','T10_val',
    #                                  'T11x','T11y','T11_val','T12x','T12y','T12_val','L1x','L1y','L1_val',
    #                                  'L2x','L2y','L2_val','L3x','L3y','L3_val','L4x','L4y','L4_val','RMSE','correct',
    #                                  'total','id_acc'])

    # stats_df['image'] = test_names
    # stats_df.to_csv(os.path.join('//data/scratch/r094879/data/stats',name+'.csv'),index=False)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, full_filenames) in enumerate(loader):
            # inputs = inputs.to(device)
            # targets = targets.to(device)
            predictions = model(inputs)
            filenames = full_filenames[0]
            filename = filenames.split("\\")[-1]
            metric_avg = eval_metric(targets,predictions,filename,params,name)
            metrics.append(metric_avg)
#             if prediction_dir is not None:
#                 if "ROI_LM" in str(name):
#                     csv_name = roi_lm_post_process(name,extra,root,data_dir,params,prediction_dir,
#                                                    predictions,filename,fold_num,metric_avg,csv_name)
#                 elif "LM" in str(name):
#                     count = lm_post_process(name,extra,root,data_dir,params,prediction_dir,
#                                             predictions,filename,fold_num,metric_avg,count=count)
#                 elif "FemHead" or "UNet_ROI" in str(name): 
# #                     dice_post_process(name,extra,root,data_dir,params,prediction_dir,
# #                                       predictions,filename,fold_num,metric_avg)
#                     test_dice_post_process(name,extra,root,data_dir,params,prediction_dir,
#                                       predictions,filename,metric_avg)

    acc = sum(metrics)/len(metrics)        
    return acc


def final_test(model, device, loader, eval_metric, params, checkpoint=None, name=None, extra=None,
         prediction_dir=None, AUG=False):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state['model']) 
        model.to(device)
        
    model.eval()
    metrics = []
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_dir = os.path.join(root,"Results","Statistics") 
    
    csv_name = None
    count = [0,0,0,0]
    
    with torch.no_grad():
        for batch_idx, (inputs, full_filenames) in enumerate(loader):
            inputs = inputs.to(device)
            predictions = model(inputs)
            filenames = full_filenames[0][:-4]
            filename = filenames.split("\\")[-1]
            metric_avg = eval_metric(predictions,filename,params,AUG,prediction_dir,square=False)
            metrics.append(metric_avg)
            if prediction_dir is not None:
                if "ROI_LM" in str(name):
                    csv_name = final_roi_lm_post_process(name,extra,root,data_dir,params,prediction_dir,
                                                   predictions,filename,metric_avg,csv_name)
                elif "LM" in str(name):
                    count = final_lm_post_process(name,extra,root,data_dir,params,prediction_dir,
                                            predictions,filename,metric_avg,count=count)

    acc = sum(metrics)/len(metrics)        
    return acc
        