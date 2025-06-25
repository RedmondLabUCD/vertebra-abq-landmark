from __future__ import print_function
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import shutil
from glob import glob
import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import models
from utils.earlystopping import EarlyStopping
from utils import datasets
from torchvision.datasets.utils import list_files
from utils.params import Params
from utils.plotting import plot_training
from utils.train_progress_tools import run_train_generator, track_running_average_loss, monitor_progress
import utils.eval_metrics as e_metric
from utils.data_prep import final_mean_and_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name",type=str,help="Pass name of model as defined in hparams.yaml.")
    parser.add_argument("--AUG",required=False,default=False,help="Set to true to use AUG images.")
    parser.add_argument("--AUG2",required=False,default=False,help="Set to true to use AUG images.")
    parser.add_argument("--roi",required=False,type=str,default=None,help="Uses ROI predictions as base.")
    parser.add_argument("--k",required=False,type=int,default=10,help="Number of times to train and evaluate model")
    parser.add_argument("--baseAUG",required=False,default=False,help="Use Augmented ROI predictions as base.")
    parser.add_argument("--alt_loss",required=False,default=False,help="Use Augmented ROI predictions as base.")
    parser.add_argument("--clahe",required=False,default=False,help="Set to true to use CLAHE images.")
    parser.add_argument("--attn",required=False,default=False,help="Set to true to use Attn UNet images as base.")
    parser.add_argument("--cl",required=False,default=False,help="Set to true to use the UNet with Custom Loss as base.")
    parser.add_argument("--ckpt",required=False,default=False,help="Set to true to use the UNet with Custom Loss as base.")
    args = parser.parse_args()

    # Parse our YAML file which has our model parameters. 
    params = Params("hparams.yaml", args.model_name)
   
    # # Use GPU if available
    # use_gpu= torch.cuda.is_available()
    # device = torch.device("cuda" if use_gpu else "cpu")
    
    # Load model that has been chosen via the command line arguments. 
    model_module = __import__('.'.join(['models', params.model_name]), fromlist=['object'])
    
    # Define evaluation metric
    eval_metric = getattr(e_metric, params.test_eval_metric)
    metrics = eval_metric()

    if args.ckpt:
        params.checkpoint_dir = str(args.ckpt)
        
     # Get root for dataset
    root = '//data/scratch/r094879/data'

    csv_file = os.path.join(root,'annotations/annotations.csv')
    csv_df = pd.read_csv(csv_file)
    csv_df = csv_df.sort_values(by=['id'], ascending=True).reset_index(drop=True)

    train_over = []
    val_over = []
    test_over = []

    train_id = 0
    val_id = 0

    for index, row in csv_df.iterrows():
        image_name = row['image']

        if index < int(0.10*len(csv_df)):
            train_over.append(image_name)
            train_id = row['id']
        elif index < int(0.20*len(csv_df)):
            if int(row['id']) == int(train_id):
                train_over.append(image_name)
            else:
                val_over.append(image_name)
                val_id = row['id']
        elif index >= int(0.90*len(csv_df)):
            if int(row['id']) == int(val_id):
                val_over.append(image_name)
            else:
                test_over.append(image_name)

    train = []
    val = []
    test = []

    all_files = list_files(os.path.join(root,params.target_dir),params.target_sfx)

    for filename in all_files:
        if any(keyword in filename for keyword in train_over):
            filename = filename.split('//')[-1]
            filename = filename[:-4]
            train.append(filename)
        if any(keyword in filename for keyword in val_over):
            filename = filename.split('//')[-1]
            filename = filename[:-4]
            val.append(filename)
        if any(keyword in filename for keyword in test_over):
            filename = filename.split('//')[-1]
            filename = filename[:-4]
            test.append(filename)
        
    Dataset = getattr(datasets,"HipSegDatasetTEST")
    
    extra = ""
    extra2 = extra
           
    if args.clahe:
        params.image_dir = params.image_dir + " CLAHE"
        extra = extra + "_clahe"
                        
    if args.cl:
        extra2 = extra2 + "_CL"
    
    # Make directories to save results 
    prediction_save = os.path.join(root,"Results",args.model_name,
                                   "Predicted" + extra + " " + params.target_dir)
    if not os.path.exists(prediction_save): os.makedirs(prediction_save)
    
    # if args.roi:
    #     prediction_roi = os.path.join(root,"Results",args.model_name,
    #                                    "Predicted" + extra2 + " Pred " + params.target_dir)
    #     if not os.path.exists(prediction_roi): os.makedirs(prediction_roi)
    #     params_roi = Params("hparams.yaml", args.model_name)
    #     insize = ""
    #     if args.attn:
    #         insize = insize + "_Attn"
    #     if args.cl:
    #         insize = insize + "_CL"
    #     if args.baseAUG:
    #         insize = insize + "_AUG"
            
    #     params_roi.image_dir = "Predicted" + insize + " " + params.image_dir 
    #     params_roi.target_dir = "Predicted" + insize + " " + params.target_dir
    
    # if args.AUG2:
    #     params.target_dir = params.target_dir + " AUG2"
    #     params.image_dir = params.image_dir + " AUG2"
    
    # ==================== EVALUATE MODEL FOR EACH FOLD ====================
        
     # Empty to hold the results of each fold
    acc_scores = []
    pred_acc_scores = []
    best_epochs = []
        
    # Calculate mean and std for dataset normalization 
    norm_mean,norm_std = final_mean_and_std(root,params)

    # Define transform for images
    transform=transforms.Compose([transforms.Resize((256,256)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=norm_mean,std=norm_std)
                                  ])

    # Define test dataset
    test_data = Dataset(root,test,params.image_dir,input_tf=transform)

    if args.roi:
        pred_test_data = Dataset(data_dir_test,params_roi.image_dir,input_tf=transform)
        pred_test_loader = DataLoader(pred_test_data, batch_size=1, shuffle=False, pin_memory=False)

    # Define dataloaders
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=False)

    # Define model
    model = model_module.net(params.num_classes)

    # Grap test function for your network.
    final_test = model_module.final_test

    # Load relevant checkpoint for the fold
    chkpt = os.path.join(root,params.checkpoint_dir,"chkpt_{}".format(args.model_name+extra+"_lr0001"))

    acc = final_test(model,test_loader,metrics,params,checkpoint=chkpt,name=args.model_name,extra=extra, 
               prediction_dir=prediction_save)
    print("Test Accuracy: {}".format(acc))

    if args.roi:
        pred_acc = final_test(model, device, pred_test_loader, metrics, params_roi, checkpoint=chkpt, 
                        name=args.model_name, extra="_Pred"+extra2, prediction_dir=prediction_roi)
        print("Test Accuracy: {}".format(pred_acc))


if __name__ == '__main__':

    main()