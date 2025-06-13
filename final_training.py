from __future__ import print_function
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import shutil
from glob import glob
import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
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
    parser.add_argument("--lr",required=False,type=float,default=0.0001,help="Specify new learning rate")
    parser.add_argument("--k",required=False,type=int,default=10,help="Number of times to train and evaluate model")
    parser.add_argument("--roi",required=False,type=str,default=None,help="Uses ROI predictions as base.")
    parser.add_argument("--custom_loss",required=False,default=False,help="Use custom loss function.")
    parser.add_argument("--ckpt",required=False,default=False,help="Set a checkpoint folder.")
    parser.add_argument("--checkpoint",required=False,default=False,help="Set a checkpoint folder.")
    args = parser.parse_args()
    
    # Parse our YAML file which has our model parameters. 
    params = Params("hparams.yaml", args.model_name)
    
    # # Use GPU if available
    # use_gpu = torch.cuda.is_available()
    # device = torch.device("cuda" if use_gpu else "cpu")
    
    # Set up "extra" bit to add to checkpoint name (i.e. specifying if it was pretrained, data augmented, etc.)
    extra = ""
    
    # Load model that has been chosen via the command line arguments. 
    model_module = __import__('.'.join(['models', params.model_name]), fromlist=['object'])
    
    # Specify learning rate
    split = str(args.lr).split('.')[-1]
    extra = extra + '_lr' + split
    
    # Define loss function 
    if args.custom_loss:
        loss_func = getattr(e_metric, params.loss)
    else:
        loss_func = getattr(nn, params.loss)
    criterion = loss_func()

    if args.ckpt:
        params.checkpoint_dir = str(args.ckpt)
    
    # Define evaluation metric
    eval_metric = getattr(e_metric, params.eval_metric)
    metrics = eval_metric()

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

        if index < int(0.05*len(csv_df)):
            train_over.append(image_name)
            train_id = row['id']
        elif index < int(0.75*len(csv_df)):
            if int(row['id']) == int(train_id):
                train_over.append(image_name)
            else:
                val_over.append(image_name)
                val_id = row['id']
        elif index >= int(0.99*len(csv_df)):
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

    Dataset = getattr(datasets,params.dataset_class)
    
    # Make directories to save results 
    if not os.path.exists(os.path.join(root,params.log_dir)): os.makedirs(os.path.join(root,params.log_dir))
    if not os.path.exists(os.path.join(root,params.checkpoint_dir)): os.makedirs(os.path.join(root,params.checkpoint_dir))
    if not os.path.exists(os.path.join(root,"figs")): os.makedirs(os.path.join(root,"figs"))
    
    # Empty to hold the results of each fold
    acc_scores = []
    pred_acc_scores = []
    best_epochs = []
        
    # # Calculate mean and std for dataset normalization 
    # norm_mean,norm_std = final_mean_and_std(root,params)

    # # Define transform for images
    # transform=transforms.Compose([transforms.Resize((256,256)),
    #                               transforms.ToTensor(),
    #                               transforms.Normalize(mean=norm_mean,std=norm_std)
    #                               ])

    # Define transform for images
    transform=transforms.Compose([transforms.Resize((256,256)),
                                  transforms.ToTensor()
                                  ])

    # Set up transforms for targets
    target_transform = transforms.ToTensor()

    val_data = Dataset(root,val,params.image_dir,params.target_dir,target_sfx=params.target_sfx,
                       input_tf=transform,output_tf=target_transform)

    # Define training dataset
    train_data = Dataset(root,train,params.image_dir,params.target_dir,target_sfx=params.target_sfx,
                                    input_tf=transform,output_tf=target_transform)

    # Define dataloaders
    train_loader = DataLoader(train_data, batch_size=params.batch_size, shuffle=True, pin_memory=False)
    val_loader_eval = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=False)

    # Define model and optimizer
    model = model_module.net(params.num_classes)
    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay= 0.0001)

    # Grap your training, val, and test functions for your network.
    train = model_module.train
    val = model_module.val

    if args.checkpoint:
        ckpt_name = os.path.join(root,args.ckpt,"chkpt_{}".format(args.model_name+extra))
        model_state = torch.load(ckpt_name)
        model.load_state_dict(model_state['model']) 

    val_accs = []
    val_losses = []
    train_losses = []
    train_accs = []

    # Initialize early stopping
    early_stopping = EarlyStopping(verbose=True, patience=20, up=params.early_stopping_up, path=os.path.join(root,params.checkpoint_dir))

    # ==================== TRAIN THE MODEL FOR ONE FOLD ====================

    epoch_max = 100
    for epoch in range(1,epoch_max):
        print("Epoch: {}".format(epoch))
        # Call training function. 
        train_generator = train(model, train_loader, optimizer, criterion, params)
        train_generator = track_running_average_loss(train_generator)
        train_generator = monitor_progress(train_generator)
        train_loss_decay = run_train_generator(train_generator)

        # Evaluate on both the training and validation set.
        val_loss, val_acc = val(model, val_loader_eval, criterion, metrics, params)

        # Collect some data for logging purposes. 
        val_losses.append(float(val_loss))
        val_accs.append(val_acc)

        early_stopping(val_acc, model, optimizer, args.model_name, extra, epoch)
        
        print('\n\tval Loss: {:.6f}\tval acc: {:.6f}'.format(val_loss, val_acc))   

        if early_stopping.early_stop:
            print("Early stopping")
            fig = plot_training(train_losses,train_accs,val_losses,val_accs,model_name=args.model_name+extra)
            fig.savefig(os.path.join(root,"figs", "{}_training_vis".format(args.model_name+extra)))
            break

    # Define "best" epoch
    if params.early_stopping_up:
        best_val_epoch = int(np.argmax(val_accs)+1)
    else:
        best_val_epoch = int(np.argmin(val_accs)+1)
    best_epochs.append(best_val_epoch)

    # ==================== SAVE RELEVANT INFORMATION FOR THE MODEL ====================

    # Some log information to help you keep track of your model information. 
    logs = {
        "model": args.model_name,
#             "train_losses": train_losses,
#             "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "best_val_epoch": best_val_epoch,
        "lr": args.lr,
        "batch_size": params.batch_size,
        "norm_mean": norm_mean,
        "norm_std": norm_std
    }

    with open(os.path.join(root,params.log_dir,"{}{}.json".format(args.model_name, extra)), 'w') as f:
        json.dump(logs, f)

    chkpt = os.path.join(root,params.checkpoint_dir,"chkpt_{}".format(args.model_name+extra))

if __name__ == '__main__':

    main()