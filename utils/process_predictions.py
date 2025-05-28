#import libraries 
import numpy as np
import pandas as pd 
import os
import math
import itertools
import skimage
from glob import glob
from PIL import Image
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms.functional as tF
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.datasets.utils import list_files
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import gca
from matplotlib.axes import Axes

from utils import datasets
from utils.params import Params
from utils.landmark_prep import prep_landmarks
from utils.feature_extraction import get_contours, femhead_centre
from utils.roi_functions import resize_roi, reverse_resize_roi_lm, extract_ROI_from_lm
from scipy.spatial.distance import directed_hausdorff
    

def superimpose(filename,superimposed_dir,index,extra="",extra2="",extra3="",extra4="",model="UNet_FemHead",
                model2=None,model3=None,model4=None,roi=0,side="l",labels=["Target","UNet-Baseline"]):
    
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    
    # Get base image
    if roi == 1:
        img_dir = os.path.join(root,"Dataset","Fold "+str(index),"ROI")
        contl, contr = get_contours(os.path.join(root,"Dataset","Fold "+str(index),"ROI Masks",
                                                 filename+".png"))
        img = Image.open(os.path.join(img_dir,filename+"_"+str(side)+".png"))
        size=15
    else:
        img_dir = os.path.join(root,"Dataset","Images")
        img = Image.open(os.path.join(img_dir,filename+".png"))
        contl = None
        contr = None
        size=2
            
    img_size = np.asarray(img.size).astype(float)

    # Define figure and legends
    custom_lines = [Line2D([0], [0], color='c', lw=1),
                    Line2D([0], [0], color='r', lw=1),
                    Line2D([0], [0], color='g', lw=1),
                    Line2D([0], [0], color='m', lw=1),
                    Line2D([0], [0], color='b', lw=1)]
    colors = ['r','g','m','b']
    fig, ax = plt.subplots(figsize=(5,5),dpi=120)
    ax.legend(custom_lines,labels,prop={"size":8},borderpad=0.4)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
        
    # Get femoral head targets
    target_path = os.path.join(root,"Dataset","Fold "+str(index),"FemHead Masks",filename+'.png')
    fem_centr, fem_centl = femhead_centre(target_path)
    fem_target = [fem_centr, fem_centl]
    fem_targets = np.asarray(fem_target).astype(float)
    tar_cent = fem_targets.reshape((-1,2))

    tar_contl, tar_contr = get_contours(target_path)
    
    fem_target = [tar_contr, tar_contl]
    
    if roi != 0:
        if side == "l":
            tar_contl = [resize_roi(tar_contl,contl,left=True)]
            fem_target = tar_contl
            tar_cent = [resize_roi(tar_cent,contl,left=True)]
            tar_cent = tar_cent[0][1]
        else:
            tar_contr = [resize_roi(tar_contr,contr)]
            fem_target = tar_contr
            tar_cent = [resize_roi(tar_cent,contr)]
            tar_cent = tar_cent[0][0]

    for contour in fem_target:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='c')
    if roi == 0:
        plt.scatter(tar_cent[:, 1], tar_cent[:, 0], s=size, marker='.', c='c')
    if roi != 0:
        plt.scatter(tar_cent[0], tar_cent[1], s=size, marker='.', c='c')

    # Get femoral head predictions
    fem_pred_1, pred_cent_1 = prep_predictions(model,extra,filename,contl,contr,side,roi)
    for contour in fem_pred_1:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=colors[0]) 
    if roi == 0 and pred_cent_1 is not None:
        plt.scatter(pred_cent_1[:, 1], pred_cent_1[:, 0], s=size, marker='.', c=colors[0])
    if roi != 0 and pred_cent_1 is not None:
        plt.scatter(pred_cent_1[0], pred_cent_1[1], s=size, marker='.', c=colors[0])
    
    if roi == 0 and pred_cent_1 is not None:
        dist_r = math.sqrt((tar_cent[0,1]-pred_cent_1[0,1])**2 + (tar_cent[0,0]-pred_cent_1[0,0])**2)
        dist_l = math.sqrt((tar_cent[1,1]-pred_cent_1[1,1])**2 + (tar_cent[1,0]-pred_cent_1[1,0])**2)
        mm_r = pixel_to_mm(filename,dist_r)
        mm_l = pixel_to_mm(filename,dist_l)
    else:
        mm_r = -1
        mm_l = -1

    if model2 is not None:
        fem_pred_2, pred_cent_2 = prep_predictions(model2,extra2,filename,contl,contr,side,roi)
        for contour in fem_pred_2:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=colors[1])  
        if roi != 0 and pred_cent_2 is not None:
            plt.scatter(pred_cent_2[1], pred_cent_2[0], s=size, marker='.', c=colors[1])
                                           
    if model3 is not None:
        fem_pred_3, pred_cent_3 = prep_predictions(model3,extra3,filename,contl,contr,side,roi)
        for contour in fem_pred_3:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=colors[2])   
        if roi != 0 and pred_cent_3 is not None:
            plt.scatter(pred_cent_3[1], pred_cent_3[0], s=size, marker='.', c=colors[2])
          
    if model4 is not None:
        fem_pred_4, pred_cent_4 = prep_predictions(model4,extra4,filename,contl,contr,side,roi)
        for contour in fem_pred_4:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=colors[3])    
        if roi != 0 and pred_cent_4 is not None:
            plt.scatter(pred_cent_4[1], pred_cent_4[0], s=size, marker='.', c=colors[3])
    
    # Save the superimposed image 
    plt.savefig(os.path.join(superimposed_dir,filename+".png"),dpi=120*5)
    plt.show()
    plt.close()
    if roi == 0:
        return mm_r, mm_l


def prep_predictions(model,extra,filename,contl,contr,side,roi):

    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    
    predicted_full_mask = os.path.join(root,"Results",model,"Predicted"+extra+" FemHead Masks Smooth",
                                       filename+'.png')
    
    pred_contl, pred_contr = get_contours(predicted_full_mask)
    
    if pred_contl is not None:
        fem_pred = [pred_contr, pred_contl]

        pred_centr, pred_centl = femhead_centre(predicted_full_mask)
        pred = [pred_centr, pred_centl]
        preds = np.asarray(pred).astype(float)
        pred_cent = preds.reshape((-1,2))

        if roi != 0:
            if side == "l":
                fem_pred = [resize_roi(pred_contl,contl,left=True)]
                pred_cent = [resize_roi(pred_cent,contl,left=True)]
                pred_cent = pred_cent[0][1]
            else:
                fem_pred = [resize_roi(pred_contr,contr)]
                pred_cent = [resize_roi(pred_cent,contr)]
                pred_cent = pred_cent[0][0]
    else:
        fem_pred = []
        pred_cent = None

    return fem_pred, pred_cent
    
    
def pixel_to_mm(filename,dist):      
    '''
    Convert a pixel value to mm for a specific image.
    '''
    filename = filename.split("_")[0]
    root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_dir = os.path.join(root,"Dataset") 
    pixel_mm = pd.read_csv(os.path.join(data_dir,"pixel_to_mm.csv"))
    df = pixel_mm.loc[pixel_mm["name"]==str(filename)+".png"]
    if df.size != 0:
        df = np.asarray(df)[0]
        mm = dist*df[8]/(df[4]-df[2])
    else:
        mm = -1
    return mm


def roi_postprocess(k=10, model_name="UNet_ROI"):
    '''
    Takes the predicted ROI masks, extracts the ROI, and creates corresponding
    femoral head ROI masks and landmark ROI heatmaps to suit the prediction.
    '''
    
    for index in range(1,k+1):
        # Create directories to store results 
        current_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        save_mask_dir = os.path.join("Dataset","Fold "+str(index),"Predicted ROI FemHead Masks")
        save_img_dir = os.path.join("Dataset","Fold "+str(index),"Predicted ROI")
        mask_dir = os.path.join("Results",model_name,"Predicted ROI Masks")
        fem_dir = os.path.join(current_dir,"Dataset","Fold "+str(index),"FemHead Masks")

        if not os.path.exists(os.path.join(current_dir,save_mask_dir)): 
            os.makedirs(os.path.join(current_dir,save_mask_dir))
        if not os.path.exists(os.path.join(current_dir,save_img_dir)): 
            os.makedirs(os.path.join(current_dir,save_img_dir))

        # Get the filenames of the ROI predicted masks
        filenames = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
                         for file in glob(os.path.join(fem_dir,"*.png"))]

        # Go through each file
        for filename in filenames:
            if filename != "33914546":
                test_extract_ROI(current_dir,filename,mask_dir=mask_dir,img_dir=os.path.join("Dataset","Images"),
                        save_dir=save_img_dir)

                extract_ROI(current_dir,filename,mask_dir=mask_dir,img_dir=fem_dir,save_dir=save_mask_dir)
                
                
def test_roi_postprocess(model_name="Final_UNet_ROI"):
    '''
    Takes the predicted ROI masks, extracts the ROI, and creates corresponding
    femoral head ROI masks and landmark ROI heatmaps to suit the prediction.
    '''
    
    # Create directories to store results 
    current_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    save_mask_dir = os.path.join("Dataset","FINAL TEST","Predicted ROI FemHead Masks")
    save_img_dir = os.path.join("Dataset","FINAL TEST","Predicted ROI")
    mask_dir = os.path.join("Results",model_name,"Predicted ROI Masks")
    fem_dir = os.path.join(current_dir,"Dataset","FINAL TEST","FemHead Masks")

    if not os.path.exists(os.path.join(current_dir,save_mask_dir)): 
        os.makedirs(os.path.join(current_dir,save_mask_dir))
    if not os.path.exists(os.path.join(current_dir,save_img_dir)): 
        os.makedirs(os.path.join(current_dir,save_img_dir))

    # Get the filenames of the ROI predicted masks
    filenames = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
                     for file in glob(os.path.join(fem_dir,"*.png"))]

    # Go through each file
    for filename in filenames:
        if filename != "33914546" and filename != "26645401":
            final_extract_ROI(current_dir,filename,mask_dir=mask_dir,img_dir=os.path.join("Dataset","FINAL TEST","Images"),
                    save_dir=save_img_dir)

            final_extract_ROI(current_dir,filename,mask_dir=mask_dir,img_dir=fem_dir,save_dir=save_mask_dir)

                
def lm_preds_postprocess(k=10, model_name="UNet_LM",pred_dir="Predicted",dim=128,extra=" "):
    '''
    Takes the predicted ROI masks, extracts the ROI, and creates corresponding
    femoral head ROI masks and landmark ROI heatmaps to suit the prediction.
    '''
    
    for index in range(1,k+1):
        # Create directories to store results 
        current_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        data_dir = os.path.join("Dataset","Fold "+str(index))
        if dim==128:
            save_mask_dir = os.path.join(data_dir,pred_dir+" ROI LM Heatmaps")
            save_img_dir = os.path.join(data_dir,pred_dir+" ROI LMs")
            tl_dir = os.path.join(data_dir,pred_dir+" ROI LM Top-Lefts")
        elif dim==256:
            save_mask_dir = os.path.join(data_dir,pred_dir+" ROI LM Heatmaps Double")
            save_img_dir = os.path.join(data_dir,pred_dir+" ROI LMs Double")
            tl_dir = os.path.join(data_dir,pred_dir+" ROI LM Top-Lefts Double")
        mask_dir = os.path.join("Results",model_name,"Predicted" + extra + "CSVs")
        csv_dir = os.path.join(current_dir,data_dir,"CSVs")
        img_dir = os.path.join(data_dir,"Images")
        
        if not os.path.exists(os.path.join(current_dir,save_img_dir)): 
            os.makedirs(os.path.join(current_dir,save_img_dir))
        if not os.path.exists(os.path.join(current_dir,save_mask_dir)): 
            os.makedirs(os.path.join(current_dir,save_mask_dir))
        if not os.path.exists(os.path.join(current_dir,tl_dir)): 
            os.makedirs(os.path.join(current_dir,tl_dir))

        # Get the filenames of the ROI predicted masks
        filenames = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
                         for file in glob(os.path.join(csv_dir,"*.csv"))]

        # Go through each file
        for filename in filenames: 
            landmarks, image_size = prep_landmarks(filename,csv_dir)
        
            pred_lm = pd.read_csv(os.path.join(current_dir,mask_dir,filename+".csv"))
            pred_lm = np.asarray(pred_lm).astype(float)

            extract_ROI_from_pred_lm(current_dir,filename,pred_lm,image_size,img_dir=img_dir,
                                     save_dir=save_img_dir,tl_dir=tl_dir,dim=dim)
            
            pb_create_hm(current_dir,filename,pred_lm,image_size,save_dir=save_mask_dir,
                     tl_dir=tl_dir,dim=dim,size=5)
            

def final_lm_preds_postprocess(model_name="UNet_LM",pred_dir="Predicted",dim=128,extra=" "):
    '''
    Takes the predicted ROI masks, extracts the ROI, and creates corresponding
    femoral head ROI masks and landmark ROI heatmaps to suit the prediction.
    '''
    
    # Create directories to store results 
    current_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_dir = os.path.join("Dataset","FINAL TEST")
    if dim==128:
        save_mask_dir = os.path.join(data_dir,pred_dir+" ROI LM Heatmaps")
        save_img_dir = os.path.join(data_dir,pred_dir+" ROI LMs")
        tl_dir = os.path.join(data_dir,pred_dir+" ROI LM Top-Lefts")
    elif dim==256:
        save_mask_dir = os.path.join(data_dir,pred_dir+" ROI LM Heatmaps Double")
        save_img_dir = os.path.join(data_dir,pred_dir+" ROI LMs Double")
        tl_dir = os.path.join(data_dir,pred_dir+" ROI LM Top-Lefts Double")
    mask_dir = os.path.join("Results",model_name,"Predicted" + extra + "CSVs")
    csv_dir = os.path.join(current_dir,data_dir,"CSVs")
    img_dir = os.path.join(data_dir,"Images")

    if not os.path.exists(os.path.join(current_dir,save_img_dir)): 
        os.makedirs(os.path.join(current_dir,save_img_dir))
    if not os.path.exists(os.path.join(current_dir,save_mask_dir)): 
        os.makedirs(os.path.join(current_dir,save_mask_dir))
    if not os.path.exists(os.path.join(current_dir,tl_dir)): 
        os.makedirs(os.path.join(current_dir,tl_dir))

    # Get the filenames of the ROI predicted masks
    filenames = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
                     for file in glob(os.path.join(csv_dir,"*.csv"))]

    # Go through each file
    for filename in filenames: 
        landmarks, image_size = prep_landmarks(filename,csv_dir)

        pred_lm = pd.read_csv(os.path.join(current_dir,mask_dir,filename+".csv"))
        pred_lm = np.asarray(pred_lm).astype(float)

        extract_ROI_from_pred_lm(current_dir,filename,pred_lm,image_size,img_dir=img_dir,
                                 save_dir=save_img_dir,tl_dir=tl_dir,dim=dim)

        pb_create_hm(current_dir,filename,pred_lm,image_size,save_dir=save_mask_dir,
                 tl_dir=tl_dir,dim=dim,size=5)

        
def final_lm_preds_postprocess(model_name="UNet_LM",dim=128,extra=" "):
    '''
    Takes the predicted ROI masks, extracts the ROI, and creates corresponding
    femoral head ROI masks and landmark ROI heatmaps to suit the prediction.
    '''
    
    # Create directories to store results 
    current_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_dir = os.path.join("Dataset","FINAL TEST")
    if dim==128:
        save_mask_dir = os.path.join(data_dir,"ROI LM Heatmaps")
        save_img_dir = os.path.join(data_dir,"ROI LMs")
        tl_dir = os.path.join(data_dir,"ROI LM Top-Lefts")
    elif dim==256:
        save_mask_dir = os.path.join(data_dir,"ROI LM Heatmaps Double")
        save_img_dir = os.path.join(data_dir,"ROI LMs Double")
        tl_dir = os.path.join(data_dir,"ROI LM Top-Lefts Double")
    mask_dir = os.path.join(data_dir,"CSVs")
    csv_dir = os.path.join(current_dir,data_dir,"CSVs")
    img_dir = os.path.join(data_dir,"Images")

    if not os.path.exists(os.path.join(current_dir,save_img_dir)): 
        os.makedirs(os.path.join(current_dir,save_img_dir))
    if not os.path.exists(os.path.join(current_dir,save_mask_dir)): 
        os.makedirs(os.path.join(current_dir,save_mask_dir))
    if not os.path.exists(os.path.join(current_dir,tl_dir)): 
        os.makedirs(os.path.join(current_dir,tl_dir))

    # Get the filenames of the ROI predicted masks
    filenames = [os.path.normpath(file).split(os.path.sep)[-1].split('.')[0]
                     for file in glob(os.path.join(csv_dir,"*.csv"))]

    # Go through each file
    for filename in filenames: 
        landmarks, image_size = prep_landmarks(filename,csv_dir)
        
        extract_ROI_from_lm_aug2(data_dir,filename,landmarks,image_size,dim=128)

        pb_create_hm(data_dir,filename,landmarks,image_size,save_dir="ROI LM Heatmaps AUG2",
                     tl_dir="ROI LM Top-Lefts AUG2",dim=128,size=5)


def extract_curve_from_heatmap(heatmap, threshold=0.5):
    """
    Extract curve by computing the center of mass for each column that has signal.
    Returns list of (x, y) coordinates.
    """
    height, width = heatmap.shape
    points = []

    for col in range(width):
        column = heatmap[:, col]
        if column.max() > threshold:
            # Create a binary image for center_of_mass
            binary_column = column / column.sum() if column.sum() > 0 else column
            y, x = center_of_mass(binary_column.reshape(-1, 1))  # reshaped for 2D
            points.append((col, y))  # (x, y)

    return np.array(points)
    
        
def hausdorff_distance(A, B):
    """
    Symmetric Hausdorff distance between two point sets A and B.
    """
    d1 = directed_hausdorff(A, B)[0]
    d2 = directed_hausdorff(B, A)[0]
    return max(d1, d2)