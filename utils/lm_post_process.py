from __future__ import print_function
import os
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
from utils.process_predictions import pixel_to_mm
import cv2 as cv
import math
from sklearn.metrics import mean_squared_error
from utils.landmark_prep import prep_landmarks
import matplotlib.pyplot as plt


def lm_post_process(name,extra,root,data_dir,params,prediction_dir,predictions,filename,subdir,metric_avg,count=[0,0,0,0]):
    '''
    Runs the dataset through the LM model and post-processes the resulting predictions - 
    including extracting landmark coordinates from the heatmaps, resizing them, and 
    superimposing them on the input image. The results are saved.
    '''
    predictions = predictions.cpu().detach().numpy()    
#     print("Prediction " + str(filename))
#     fig = plt.figure(figsize=(18,10))
#     for j in range(23):
#         ax = fig.add_subplot(4,6,j+1)
#         print(str(j) + "max:" + str(predictions[0,j,:,:].max()) + "std: "+ str(np.std(predictions[0,j,:,:])))
#         ax.imshow(predictions[0,j,:,:])
#     plt.show()
    
    targets, __ = prep_landmarks(filename,os.path.join(root,"Dataset","Fold "+subdir,"CSVs"))
    targets = targets.reshape((-1,2))
    targets = np.nan_to_num(targets)
    
    dice = name+extra

    np.save(os.path.join(prediction_dir,filename+".npy"),predictions)
    end = prediction_dir.split('\\')[-1].split(" ")[0]
    prediction_dir_2 = os.path.join(root,"Results",name,end+" CSVs")
    if not os.path.exists(prediction_dir_2): os.makedirs(prediction_dir_2)
        
    # Set up location to store landmark coordinates
    lm_preds = np.zeros((params.num_classes,2))

    # Get most likely landmark locations based on heatmap predictions
    for i in range(params.num_classes):
#         if predictions[0,i,:,:].max() >= 0.35 and np.std(predictions[0,i,:,:]) >= 0.03:
            lm_location = np.unravel_index(predictions[0,i,:,:].argmax(),
                                           (params.input_size,params.input_size))
            lm_location = np.asarray(lm_location).astype(float)
            lm_preds[i,0] = lm_location[1]
            lm_preds[i,1] = lm_location[0]
            
    # Use input image to resize predictions
    img = Image.open(os.path.join(root,"Dataset","Fold "+subdir,"Images",filename+".png"))
    img_size = img.size
    img_size = np.asarray(img_size).astype(float)

    lm_preds[:,0] = lm_preds[:,0] * img_size[0]/float(params.input_size)
    lm_preds[:,1] = lm_preds[:,1] * img_size[1]/float(params.input_size)

    # Save the prediction
    pd.DataFrame(lm_preds).to_csv(os.path.join(prediction_dir_2,filename+".csv"),index = False)
    
#     # Convert pixel values to mm for both targets and preds
#     targets[:,0] = pixel_to_mm(filename,targets[:,0])
#     targets[:,1] = pixel_to_mm(filename,targets[:,1])
#     lm_preds[:,0] = pixel_to_mm(filename,lm_preds[:,0])
#     lm_preds[:,1] = pixel_to_mm(filename,lm_preds[:,1])
        
    # Create/open csv file to store results    
    if not os.path.exists(os.path.join(data_dir,"lm_stats.csv")):
        image_names = os.listdir(os.path.join(root,"Dataset","Images"))
        names = []
        for image_name in image_names:
            image_name = image_name.split(".")[0]
            names.append(image_name)
        lm_stats = pd.DataFrame({"Image":names})
        lm_stats["Fold"] = ""
        col = str(dice)+"_avg"
        lm_stats[str(col)] = ""
        for i in range(1,23):
            col = str(dice) + "_LM" + str(i)
            lm_stats[str(col)] = ""
        lm_stats.to_csv(os.path.join(data_dir,'lm_stats.csv'),index=False)
        lm_stats = pd.read_csv(os.path.join(data_dir,"lm_stats.csv"))
    else:
        lm_stats = pd.read_csv(os.path.join(data_dir,"lm_stats.csv"))
        col = str(dice)+"_avg"
        if str(col) not in lm_stats.columns:
            lm_stats[str(col)] = ""
        for i in range(1,23):
            col = str(dice) + "_LM" + str(i)
            if str(col) not in lm_stats.columns:
                lm_stats[str(col)] = ""
        lm_stats.to_csv(os.path.join(data_dir,'lm_stats.csv'),index=False)
        lm_stats = pd.read_csv(os.path.join(data_dir,"lm_stats.csv"))
    col = str(dice)+"_avg"    
    lm_stats.loc[lm_stats.Image==int(filename),[str(col),"Fold"]] = metric_avg, subdir  
    
    for i in range(0,22):
        if (0. in lm_preds[i,:].tolist()) and (0.0 in targets[i,:].tolist()):
            dist = 0
#             print("Prediction CORRECTLY missing: %s %s" %(filename,i+1))
            count[0] = count[0]+1
        elif 0. in lm_preds[i,:].tolist():
            dist = -1
#             print("Prediction missing: %s %s" %(filename,i+1))
            count[1] = count[1]+1
        elif 0.0 in targets[i,:].tolist():
            dist = -2
#             print("Target missing: %s %s" %(filename,i+1))
            count[2] = count[2]+1
        else:
            dist = abs(math.dist(targets[i,:], lm_preds[i,:]))
            dist = pixel_to_mm(filename,dist)
            count[3] = count[3]+1
        col = str(dice) + "_LM" + str(i+1)
        lm_stats.loc[lm_stats.Image==int(filename),str(col)] = dist
        
    lm_stats.to_csv(os.path.join(data_dir,'lm_stats.csv'),index=False)
    
    return count


def final_lm_post_process(name,extra,root,data_dir,params,prediction_dir,predictions,filename,metric_avg,count=[0,0,0,0]):
    '''
    Runs the dataset through the LM model and post-processes the resulting predictions - 
    including extracting landmark coordinates from the heatmaps, resizing them, and 
    superimposing them on the input image. The results are saved.
    '''
    predictions = predictions.cpu().detach().numpy()    
    
    targets, __ = prep_landmarks(filename,os.path.join(root,"Dataset","FINAL TEST","CSVs"))
    targets = targets.reshape((-1,2))
    targets = np.nan_to_num(targets)
    
    dice = name+extra

    np.save(os.path.join(prediction_dir,filename+".npy"),predictions)
    end = prediction_dir.split('\\')[-1].split(" ")[0]
    prediction_dir_2 = os.path.join(root,"Results",name,end+" CSVs")
    if not os.path.exists(prediction_dir_2): os.makedirs(prediction_dir_2)
        
    # Set up location to store landmark coordinates
    lm_preds = np.zeros((params.num_classes,2))

    # Get most likely landmark locations based on heatmap predictions
    for i in range(params.num_classes):
#         if predictions[0,i,:,:].max() >= 0.35 and np.std(predictions[0,i,:,:]) >= 0.03:
            lm_location = np.unravel_index(predictions[0,i,:,:].argmax(),
                                           (params.input_size,params.input_size))
            lm_location = np.asarray(lm_location).astype(float)
            lm_preds[i,0] = lm_location[1]
            lm_preds[i,1] = lm_location[0]
            
    # Use input image to resize predictions
    img = Image.open(os.path.join(root,"Dataset","FINAL TEST","Images",filename+".png"))
    img_size = img.size
    img_size = np.asarray(img_size).astype(float)

    lm_preds[:,0] = lm_preds[:,0] * img_size[0]/float(params.input_size)
    lm_preds[:,1] = lm_preds[:,1] * img_size[1]/float(params.input_size)

    # Save the prediction
    pd.DataFrame(lm_preds).to_csv(os.path.join(prediction_dir_2,filename+".csv"),index = False)
        
    # Create/open csv file to store results    
    if not os.path.exists(os.path.join(data_dir,"final_lm_stats.csv")):
        image_names = os.listdir(os.path.join(root,"Dataset","FINAL TEST","Images"))
        names = []
        for image_name in image_names:
            image_name = image_name.split(".")[0]
            names.append(image_name)
        lm_stats = pd.DataFrame({"Image":names})
        col = str(dice)+"_avg"
        lm_stats[str(col)] = ""
        for i in range(1,23):
            col = str(dice) + "_LM" + str(i)
            lm_stats[str(col)] = ""
        lm_stats.to_csv(os.path.join(data_dir,'final_lm_stats.csv'),index=False)
        lm_stats = pd.read_csv(os.path.join(data_dir,"final_lm_stats.csv"))
    else:
        lm_stats = pd.read_csv(os.path.join(data_dir,"final_lm_stats.csv"))
        col = str(dice)+"_avg"
        if str(col) not in lm_stats.columns:
            lm_stats[str(col)] = ""
        for i in range(1,23):
            col = str(dice) + "_LM" + str(i)
            if str(col) not in lm_stats.columns:
                lm_stats[str(col)] = ""
        lm_stats.to_csv(os.path.join(data_dir,'final_lm_stats.csv'),index=False)
        lm_stats = pd.read_csv(os.path.join(data_dir,"final_lm_stats.csv"))
    col = str(dice)+"_avg"    
    lm_stats.loc[lm_stats.Image==int(filename),str(col)] = metric_avg 
    
    for i in range(0,22):
        if (0. in lm_preds[i,:].tolist()) and (0.0 in targets[i,:].tolist()):
            dist = 0
#             print("Prediction CORRECTLY missing: %s %s" %(filename,i+1))
            count[0] = count[0]+1
        elif 0. in lm_preds[i,:].tolist():
            dist = -1
#             print("Prediction missing: %s %s" %(filename,i+1))
            count[1] = count[1]+1
        elif 0.0 in targets[i,:].tolist():
            dist = -2
#             print("Target missing: %s %s" %(filename,i+1))
            count[2] = count[2]+1
        else:
            dist = abs(math.dist(targets[i,:], lm_preds[i,:]))
            dist = pixel_to_mm(filename,dist)
            count[3] = count[3]+1
        col = str(dice) + "_LM" + str(i+1)
        lm_stats.loc[lm_stats.Image==int(filename),str(col)] = dist
        
    lm_stats.to_csv(os.path.join(data_dir,'final_lm_stats.csv'),index=False)
    
    return count
    
            
def roi_lm_post_process(name,extra,root,data_dir,params,prediction_dir,predictions,filename,subdir,metric_avg,csv_name):
    '''
    Runs the dataset through the LM model and post-processes the resulting predictions - 
    including extracting landmark coordinates from the heatmaps, resizing them, and 
    superimposing them on the input image. The results are saved.
    '''
    dice = name+extra
    predictions = predictions.cpu().detach().numpy()  
    np.save(os.path.join(prediction_dir,filename+".npy"),predictions)
    end3 = prediction_dir.split('\\')[-1].split(" ")[0:2]
    ex = ""
    end = end3[0]
    if "baseAUG" in dice.split("_"):
        end2 = end[0:-8]
    elif "_AUG2" in end:
        end2 = end[0:-5]
        ex = " AUG2"
    elif "_AUG" in end:
        end2 = end[0:-4]
        ex = " AUG"
    else:
        end2 = end
    end2 = end2 + " "
    
    if "Double" in name:
        ex = " Double" + ex
        
    if end3[1] == "Pred":
        centre_dir = os.path.join(root,"Dataset","Fold "+subdir,end2+"ROI LM Top-Lefts")
        end = end3[0] + " " + end3[1]
    else:
        centre_dir = os.path.join(root,"Dataset","Fold "+subdir,"ROI LM Top-Lefts" + ex)
        end = end3[0]
    prediction_dir_2 = os.path.join(root,"Results",name,end+" CSVs")
    if not os.path.exists(prediction_dir_2): os.makedirs(prediction_dir_2) 
    
    image_num = filename.split("_")[0]
    lm_num = int(filename.split("_")[-2])
    
    targets, __ = prep_landmarks(image_num,os.path.join(root,"Dataset","Fold "+subdir,"CSVs"))
    targets = targets.reshape((-1,2))
    targets = np.nan_to_num(targets)
    targets = np.delete(targets, 21, 0)
    targets = np.delete(targets, 20, 0)
    targets = np.delete(targets, 10, 0)
    targets = np.delete(targets, 9, 0)

    if csv_name == None:
        lms = np.zeros((18,2))
        csv_name = image_num
        centres = pd.read_csv(os.path.join(centre_dir,csv_name+'.csv'))
        centres = np.asarray(centres)
    elif image_num != csv_name:
        csv_name = image_num
        lms = np.zeros((18,2))
        centres = pd.read_csv(os.path.join(centre_dir,csv_name+'.csv'))
        centres = np.asarray(centres)
    else:
        lms = pd.read_csv(os.path.join(prediction_dir_2,csv_name+".csv"))
        lms = np.asarray(lms)
        centres = pd.read_csv(os.path.join(centre_dir,csv_name+'.csv'))
        centres = np.asarray(centres)

     # Adjust left landmark numbering to right landmark equivalent
    if lm_num >= 12:
        lm_num_adj = lm_num-11
    else:
        lm_num_adj = lm_num

    # Get most likely landmark locations based on heatmap predictions
    lm_preds = np.unravel_index(predictions[0,lm_num_adj-1,:,:].argmax(),
                                   (params.input_size,params.input_size))
    lm_preds = np.asarray(lm_preds).astype(float)

    # Adjust from ROI to full-scale image values (including unflipping left)
    if lm_num >= 12:
        lms[lm_num-3,0] = -lm_preds[1] + (centres[lm_num-3,1]+params.input_size-1)
        lms[lm_num-3,1] = lm_preds[0] + centres[lm_num-3,2]
    else:
        lms[lm_num-1,0] = lm_preds[1] + centres[lm_num-1,1]
        lms[lm_num-1,1] = lm_preds[0] + centres[lm_num-1,2]
      
    pd.DataFrame(lms).to_csv(os.path.join(prediction_dir_2,csv_name+".csv"),index=False)

     # Create/open csv file to store results    
    if not os.path.exists(os.path.join(data_dir,"lm_stats.csv")):
        image_names = os.listdir(os.path.join(root,"Dataset","Images"))
        names = []
        for image_name in image_names:
            image_name = image_name.split(".")[0]
            names.append(image_name)
        lm_stats = pd.DataFrame({"Image":names})
        lm_stats["Fold"] = ""
        col = str(dice)+"_avg"
        lm_stats[str(col)] = ""
        for i in range(1,19):
            col = str(dice) + "_LM" + str(i)
            lm_stats[str(col)] = ""
        lm_stats.to_csv(os.path.join(data_dir,'lm_stats.csv'),index=False)
        lm_stats = pd.read_csv(os.path.join(data_dir,"lm_stats.csv"))
    else:
        lm_stats = pd.read_csv(os.path.join(data_dir,"lm_stats.csv"))
        col = str(dice)+"_avg"
        if str(col) not in lm_stats.columns:
            lm_stats[str(col)] = ""
        for i in range(1,19):
            col = str(dice) + "_LM" + str(i)
            if str(col) not in lm_stats.columns:
                lm_stats[str(col)] = ""
        lm_stats.to_csv(os.path.join(data_dir,'lm_stats.csv'),index=False)
        lm_stats = pd.read_csv(os.path.join(data_dir,"lm_stats.csv"))
    col = str(dice)+"_avg"    
    lm_stats.loc[lm_stats.Image==int(csv_name),[str(col),"Fold"]] = metric_avg, subdir  
    
    for i in range(0,18):
        mse = mean_squared_error(targets[i,:], lms[i,:], squared=True)
        col = str(dice) + "_LM" + str(i+1)
        lm_stats.loc[lm_stats.Image==int(csv_name),str(col)] = mse
        
    lm_stats.to_csv(os.path.join(data_dir,'lm_stats.csv'),index=False)
    
    return csv_name
   
           
def final_roi_lm_post_process(name,extra,root,data_dir,params,prediction_dir,
                              predictions,filename,metric_avg,csv_name):
    '''
    Runs the dataset through the LM model and post-processes the resulting predictions - 
    including extracting landmark coordinates from the heatmaps, resizing them, and 
    superimposing them on the input image. The results are saved.
    '''
    dice = name+extra
    predictions = predictions.cpu().detach().numpy()  
    np.save(os.path.join(prediction_dir,filename+".npy"),predictions)
    end3 = prediction_dir.split('\\')[-1].split(" ")[0:2]
    ex = ""
    end = end3[0]
    if "baseAUG" in dice.split("_"):
        end2 = end[0:-8]
    elif "_AUG2" in end:
        end2 = end[0:-5]
        ex = " AUG2"
    elif "_AUG" in end:
        end2 = end[0:-4]
        ex = " AUG"
    else:
        end2 = end
    end2 = end2 + " "
    
    if "Double" in name:
        ex = " Double" + ex
        
    if end3[1] == "Pred":
        centre_dir = os.path.join(root,"Dataset","FINAL TEST",end2+"ROI LM Top-Lefts")
        end = end3[0] + " " + end3[1]
    else:
        centre_dir = os.path.join(root,"Dataset","FINAL TEST","ROI LM Top-Lefts" + ex)
        end = end3[0]
    prediction_dir_2 = os.path.join(root,"Results",name,end+" CSVs")
    if not os.path.exists(prediction_dir_2): os.makedirs(prediction_dir_2) 
    
    image_num = filename.split("_")[0]
    lm_num = int(filename.split("_")[-2])
    
    targets, __ = prep_landmarks(image_num,os.path.join(root,"Dataset","FINAL TEST","CSVs"))
    targets = targets.reshape((-1,2))
    targets = np.nan_to_num(targets)
    targets = np.delete(targets, 21, 0)
    targets = np.delete(targets, 20, 0)
    targets = np.delete(targets, 10, 0)
    targets = np.delete(targets, 9, 0)

    if csv_name == None:
        lms = np.zeros((18,2))
        csv_name = image_num
        centres = pd.read_csv(os.path.join(centre_dir,csv_name+'.csv'))
        centres = np.asarray(centres)
    elif image_num != csv_name:
        csv_name = image_num
        lms = np.zeros((18,2))
        centres = pd.read_csv(os.path.join(centre_dir,csv_name+'.csv'))
        centres = np.asarray(centres)
    else:
        lms = pd.read_csv(os.path.join(prediction_dir_2,csv_name+".csv"))
        lms = np.asarray(lms)
        centres = pd.read_csv(os.path.join(centre_dir,csv_name+'.csv'))
        centres = np.asarray(centres)

     # Adjust left landmark numbering to right landmark equivalent
    if lm_num >= 12:
        lm_num_adj = lm_num-11
    else:
        lm_num_adj = lm_num

    # Get most likely landmark locations based on heatmap predictions
    lm_preds = np.unravel_index(predictions[0,lm_num_adj-1,:,:].argmax(),
                                   (params.input_size,params.input_size))
    lm_preds = np.asarray(lm_preds).astype(float)

    # Adjust from ROI to full-scale image values (including unflipping left)
    if lm_num >= 12:
        lms[lm_num-3,0] = -lm_preds[1] + (centres[lm_num-3,1]+params.input_size-1)
        lms[lm_num-3,1] = lm_preds[0] + centres[lm_num-3,2]
    else:
        lms[lm_num-1,0] = lm_preds[1] + centres[lm_num-1,1]
        lms[lm_num-1,1] = lm_preds[0] + centres[lm_num-1,2]
      
    pd.DataFrame(lms).to_csv(os.path.join(prediction_dir_2,csv_name+".csv"),index=False)

     # Create/open csv file to store results    
    if not os.path.exists(os.path.join(data_dir,"final_lm_stats.csv")):
        image_names = os.listdir(os.path.join(root,"Dataset","FINAL TEST","Images"))
        names = []
        for image_name in image_names:
            image_name = image_name.split(".")[0]
            names.append(image_name)
        lm_stats = pd.DataFrame({"Image":names})
        col = str(dice)+"_avg"
        lm_stats[str(col)] = ""
        for i in range(1,19):
            col = str(dice) + "_LM" + str(i)
            lm_stats[str(col)] = ""
        lm_stats.to_csv(os.path.join(data_dir,'final_lm_stats.csv'),index=False)
        lm_stats = pd.read_csv(os.path.join(data_dir,"final_lm_stats.csv"))
    else:
        lm_stats = pd.read_csv(os.path.join(data_dir,"final_lm_stats.csv"))
        col = str(dice)+"_avg"
        if str(col) not in lm_stats.columns:
            lm_stats[str(col)] = ""
        for i in range(1,19):
            col = str(dice) + "_LM" + str(i)
            if str(col) not in lm_stats.columns:
                lm_stats[str(col)] = ""
        lm_stats.to_csv(os.path.join(data_dir,'final_lm_stats.csv'),index=False)
        lm_stats = pd.read_csv(os.path.join(data_dir,"final_lm_stats.csv"))
    col = str(dice)+"_avg"    
    lm_stats.loc[lm_stats.Image==int(csv_name),str(col)] = metric_avg 
    
    for i in range(0,18):
        mse = mean_squared_error(targets[i,:], lms[i,:], squared=True)
        col = str(dice) + "_LM" + str(i+1)
        lm_stats.loc[lm_stats.Image==int(csv_name),str(col)] = mse
        
    lm_stats.to_csv(os.path.join(data_dir,'final_lm_stats.csv'),index=False)
    
    return csv_name