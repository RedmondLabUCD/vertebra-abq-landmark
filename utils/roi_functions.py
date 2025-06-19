#import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
import itertools
import math
from PIL import Image, ImageDraw
import cv2 as cv
import random

from utils.feature_extraction import femhead_centre, get_contours

        
def extract_ROI_from_lm(image_name,img,xy_pairs,image_size,dim=200,save_dir="/data/scratch/r094879/data/test",tl_dir="/data/scratch/r094879/data/test"):
    '''
    Given the image and the ROI mask, the ROI section of the image is extracted and saved.
    This same function is used to extract the ROI section of the femhead masks.
    '''

    lm = xy_pairs.copy()
    
    # Define array to collect the centre coordinates, landmark number, 
    # and off-centre number for each coordinate
    tl = np.zeros((13,3))
    tl[:,0] = range(1,14)
    
    # All points will have an ROI where they are at the centre
    width = dim/2
    
    for i in range(13):
        if not math.isnan(lm[i,1]) and not math.isnan(lm[i,0]):
            # Deal with exceptions where the point desired as the centre is too close to edge of image
            if int(lm[i,1]) < width:
                lm[i,1] = width
            if int(lm[i,1]) > image_size[0] - width:
                lm[i,1] = int(image_size[0]) - width
            if int(lm[i,0]) < width:
                lm[i,0] = width

            # Define and save cropped ROI
            cropped_img_r = img[int(int(lm[i,1])-width):int(int(lm[i,1])+width),
                                int(int(lm[i,0])-width):int(int(lm[i,0])+width)]
            # cv.imwrite(os.path.join(save_dir,image_name+"_"+str(i) +".png"),cropped_img_r)

            # img = cv.imread(os.path.join(save_dir,image_name+"_"+str(i) +".png"), 0)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_clahe = clahe.apply(cropped_img_r)
            cv.imwrite(os.path.join(save_dir,image_name+"_"+str(i) +".png"),img_clahe)
    
            # Collect the top-left coordinate of the ROI 
            tl[i] = [int(i),int(int(lm[i,0])-width),int(int(lm[i,1])-width)]
                
    pd.DataFrame(tl).to_csv(os.path.join(tl_dir,image_name+".csv"),index=False)


def resize_roi_lm(landmarks, contr, contl):
    '''
    Resize landmarks to fit axis of ROI (including the flipped left).
    '''
    
     # Alter the CSV values to match ROI
    landmarks[2:8,0] = landmarks[2:8,0] - np.ones((1,6))*np.min(contr[:,0])
    landmarks[2:8,1] = landmarks[2:8,1] - np.ones((1,6))*np.min(contr[:,1])
    landmarks[13:19,0] = -landmarks[13:19,0] + np.ones((1,6))*np.max(contl[:,0])
    landmarks[13:19,1] = landmarks[13:19,1] - np.ones((1,6))*np.min(contl[:,1])

    lm_r = landmarks[2:8,:] 
    lm_l = landmarks[13:19,:]
    
    return lm_r, lm_l


def resize_roi(lm, cont, left=False):
    '''
    Resize landmarks to fit axis of ROI (including the flipped left).
    '''
    
     # Alter the CSV values to match ROI
    if not left:
        lm[:,0] = lm[:,0] - np.ones((1,len(lm)))*np.min(cont[:,0])
    else:
        lm[:,0] = -lm[:,0] + np.ones((1,len(lm)))*np.max(cont[:,0])

    lm[:,1] = lm[:,1] - np.ones((1,len(lm)))*np.min(cont[:,1])
    
    return lm


def reverse_resize_roi_lm(lm, cont, left=False):
    '''
    Reverse of the above function: resize_roi_lm
    '''
    
     # Alter the CSV values to match ROI
    if not left:
        lm[:,0] = lm[:,0] + np.ones((1,len(lm)))*np.min(cont[:,0])
    else:
        lm[:,0] = -lm[:,0] + np.ones((1,len(lm)))*np.max(cont[:,0])
        
    lm[:,1] = lm[:,1] + np.ones((1,len(lm)))*np.min(cont[:,1])

    return lm


def roi_contour_dims(cont):
    '''
    Get the rectangular dimensions of a contour. Used for ROI mask contours.
    '''
                           
    x_dist = np.max(cont[:,0]) - np.min(cont[:,0]) + 1
    y_dist = np.max(cont[:,1]) - np.min(cont[:,1]) + 1
                           
    return [x_dist,y_dist]