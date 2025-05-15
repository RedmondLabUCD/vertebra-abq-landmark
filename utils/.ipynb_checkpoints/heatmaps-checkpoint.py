#import libraries 
import numpy as np
import pandas as pd 
import os
import math

from utils.roi_functions import resize_roi_lm, roi_contour_dims
from utils.feature_extraction import get_contours
from utils.landmark_prep import resize_lm
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter


def create_hm(landmarks,old_dim,new_dim,size=3):
    '''
    Resize landmarks to desired image dimensions and create heatmaps.
    '''
    lm = landmarks.copy()
    
    # Rescale landmark coordinates
    lm = resize_lm(lm,old_dim,new_dim)

    # Create heatmap
    hm = generate_hm(lm,new_dim,s=size)
                           
    return hm
    

def resize_coords(points, tl_dir, filename, vertebra):

    df = pd.read_csv(os.path.join(tl_dir,filename+".csv"))
    x_tl = df.iloc[vertebra,1]
    y_tl = df.iloc[vertebra,2]

    points = np.array(points)
    
    points[:,0] = points[:,0] - np.ones(len(points[:,0]))*x_tl
    points[:,1] = points[:,1] - np.ones(len(points[:,0]))*y_tl
    
    return points

    
def interpolate_curve(points, num_points=300):
    
    x, y = points[:, 0], points[:, 1]
    tck, u = splprep([x, y], s=0)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    return np.stack([x_new, y_new], axis=1)


def create_roi_hm(filename,landmarks,y_avg,save_dir="ROI LM Heatmaps",
                     tl_dir="ROI LM Top-Lefts",vertebra=0):
    '''
    Creates local heatmaps for Plan B isolate landmark ROIs.
    
    '''
    lm = landmarks.copy()
    tl_dir = '//data/scratch/r094879/data/roi_tls'
    
    scale_points = resize_coords(lm,tl_dir,filename,vertebra)
    rp = resize_lm(scale_points,[y_avg,y_avg],[256,256])

    endplate_top_x = [rp[11,0],rp[72,0],rp[73,0],rp[74,0],rp[75,0],rp[76,0],rp[77,0],rp[78,0],rp[79,0],rp[80,0],
                     rp[81,0],rp[82,0],rp[83,0],rp[84,0],rp[85,0],rp[86,0],rp[0,0]]
    endplate_top_y = [rp[11,1],rp[72,1],rp[73,1],rp[74,1],rp[75,1],rp[76,1],rp[77,1],rp[78,1],rp[79,1],rp[80,1],
                     rp[81,1],rp[82,1],rp[83,1],rp[84,1],rp[85,1],rp[86,1],rp[0,1]]
    endplate_top = np.array(list(zip(endplate_top_x,endplate_top_y)))
    
    endplate_bottom_x = [rp[3,0],rp[37,0],rp[38,0],rp[39,0],rp[40,0],rp[41,0],rp[42,0],rp[43,0],rp[44,0],rp[45,0],
                     rp[46,0],rp[47,0],rp[48,0],rp[10,0]]
    endplate_bottom_y = [rp[3,1],rp[37,1],rp[38,1],rp[39,1],rp[40,1],rp[41,1],rp[42,1],rp[43,1],rp[44,1],rp[45,1],
                     rp[46,1],rp[47,1],rp[48,1],rp[10,1]]
    endplate_bottom = np.array(list(zip(endplate_bottom_x,endplate_bottom_y)))
    
    interp_points_top = interpolate_curve(endplate_top,num_points=500)
    interp_points_bottom = interpolate_curve(endplate_bottom,num_points=500)

    heatmap = np.zeros([256,256,2], dtype=np.float32)
    for x, y in interp_points_top.astype(int):
        if 0 <= x < 256 and 0 <= y < 256:
            heatmap[y, x, 0] = 1.0

    for x, y in interp_points_bottom.astype(int):
        if 0 <= x < 256 and 0 <= y < 256:
            heatmap[y, x, 1] = 1.0
    
    # Optional: Apply Gaussian blur to create smooth heatmap
    heatmap = gaussian_filter(heatmap, sigma=2)

    np.save(os.path.join(save_dir,filename+"_"+str(vertebra) +".npy"),heatmap)

    if not os.path.exists("//data/scratch/r094879/data/data_check/cumulative_sum_roi"):
        os.makedirs("//data/scratch/r094879/data/data_check/cumulative_sum_roi")
    
    hm = heatmap[:,:,0] + heatmap[:,:,1]
    plt.imshow(hm, cmap='gray')
    plt.title("Cumulative Sum of All Slices")
    plt.savefig(os.path.join("//data/scratch/r094879/data/data_check/cumulative_sum_roi",filename+"_"+str(vertebra)+".png"))
    plt.close()


def gaussian_k(x0,y0,sigma,height,width):
        x = np.arange(0,height,1,float) ## (width,)
        y = np.arange(0,width,1,float)[:,np.newaxis] ## (height,1)
        return np.exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2))


def generate_hm(landmarks,dim,s=3):
        ''' 
        Generate a full Heap Map for every landmarks in an array
        Args:
            height    : The height of Heat Map (the height of target output)
            width     : The width  of Heat Map (the width of target output)
            joints    : [(x1,y1),(x2,y2)...] containing landmarks
            maxlenght : Lenght of the Bounding Box
            
        '''
        dim = int(dim)
        Nlandmarks = len(landmarks)
        hm = np.zeros((dim,dim,Nlandmarks), dtype = np.float32)
        for i in range(Nlandmarks):
            if not np.isnan(landmarks[i]).any():
                hm[:,:,i] = gaussian_k(landmarks[i][0],landmarks[i][1],s,dim,dim)
            else:
                hm[:,:,i] = np.zeros((dim,dim), dtype = np.float32)
        return hm
    
