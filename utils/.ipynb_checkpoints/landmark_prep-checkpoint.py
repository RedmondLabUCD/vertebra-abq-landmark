#import libraries 
import numpy as np
import pandas as pd 
import os
import itertools
import math


def resize_lm(points,old_dim,new_dim):
    '''
    Rescale landmarks to suit a change in the corresponding image size.
    '''

    landmarks = np.array(points)

    print(new_dim)
    print(old_dim)
    
    landmarks[:,0] = landmarks[:,0] * new_dim[0]/old_dim[0]
    landmarks[:,1] = landmarks[:,1] * new_dim[1]/old_dim[1]
    
    return landmarks 


def prep_landmarks(filename,csv_path):
    '''
    Extracts the landmark coordinates and image dimensions from landmark CSV file.
    Replaces any missing points with 'nan'.
    Input: image name, csv folder 
    Output: array of landmark coordinates, array of image dimensions
    
    '''

    data = pd.read_csv(os.path.join(csv_path,filename+'.csv'), header=None)
    image_size = np.asarray(data.iloc[0,4:6])

    # Get all the landmarks (insert NaN where a landmark does not exist)
    landmarks = pd.DataFrame(columns=[1,2])
    row = 0

    for num in range(1,23):
        if row >= len(data):
            landmarks = pd.concat([landmarks, pd.DataFrame.from_records([{ 1: np.nan, 2: np.nan}])])
        elif str(num) in str(data.loc[row,0]):
            landmarks = pd.concat([landmarks, pd.DataFrame.from_records([{ 1: data.iloc[row,1], 2: data.iloc[row,2]}])])
            row+=1
        else:
            # print(csv + ' is missing a key point!')
            landmarks = pd.concat([landmarks, pd.DataFrame.from_records([{ 1: np.nan, 2: np.nan}])])

    landmarks = landmarks.reset_index(drop=True)
    landmarks = np.asarray(landmarks).astype(float)

    return landmarks, image_size
