#import libraries 
import numpy as np
import pandas as pd
import time
import os
import itertools
import torch
import torch.nn as nn
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from PIL import Image
from utils.landmark_prep import prep_landmarks
from utils.process_predictions import pixel_to_mm, extract_curve_from_heatmap, hausdorff_distance
import matplotlib.pyplot as plt
from pydicom import dcmread
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import directed_hausdorff
    

class mse_metric(nn.Module):
    def __init__(self):
        super(mse_metric, self).__init__()
    
    def forward(self,target,prediction,filename,params,square=True):
        prediction = prediction.cpu().detach().numpy()
        lm_pred = np.zeros((params.num_classes,2))
        root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        root2 = os.path.join(root,"Dataset")
        
        # Get most likely landmark locations based on heatmap predictions
        for i in range(params.num_classes):
            lm_preds = np.unravel_index(prediction[0,i,:,:].argmax(),
                                           (params.input_size,params.input_size))
            lm_preds = np.asarray(lm_preds).astype(float)
            lm_pred[i,0] = lm_preds[1]
            lm_pred[i,1] = lm_preds[0]
            
        # Use input image to resize predictions
        img = Image.open(os.path.join(root2,params.image_dir,filename+".png"))
        img_size = img.size
        img_size = np.asarray(img_size).astype(float)

        lm_pred[:,0] = lm_pred[:,0] * img_size[0]/float(params.input_size)
        lm_pred[:,1] = lm_pred[:,1] * img_size[1]/float(params.input_size)

        if params.num_classes > 6:
            csv_dir = os.path.join(root2,"CSVs")
            lm_targets, __ = prep_landmarks(filename,csv_dir)
        else: 
            csv_dir = os.path.join(root2,"ROI CSVs")
            lm_targets = pd.read_csv(os.path.join(csv_dir,filename+'.csv'))
            lm_targets = np.asarray(lm_targets).astype(float)
        lm_targets = lm_targets.reshape((-1,2))
        lm_targets = np.nan_to_num(lm_targets)
        
        # Convert from pixels to mm
        lm_targets[:,0] = pixel_to_mm(filename,lm_targets[:,0])
        lm_targets[:,1] = pixel_to_mm(filename,lm_targets[:,1])
        lm_pred[:,0] = pixel_to_mm(filename,lm_pred[:,0])
        lm_pred[:,1] = pixel_to_mm(filename,lm_pred[:,1])

        mse = mean_squared_error(lm_targets, lm_pred, squared=square)

        return mse


class pb_mse_metric_back(nn.Module):
    def __init__(self):
        super(pb_mse_metric_back, self).__init__()
    
    def forward(self,target,prediction,filename,params,subdir,AUG,square=True):
        prediction = prediction.cpu().detach().numpy()
        lm_pred = np.zeros((params.num_classes-1,2))
        root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        root2 = os.path.join(root,"Dataset",subdir)
        
        # Get most likely landmark locations based on heatmap predictions
        for i in range(params.num_classes-1):
            lm_preds = np.unravel_index(prediction[0,i,:,:].argmax(),
                                           (params.input_size,params.input_size))
            lm_preds = np.asarray(lm_preds).astype(float)
            lm_pred[i,0] = lm_preds[1]
            lm_pred[i,1] = lm_preds[0]
            
        # Use input image to resize predictions
        if AUG:
            image_dir = params.image_dir + " AUG"
            target_dir = "CSVs AUG"
        else:
            image_dir = params.image_dir
            target_dir = "CSVs"
        img = Image.open(os.path.join(root2,image_dir,filename+".png"))
        img_size = img.size
        img_size = np.asarray(img_size).astype(float)
        
        lm_pred[:,0] = lm_pred[:,0] * img_size[0]/float(params.input_size)
        lm_pred[:,1] = lm_pred[:,1] * img_size[1]/float(params.input_size)

        # Get targets
        csv_dir = os.path.join(root2,target_dir)
        lm_targets, __ = prep_landmarks(filename,csv_dir)
        lm_targets = lm_targets.reshape((-1,2))
        lm_targets = np.nan_to_num(lm_targets)
        
        # Convert from pixels to mm
        lm_targets[:,0] = pixel_to_mm(filename,lm_targets[:,0])
        lm_targets[:,1] = pixel_to_mm(filename,lm_targets[:,1])
        lm_pred[:,0] = pixel_to_mm(filename,lm_pred[:,0])
        lm_pred[:,1] = pixel_to_mm(filename,lm_pred[:,1])
        
        mse = mean_squared_error(lm_targets, lm_pred, squared=square)

        return mse


class curve_compare_metric(nn.Module):
    def __init__(self):
        super(curve_compare_metric, self).__init__()
    
    def forward(self,target,prediction,filename,params):
        prediction = prediction.cpu().detach().numpy()
        ground_truth = target.cpu().detach().numpy()

        # Extract curve from heatmap
        pc_1 = extract_curve_from_heatmap(prediction[0,0,:,:])
        pc_2 = extract_curve_from_heatmap(prediction[0,1,:,:])
        gt_1 = extract_curve_from_heatmap(ground_truth[0,0,:,:])
        gt_2 = extract_curve_from_heatmap(ground_truth[0,1,:,:])
    
        # Compute Hausdorff distance
        hd1 = hausdorff_distance(pc_1, gt_1)
        hd2 = hausdorff_distance(pc_2, gt_2)
        hd = (hd1+hd2)/2
        print(f"Hausdorff Distance: {hd:.2f}")
    
        # if not os.path.exists('//data/scratch/r094879/data/data_check/output_heatmap_curve/'):
        #     os.makedirs('//data/scratch/r094879/data/data_check/output_heatmap_curve/')
            
        # # Visualization
        # plt.imshow(heatmap, cmap='hot', origin='upper')
        # if pc_1.size > 0:
        #     plt.plot(pc_1[:, 0], pc_1[:, 1], 'b-', label='Predicted Curve')
        # if pc_2.size > 0:
        #     plt.plot(pc_2[:, 0], pc_2[:, 1], 'b-', label='Predicted Curve')
        # plt.plot(gt_1[:, 0], gt_1[:, 1], 'g--', label='Ground Truth')
        # plt.plot(gt_2[:, 0], gt_2[:, 1], 'g--', label='Ground Truth')
        # plt.legend()
        # plt.title(f"Hausdorff Distance: {hd:.2f}")
        # plt.gca().invert_yaxis()
        # plt.savefig('//data/scratch/r094879/data/data_check/output_heatmap_curve/'+str(filename)+'.png')

    
class pb_mse_metric(nn.Module):
    def __init__(self):
        super(pb_mse_metric, self).__init__()
    
    def forward(self,target,prediction,filename,params):
        prediction = prediction.cpu().detach().numpy()

        
        lm_pred = np.zeros((params.num_classes,2))
        root = '//data/scratch/r094879/data/'

        cumulative_sum = np.zeros(prediction.shape[2:])
        # Get most likely landmark locations based on heatmap predictions
        for i in range(params.num_classes):
            lm_preds = np.unravel_index(prediction[0,i,:,:].argmax(),
                                           (params.input_size,params.input_size))
            lm_preds = np.asarray(lm_preds).astype(float)
            lm_pred[i,0] = lm_preds[1]
            lm_pred[i,1] = lm_preds[0]
            # cumulative_sum += prediction[0,i,:,:]
            # plt.imshow(prediction[0,i,:,:], cmap='gray')
            # plt.axis('off')
            # plt.savefig(os.path.join("//data/scratch/r094879/data/data_check_long",filename+'_'+str(i)+'.png'))
            # plt.close()     

        # plt.imshow(cumulative_sum, cmap='gray')
        # plt.title("Cumulative Sum of All Slices")
        # plt.axis('off')
        # plt.savefig(os.path.join("//data/scratch/r094879/data/data_check_long",filename+'.png'))
        # plt.close()
    
        # Use input image to resize predictions
        image_dir = params.image_dir
        target_dir = "annotations/"
        img = dcmread(os.path.join(root,image_dir,filename+".dcm"))
        img_size = img.pixel_array.shape
        img_size = np.asarray(img_size).astype(float)
        
        lm_pred[:,0] = lm_pred[:,0] * float(img_size[1])/float(params.input_size)
        lm_pred[:,1] = lm_pred[:,1] * float(img_size[0])/float(params.input_size)

        # Get targets

        csv_file = os.path.join(root,'annotations/annotations.csv')
        csv_df = pd.read_csv(csv_file)

        filtered_row = csv_df[csv_df['image'] == filename]

        x_values = np.array(filtered_row.iloc[:,3:29:2].values).reshape((-1,1))
        y_values = np.array(filtered_row.iloc[:,4:29:2].values).reshape((-1,1))

        # Combine x and y values and filter out NaN pairs
        xy_pairs = np.concatenate([x_values,y_values],axis=1)
        
        # print(lm_pred)
        # print(xy_pairs)

        lm_targets = xy_pairs.reshape((-1,2))
        lm_targets = np.nan_to_num(lm_targets)

        lm_tars = []
        lm_preds = []

        for i in range(len(lm_targets)):
            if int(lm_targets[i][0]) != 0:
                lm_tars.append(lm_targets[i])
                lm_preds.append(lm_pred[i])

        lm_targets = np.array(lm_tars).reshape((-1,2))
        lm_pred = np.array(lm_preds).reshape((-1,2))

        mse = mean_squared_error(lm_targets, lm_pred)

        # print(filename)
        # print(mse)

        return mse


class pb_mse_metric_test(nn.Module):
    def __init__(self):
        super(pb_mse_metric_test, self).__init__()
    
    def forward(self,target,prediction,filename,params,name):
        prediction = prediction.cpu().detach().numpy()
        lm_pred = np.zeros((params.num_classes,2))
        root = '//data/scratch/r094879/data/'

        cumulative_sum = np.zeros(prediction.shape[2:])
        max_val = np.zeros(13)

        # Get most likely landmark locations based on heatmap predictions
        for i in range(params.num_classes):
            lm_preds = np.unravel_index(prediction[0,i,:,:].argmax(),
                                           (params.input_size,params.input_size))
            lm_preds = np.asarray(lm_preds).astype(float)
            lm_pred[i,0] = lm_preds[1]
            lm_pred[i,1] = lm_preds[0]
            max_val[i] = prediction[0,i,int(lm_preds[0]),int(lm_preds[1])]
        #     cumulative_sum += prediction[0,i,:,:]
        #     plt.imshow(prediction[0,i,:,:], cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(os.path.join("//data/scratch/r094879/data/results",name,'heatmaps',filename+'_'+str(i)+'.png'))
        #     plt.close()     

        # plt.imshow(cumulative_sum, cmap='gray')
        # plt.title("Cumulative Sum of All Slices")
        # plt.axis('off')
        # plt.savefig(os.path.join("//data/scratch/r094879/data/results",name,'heatmaps',filename+'.png'))
        # plt.close()
    
        # Use input image to resize predictions
        image_dir = params.image_dir
        target_dir = "annotations/"
        img = dcmread(os.path.join(root,image_dir,filename+".dcm"))    
        img_size = img.pixel_array.shape
        img_size = np.asarray(img_size).astype(float)
        
        lm_pred[:,0] = lm_pred[:,0] * float(img_size[1])/float(params.input_size)
        lm_pred[:,1] = lm_pred[:,1] * float(img_size[0])/float(params.input_size)

        # Get targets

        csv_file = os.path.join(root,'annotations/annotations.csv')
        csv_df = pd.read_csv(csv_file)

        filtered_row = csv_df[csv_df['image'] == filename]

        x_values = np.array(filtered_row.iloc[:,3:29:2].values).reshape((-1,1))
        y_values = np.array(filtered_row.iloc[:,4:29:2].values).reshape((-1,1))

        stats_df = pd.read_csv(os.path.join('//data/scratch/r094879/data/stats',name+'.csv'))

        stats_df.loc[stats_df["image"]==str(filename),"T4x"] = lm_pred[0,0]
        stats_df.loc[stats_df["image"]==str(filename),"T4y"] = lm_pred[0,1]
        stats_df.loc[stats_df["image"]==str(filename),"T4_val"] = max_val[0]
        stats_df.loc[stats_df["image"]==str(filename),"T5x"] = lm_pred[1,0]
        stats_df.loc[stats_df["image"]==str(filename),"T5y"] = lm_pred[1,1]
        stats_df.loc[stats_df["image"]==str(filename),"T5_val"] = max_val[1]
        stats_df.loc[stats_df["image"]==str(filename),"T6x"] = lm_pred[2,0]
        stats_df.loc[stats_df["image"]==str(filename),"T6y"] = lm_pred[2,1]
        stats_df.loc[stats_df["image"]==str(filename),"T6_val"] = max_val[2]
        stats_df.loc[stats_df["image"]==str(filename),"T7x"] = lm_pred[3,0]
        stats_df.loc[stats_df["image"]==str(filename),"T7y"] = lm_pred[3,1]
        stats_df.loc[stats_df["image"]==str(filename),"T7_val"] = max_val[3]
        stats_df.loc[stats_df["image"]==str(filename),"T8x"] = lm_pred[4,0]
        stats_df.loc[stats_df["image"]==str(filename),"T8y"] = lm_pred[4,1]
        stats_df.loc[stats_df["image"]==str(filename),"T8_val"] = max_val[4]
        stats_df.loc[stats_df["image"]==str(filename),"T9x"] = lm_pred[5,0]
        stats_df.loc[stats_df["image"]==str(filename),"T9y"] = lm_pred[5,1]
        stats_df.loc[stats_df["image"]==str(filename),"T9_val"] = max_val[5]
        stats_df.loc[stats_df["image"]==str(filename),"T10x"] = lm_pred[6,0]
        stats_df.loc[stats_df["image"]==str(filename),"T10y"] = lm_pred[6,1]
        stats_df.loc[stats_df["image"]==str(filename),"T10_val"] = max_val[6]
        stats_df.loc[stats_df["image"]==str(filename),"T11x"] = lm_pred[7,0]
        stats_df.loc[stats_df["image"]==str(filename),"T11y"] = lm_pred[7,1]
        stats_df.loc[stats_df["image"]==str(filename),"T11_val"] = max_val[7]
        stats_df.loc[stats_df["image"]==str(filename),"T12x"] = lm_pred[8,0]
        stats_df.loc[stats_df["image"]==str(filename),"T12y"] = lm_pred[8,1]
        stats_df.loc[stats_df["image"]==str(filename),"T12_val"] = max_val[8]
        stats_df.loc[stats_df["image"]==str(filename),"L1x"] = lm_pred[9,0]
        stats_df.loc[stats_df["image"]==str(filename),"L1y"] = lm_pred[9,1]
        stats_df.loc[stats_df["image"]==str(filename),"L1_val"] = max_val[9]
        stats_df.loc[stats_df["image"]==str(filename),"L2x"] = lm_pred[10,0]
        stats_df.loc[stats_df["image"]==str(filename),"L2y"] = lm_pred[10,1]
        stats_df.loc[stats_df["image"]==str(filename),"L2_val"] = max_val[10]
        stats_df.loc[stats_df["image"]==str(filename),"L3x"] = lm_pred[11,0]
        stats_df.loc[stats_df["image"]==str(filename),"L3y"] = lm_pred[11,1]
        stats_df.loc[stats_df["image"]==str(filename),"L3_val"] = max_val[11]
        stats_df.loc[stats_df["image"]==str(filename),"L4x"] = lm_pred[12,0]
        stats_df.loc[stats_df["image"]==str(filename),"L4y"] = lm_pred[12,1]
        stats_df.loc[stats_df["image"]==str(filename),"L4_val"] = max_val[12]

        # Combine x and y values and filter out NaN pairs
        xy_pairs = np.concatenate([x_values,y_values],axis=1)
        
        # print(lm_pred)
        # print(xy_pairs)

        lm_targets = xy_pairs.reshape((-1,2))
        lm_targets = np.nan_to_num(lm_targets)

        lm_tars = []
        lm_preds = []

        for i in range(len(lm_targets)):
            if int(lm_targets[i][0]) != 0:
                lm_tars.append(lm_targets[i])
                lm_preds.append(lm_pred[i])

        lm_targets = np.array(lm_tars).reshape((-1,2))
        lm_pred = np.array(lm_preds).reshape((-1,2))

        error_tar = []
        error_pred = []
        count = 0
        dist = (lm_targets[1,1]-lm_targets[0,1])/2
        print(dist)
        for i in range(len(lm_targets)):
            if abs(int(lm_preds[i][1])-int(lm_targets[i][1])) < dist:
                if abs(int(lm_preds[i][0])-int(lm_targets[i][0])) < dist:
                    count+=1
                    error_tar.append(lm_targets[i])
                    error_pred.append(lm_pred[i])
        
        if error_pred:            
            error_tar = np.array(error_tar).reshape((-1,2))
            error_pred = np.array(error_pred).reshape((-1,2))

            rmse = root_mean_squared_error(error_tar, error_pred)

            stats_df.loc[stats_df["image"]==str(filename),"RMSE"] = rmse   
        else:
            rmse = np.nan
            
                    
        id_acc = count/len(lm_targets)
        
        stats_df.loc[stats_df["image"]==str(filename),"correct"] = count
        stats_df.loc[stats_df["image"]==str(filename),"total"] = len(lm_targets)
        stats_df.loc[stats_df["image"]==str(filename),"id_acc"] = id_acc

        stats_df.to_csv(os.path.join('//data/scratch/r094879/data/stats',name+'.csv'),index=False)

        # print(filename)
        # print(mse)

        return rmse
    

class test_pb_mse_metric(nn.Module):
    def __init__(self):
        super(test_pb_mse_metric, self).__init__()
    
    def forward(self,prediction,filename,params,AUG,square=True):
        prediction = prediction.cpu().detach().numpy()
        lm_pred = np.zeros((params.num_classes,2))
        root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        root2 = os.path.join(root,"Dataset","FINAL TEST")
        
        # Get most likely landmark locations based on heatmap predictions
        for i in range(params.num_classes):
            lm_preds = np.unravel_index(prediction[0,i,:,:].argmax(),
                                           (params.input_size,params.input_size))
            lm_preds = np.asarray(lm_preds).astype(float)
            lm_pred[i,0] = lm_preds[1]
            lm_pred[i,1] = lm_preds[0]
            
        # Use input image to resize predictions
        if AUG:
            image_dir = params.image_dir + " AUG"
            target_dir = "CSVs AUG"
        else:
            image_dir = params.image_dir
            target_dir = "CSVs"
        img = Image.open(os.path.join(root2,image_dir,filename+".png"))
        img_size = img.size
        img_size = np.asarray(img_size).astype(float)
        
        lm_pred[:,0] = lm_pred[:,0] * img_size[0]/float(params.input_size)
        lm_pred[:,1] = lm_pred[:,1] * img_size[1]/float(params.input_size)

        # Get targets
        csv_dir = os.path.join(root2,target_dir)
        lm_targets, __ = prep_landmarks(filename,csv_dir)
        lm_targets = lm_targets.reshape((-1,2))
        lm_targets = np.nan_to_num(lm_targets)
        
        # Convert from pixels to mm
        lm_targets[:,0] = pixel_to_mm(filename,lm_targets[:,0])
        lm_targets[:,1] = pixel_to_mm(filename,lm_targets[:,1])
        lm_pred[:,0] = pixel_to_mm(filename,lm_pred[:,0])
        lm_pred[:,1] = pixel_to_mm(filename,lm_pred[:,1])

        mse = mean_squared_error(lm_targets, lm_pred, squared=square)

        return mse
   
        
class pb_roi_mse_metric(nn.Module):
    def __init__(self):
        super(pb_roi_mse_metric, self).__init__()
    
    def forward(self,target,prediction,filename,params,square=True):
        prediction = prediction.cpu().detach().numpy()
        lm_pred = [0,0]
        target = target.cpu().detach().numpy()
        lm_tar = [0,0]
        
        lm_num = int(filename.split("_")[-2])
        if lm_num >= 12:
            lm_num = lm_num-11
        # Get most likely landmark locations based on heatmap predictions
        lm_preds = np.unravel_index(prediction[0,lm_num-1,:,:].argmax(),
                                       (params.input_size,params.input_size))
        lm_preds = np.asarray(lm_preds).astype(float)
        lm_pred[0] = lm_preds[1]
        lm_pred[1] = lm_preds[0]
        
        # Get most likely landmark locations based on heatmap predictions
        lm_tars = np.unravel_index(target[0,lm_num-1,:,:].argmax(),
                                       (params.input_size,params.input_size))
        lm_tars = np.asarray(lm_tars).astype(float)
        lm_tar[0] = lm_tars[1]
        lm_tar[1] = lm_tars[0]
        
        # Convert from pixels to mm
        lm_tar[0] = pixel_to_mm(filename,lm_tar[0])
        lm_tar[1] = pixel_to_mm(filename,lm_tar[1])
        lm_pred[0] = pixel_to_mm(filename,lm_pred[0])
        lm_pred[1] = pixel_to_mm(filename,lm_pred[1])

        mse = mean_squared_error(lm_tar, lm_pred, squared=square)
#         mse = math.dist(lm_tar,lm_pred)

        return mse


class test_pb_roi_mse_metric(nn.Module):
    def __init__(self):
        super(test_pb_roi_mse_metric, self).__init__()
    
    def forward(self,prediction,filename,params,AUG,prediction_dir,square=True):
        root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        prediction = prediction.cpu().detach().numpy()
        lm_pred = [0,0]
        
        lm_num = int(filename.split("_")[-2])
        if lm_num >= 12:
            lm_num = lm_num-11
        # Get most likely landmark locations based on heatmap predictions
        lm_preds = np.unravel_index(prediction[0,lm_num-1,:,:].argmax(),
                                       (params.input_size,params.input_size))
        lm_preds = np.asarray(lm_preds).astype(float)
        
        image_num = filename.split("_")[0]
        lm_num = int(filename.split("_")[-2])
    
        targets, __ = prep_landmarks(image_num,os.path.join(root,"Dataset","FINAL TEST","CSVs"))
        targets = targets.reshape((-1,2))
        targets = np.nan_to_num(targets)
        
        end3 = prediction_dir.split('\\')[-1].split(" ")[0:2]
        ex = ""
        end = end3[0]
        if "_AUG2" in end:
            end2 = end[0:-5]
            ex = " AUG2"
        else:
            end2 = end
        end2 = end2 + " "

        if end3[1] == "Pred":
            centre_dir = os.path.join(root,"Dataset","FINAL TEST",end2+"ROI LM Top-Lefts")
            end = end3[0] + " " + end3[1]
        else:
            centre_dir = os.path.join(root,"Dataset","FINAL TEST","ROI LM Top-Lefts" + ex)
            end = end3[0]
        
        centres = pd.read_csv(os.path.join(centre_dir,image_num+'.csv'))
        centres = np.asarray(centres)
        
        if lm_num >= 12:
            lm_pred[0] = -lm_preds[1] + (centres[lm_num-3,1]+params.input_size-1)
            lm_pred[1] = lm_preds[0] + centres[lm_num-3,2]
        else:
            lm_pred[0] = lm_preds[1] + centres[lm_num-1,1]
            lm_pred[1] = lm_preds[0] + centres[lm_num-1,2]

        # Convert from pixels to mm
        targets[lm_num-1,0] = pixel_to_mm(filename,targets[lm_num-1,0])
        targets[lm_num-1,1] = pixel_to_mm(filename,targets[lm_num-1,1])
        lm_pred[0] = pixel_to_mm(filename,lm_pred[0])
        lm_pred[1] = pixel_to_mm(filename,lm_pred[1])

        mse = mean_squared_error(targets[lm_num-1,:], lm_pred, squared=square)
#         mse = math.dist(lm_tar,lm_pred)

        return mse


class custom_weighted_loss(nn.Module):
    def __init__(self):
        super(custom_weighted_loss, self).__init__()
    
    def forward(self,prediction,target):
        weight_map = target.detach().clone()
        weight_map = (weight_map>0.5).float()
        diff = (prediction - target)**2
        weighted = diff*(weight_map*9+1)
        loss = torch.mean(weighted)
        return loss
    
    
class custom_weighted_loss2(nn.Module):
    def __init__(self):
        super(custom_weighted_loss2, self).__init__()
    
    def forward(self,prediction,target):
        weight_map = target.detach().clone()
        weight_map = (weight_map*10).int()
        diff = (prediction - target)**2
        weighted = diff*(weight_map+1)
        loss = torch.mean(weighted)
        return loss


class custom_weighted_loss_2(nn.Module):
    # Custom weight loss that:
    #     1) ignore the layers that are immediately before/after layers that contain landmarks
    #     2) increase the error weighting in layers that contain no landmark at all 
    def __init__(self):
        super(custom_weighted_loss_2, self).__init__()
    
    def forward(self,prediction,target):
        weight_map = target.detach().clone()
        weight_map = (weight_map>0.5).float()
        weight_map = weight_map*9 + 1
        for k in range(weight_map.shape[0]):
            first = False
            last = False
            next = True
            for i in range(13):
                if 10 in weight_map[k,i,:,:]:
                    if not first:
                        first = True 
                        last = False
                        next = False
                        if i != 0:
                            weight_map[k,i-1,:,:] = weight_map[k,i-1,:,:]*0
                else:    
                    if first and not last:
                        first = False
                        last = True
                        next = True
                        weight_map[k,i,:,:] = weight_map[k,i,:,:]*0
                    elif next:
                        weight_map[k,i,:,:] = weight_map[k,i,:,:]*5
                        
        diff = (prediction - target)**2
        weighted = diff*weight_map
        loss = torch.mean(weighted)
        return loss


class custom_weighted_loss_3(nn.Module):
    # Custom weight loss that:
    #     1) ignore the layers that are immediately before/after layers that contain landmarks
    #     2) increase the error weighting in layers that contain no landmark at all 
    def __init__(self):
        super(custom_weighted_loss_3, self).__init__()
    
    def forward(self,prediction,target):
        weight_map = target.detach().clone()
        weight_map = (weight_map>0.5).float()

        weight_map_equal = np.zeros((1,1,weight_map.shape[2],weight_map.shape[2]))
        
        for k in range(weight_map.shape[0]):
            weight_map_equal = torch.sum(weight_map[k,:,:,:],0)
            # plt.imshow(weight_map_equal*255, cmap='gray')
            # plt.title("Weighted")
            # plt.axis('off')
            # plt.savefig(os.path.join("//data/scratch/r094879/data/data_check_weighted",str(k)+'.png'))
            # plt.close()
            
            first = False
            last = False
            next = True
            for i in range(13):
                if 1 in weight_map[k,i,:,:]:
                    weight_map[k,i,:,:] = weight_map_equal*9+1 
                    if not first:
                        first = True 
                        last = False
                        next = False
                        if i != 0:
                            weight_map[k,i-1,:,:] = weight_map_equal*9+1 
                else:    
                    if first and not last:
                        first = False
                        last = True
                        weight_map[k,i,:,:] = weight_map_equal*9+1 
                    elif last and not next:
                        next = True
                        weight_map[k,i,:,:] = (weight_map_equal+1)*5
                    elif next:
                        weight_map[k,i,:,:] = (weight_map_equal+1)*5
                        
        diff = (prediction - target)**2
        weighted = diff*weight_map
        loss = torch.mean(weighted)
        return loss