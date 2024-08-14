import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import sys
import h5py
import time
import pickle
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display as ipy_display, clear_output



sys.path.append('../')
import networks
dataset = __import__('dataset-step4')



source_mr_train_dir = "../../data/h5py/"
source_mr_test_dir = "../../data/h5py/"
target_ct_train_dir = "../../data/h5py/"
target_ct_test_dir = "../../data/h5py/"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device


def sliding_window(input_volume, window_size=(32, 32, 32), stride=(16, 16, 16)):
    
    z_max = input_volume.shape[0] - window_size[0] + 1
    x_max = input_volume.shape[1] - window_size[1] + 1
    y_max = input_volume.shape[2] - window_size[2] + 1

    windows = []

    for y in range(0, y_max, stride[2]):
        for x in range(0, x_max, stride[1]):
            for z in range(0, z_max, stride[0]):
                window = input_volume[z:z+window_size[0], x:x+window_size[1], y:y+window_size[2]]
                windows.append(window)

            # z_remaining
            z_remaining = input_volume.shape[0] - window_size[0] # z_remaining = 78 - 32 = 46
            window = input_volume[z_remaining:, x:x+window_size[1], y:y+window_size[2]]
            windows.append(window)
        
        # x_remaining
        x_remaining = input_volume.shape[1] - window_size[1] # z_remaining = 78 - 32 = 46
        for z in range(0, z_max, stride[0]):
            window = input_volume[z:z+window_size[0], x_remaining: , y:y+window_size[2]]
            windows.append(window)
            
        # x_remaining z_remaining
        z_remaining = input_volume.shape[0] - window_size[0] # z_remaining = 78 - 32 = 46
        window = input_volume[z_remaining:, x_remaining: , y:y+window_size[2]]
        windows.append(window)
    
    # y_remaining
    y_remaining = input_volume.shape[2] - window_size[2] # z_remaining = 78 - 32 = 46
    for x in range(0, x_max, stride[1]):
        for z in range(0, z_max, stride[0]):
            window = input_volume[z:z+window_size[0], x:x+window_size[1], y_remaining: ]
            windows.append(window)
            
        # y_remaining z_remaining
        z_remaining = input_volume.shape[0] - window_size[0] # z_remaining = 78 - 32 = 46
        window = input_volume[z_remaining:, x:x+window_size[1], y_remaining:]
        windows.append(window)

    # y_remaining x_remaining
    x_remaining = input_volume.shape[1] - window_size[1] # z_remaining = 78 - 32 = 46
    for z in range(0, z_max, stride[0]):
        window = input_volume[z:z+window_size[0], x_remaining: , y_remaining:]
        windows.append(window)

    # y_remaining x_remaining z_remaining
    z_remaining = input_volume.shape[0] - window_size[0] # z_remaining = 78 - 32 = 46
    window = input_volume[z_remaining:, x_remaining: , y_remaining:]
    windows.append(window)

    return windows


def combine_windows(window_outputs, input_volume_shape, window_size=(32, 32, 32), stride=(16, 16, 16)):
    num_classes = window_outputs[0].shape[1] # 5
    combined_prob = torch.zeros((num_classes,) + input_volume_shape).to(device)
    count_matrix = torch.zeros(input_volume_shape).to(device)

    z_max = input_volume_shape[0] - window_size[0] + 1
    x_max = input_volume_shape[1] - window_size[1] + 1
    y_max = input_volume_shape[2] - window_size[2] + 1

    idx = 0
    
    
    for y in range(0, y_max, stride[2]):
        for x in range(0, x_max, stride[1]):
            for z in range(0, z_max, stride[0]):
                output = window_outputs[idx].squeeze() # output.cpu().numpy().shape: (5, 32, 256, 256)
                combined_prob[:, z:z+window_size[0], x:x+window_size[1], y:y+window_size[2]] += output
                count_matrix[z:z+window_size[0], x:x+window_size[1], y:y+window_size[2]] += 1
                idx += 1
                

            # z_remaining
            z_remaining = input_volume_shape[0] - window_size[0] # z_remaining = 78 - 32 = 46
            output = window_outputs[idx].squeeze()
            combined_prob[:, z_remaining:, x:x+window_size[1], y:y+window_size[2]] += output
            count_matrix[z_remaining:, x:x+window_size[1], y:y+window_size[2]] += 1
            idx += 1
        
        # x_remaining
        x_remaining = input_volume_shape[1] - window_size[1] # z_remaining = 78 - 32 = 46
        for z in range(0, z_max, stride[0]):
            output = window_outputs[idx].squeeze()
            combined_prob[:, z:z+window_size[0], x_remaining: , y:y+window_size[2]] += output
            count_matrix[z:z+window_size[0], x_remaining: , y:y+window_size[2]] += 1
            idx += 1
            
            
        # x_remaining z_remaining
        z_remaining = input_volume_shape[0] - window_size[0] # z_remaining = 78 - 32 = 46
        output = window_outputs[idx].squeeze()
        combined_prob[:, z_remaining:, x_remaining: , y:y+window_size[2]] += output
        count_matrix[z_remaining:, x_remaining: , y:y+window_size[2]] += 1
        idx += 1
        
    
    # y_remaining
    y_remaining = input_volume_shape[2] - window_size[2] # z_remaining = 78 - 32 = 46
    for x in range(0, x_max, stride[1]):
        for z in range(0, z_max, stride[0]):
            output = window_outputs[idx].squeeze()
            combined_prob[:, z:z+window_size[0], x:x+window_size[1], y_remaining: ] += output
            count_matrix[z:z+window_size[0], x:x+window_size[1], y_remaining: ] += 1
            idx += 1
            
            
        # y_remaining z_remaining
        z_remaining = input_volume_shape[0] - window_size[0] # z_remaining = 78 - 32 = 46
        output = window_outputs[idx].squeeze()
        combined_prob[:, z_remaining:, x:x+window_size[1], y_remaining:] += output
        count_matrix[z_remaining:, x:x+window_size[1], y_remaining:] += 1
        idx += 1
        

    # y_remaining x_remaining
    x_remaining = input_volume_shape[1] - window_size[1] # z_remaining = 78 - 32 = 46
    for z in range(0, z_max, stride[0]):
        output = window_outputs[idx].squeeze()
        combined_prob[:, z:z+window_size[0], x_remaining: , y_remaining:] += output
        count_matrix[z:z+window_size[0], x_remaining: , y_remaining:] += 1
        idx += 1

    # y_remaining x_remaining z_remaining
    z_remaining = input_volume_shape[0] - window_size[0] # z_remaining = 78 - 32 = 46
    output = window_outputs[idx].squeeze()
    combined_prob[:, z_remaining:, x_remaining: , y_remaining:] += output
    count_matrix[z_remaining:, x_remaining: , y_remaining:] += 1
    idx += 1
    
    
    # Normalize the class probabilities
    combined_prob /= count_matrix

    # Take the argmax of the accumulated probabilities
    combined_output = torch.argmax(combined_prob, dim=0)

    return combined_output


def create_dataloader():
    
    dataloader = dataset.get_dataloader( target_ct_train_dir,  target_ct_test_dir, 5, 1,  domain = 'target' )

    train_dataset = dataloader["train"].dataset
    test_dataset = dataloader["test"].dataset
    
    return train_dataset, test_dataset
    
def compute_miou_all_training_examples(Unet, classifier, label_ids):
    
    train_dataset, test_dataset = create_dataloader()
    
    test_output = []

    for img_idx in range(len(test_dataset)): # 0, 1, 2, 3

        data_vol, label_vol = test_dataset[img_idx] # data_vol: torch.Size([1, 60, 512, 512])
        data_vol = data_vol.to(device)
        label_vol = label_vol.to(device)

        data_vol = torch.squeeze(data_vol, 0) # data_vol:  torch.Size([60, 512, 512])
        windows = sliding_window(data_vol) # slice 3D image based on window size and stride



        window_outputs = []

        Unet.eval()
        classifier.eval() 
        with torch.no_grad():
            for window in windows:
                window = window.unsqueeze(0)  # Add a channel dimension: torch.Size([1, 32, 256, 256])
                window = torch.unsqueeze(window, 0)  # Add a batch dimension: torch.Size([1, 1, 32, 256, 256])

                # inference
                output = Unet(window)
                output = classifier(output) # torch.Size([1, 5, 32, 256, 256])

                # collect outputs
                window_outputs.append(output)  # len(window_outputs) = 27
                # window_outputs[0].cpu().numpy().shape： (1, 5, 32, 256, 256)

        combined_output = combine_windows(window_outputs, data_vol.size())
        test_output.append(combined_output)
        
        
    numpy_arrays = [tensor.cpu().numpy() for tensor in test_output]
    
    id_to_ignore = 0
    intersection = dict()
    total = dict()
    for label in label_ids:
        intersection[label] = total[label] = 0


    for img_idx in range(len(test_dataset)): # 0, 1, 2, 3

        _, y_true = test_dataset[img_idx] # data_vol: torch.Size([1, 60, 512, 512])

        y_hat = numpy_arrays[img_idx]
        y_true = y_true.cpu().numpy() 

        for label in label_ids:
            if label_ids[label] == id_to_ignore:
                continue

            curr_id = label_ids[label]

            idx_gt = y_true == curr_id
            idx_hat = y_hat == curr_id

            intersection[label] += 2 * np.sum(idx_gt & idx_hat)
            total[label] += np.sum(idx_gt) + np.sum(idx_hat)

        dice = []
        res = dict()
        for label in label_ids:
            if label_ids[label] == id_to_ignore:
                continue

            if total[label] != 0:
                res[label] = intersection[label] / total[label]
            else:
                print('total is zero')
                res[label] = np.float64(0)

            dice.append(res[label])
            
            
    return np.mean(dice)