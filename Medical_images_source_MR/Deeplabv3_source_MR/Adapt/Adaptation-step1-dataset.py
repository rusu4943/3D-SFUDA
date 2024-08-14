import h5py
import torch
import random
import numpy as np
from scipy import ndimage





def sliding_window(input_volume, label_volume, window_size=(32, 32, 32), stride=(32, 32, 32)):
    
    z_max = input_volume.shape[0] - window_size[0] + 1
    x_max = input_volume.shape[1] - window_size[1] + 1
    y_max = input_volume.shape[2] - window_size[2] + 1

    windows = []
    window_labels = []
    sample_counts = np.zeros_like(input_volume, dtype=float)

    for y in range(0, y_max, stride[2]):
        for x in range(0, x_max, stride[1]):
            for z in range(0, z_max, stride[0]):
                window = input_volume[z:z+window_size[0], x:x+window_size[1], y:y+window_size[2]]
                window_label = label_volume[z:z+window_size[0], x:x+window_size[1], y:y+window_size[2]]
                sample_counts[z:z+window_size[0], x:x+window_size[1], y:y+window_size[2]] += 1
                windows.append(window)
                window_labels.append(window_label)

            # z_remaining
            z_remaining = input_volume.shape[0] - window_size[0] # z_remaining = 78 - 32 = 46
            window = input_volume[z_remaining:, x:x+window_size[1], y:y+window_size[2]]
            window_label = label_volume[z_remaining:, x:x+window_size[1], y:y+window_size[2]]
            sample_counts[z_remaining:, x:x+window_size[1], y:y+window_size[2]] += 1
            windows.append(window)
            window_labels.append(window_label)
        
        # x_remaining
        x_remaining = input_volume.shape[1] - window_size[1] # z_remaining = 78 - 32 = 46
        for z in range(0, z_max, stride[0]):
            window = input_volume[z:z+window_size[0], x_remaining: , y:y+window_size[2]]
            window_label = label_volume[z:z+window_size[0], x_remaining: , y:y+window_size[2]]
            sample_counts[z:z+window_size[0], x_remaining: , y:y+window_size[2]] += 1
            windows.append(window)
            window_labels.append(window_label)
            
        # x_remaining z_remaining
        z_remaining = input_volume.shape[0] - window_size[0] # z_remaining = 78 - 32 = 46
        window = input_volume[z_remaining:, x_remaining: , y:y+window_size[2]]
        window_label = label_volume[z_remaining:, x_remaining: , y:y+window_size[2]]
        sample_counts[z_remaining:, x_remaining: , y:y+window_size[2]] += 1
        windows.append(window)
        window_labels.append(window_label)
    
    # y_remaining
    y_remaining = input_volume.shape[2] - window_size[2] # z_remaining = 78 - 32 = 46
    for x in range(0, x_max, stride[1]):
        for z in range(0, z_max, stride[0]):
            window = input_volume[z:z+window_size[0], x:x+window_size[1], y_remaining: ]
            window_label = label_volume[z:z+window_size[0], x:x+window_size[1], y_remaining: ]
            sample_counts[z:z+window_size[0], x:x+window_size[1], y_remaining: ] += 1
            windows.append(window)
            window_labels.append(window_label)
            
        # y_remaining z_remaining
        z_remaining = input_volume.shape[0] - window_size[0] # z_remaining = 78 - 32 = 46
        window = input_volume[z_remaining:, x:x+window_size[1], y_remaining:]
        window_label = label_volume[z_remaining:, x:x+window_size[1], y_remaining:]
        sample_counts[z_remaining:, x:x+window_size[1], y_remaining:] += 1
        windows.append(window)
        window_labels.append(window_label)

    # y_remaining x_remaining
    x_remaining = input_volume.shape[1] - window_size[1] # z_remaining = 78 - 32 = 46
    for z in range(0, z_max, stride[0]):
        window = input_volume[z:z+window_size[0], x_remaining: , y_remaining:]
        window_label = label_volume[z:z+window_size[0], x_remaining: , y_remaining:]
        sample_counts[z:z+window_size[0], x_remaining: , y_remaining:] += 1
        windows.append(window)
        window_labels.append(window_label)

    # y_remaining x_remaining z_remaining
    z_remaining = input_volume.shape[0] - window_size[0] # z_remaining = 78 - 32 = 46
    window = input_volume[z_remaining:, x_remaining: , y_remaining:]
    window_label = label_volume[z_remaining:, x_remaining: , y_remaining:]
    sample_counts[z_remaining:, x_remaining: , y_remaining:] += 1
    windows.append(window)
    window_labels.append(window_label)

    return windows, window_labels, sample_counts
    
class abdominal_dataset( torch.utils.data.Dataset ):

    def __init__( self, data_dir, split, num_classes, domain = 'source', transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.domain = domain
        
        with h5py.File(f"{self.data_dir}{self.domain}_{self.split}.h5", 'r') as h5f:
            self.num_samples = len([key for key in h5f.keys() if key.startswith('X_')])

    def __getitem__(self, idx):
        
        with h5py.File(f"{self.data_dir}{self.domain}_{self.split}.h5", 'r') as h5f:
            X = h5f['X_' + str(idx)][()]
            Y = h5f['Y_' + str(idx)][()]
        
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == Y.shape[1]
        assert X.shape[2] == Y.shape[2]

        
        data_vol = X
        label_vol = Y
        
        # right now is [0, 1], we need to perform min-max normalization
        #print(np.min(data_vol), np.max(data_vol), np.mean(data_vol), np.var(data_vol))
        
        data_vol, label_vol, samples_cts = sliding_window(data_vol, label_vol)
        
        data_vol = np.stack(data_vol) # (140, 30, 30, 30)
        label_vol = np.stack(label_vol) # (140, 30, 30, 30)
        
        data_vol = data_vol.transpose((0, 3, 1, 2))
        data_vol = np.expand_dims( data_vol, 1)
        data_vol = data_vol * 2 - 1 # min-max normalization
        
        label_vol = label_vol.transpose((0, 3, 1, 2))
        
        samples_cts = samples_cts.transpose((2, 0, 1))
        
        assert data_vol.shape[2:] == label_vol.shape[1:]
        
        return data_vol, label_vol, samples_cts

    def __len__(self):
        return self.num_samples
    
    
    
def get_dataloader( train_dir, val_dir, num_classes, batch_size, domain = 'source'):
    dataloader = {}
    
    splits = [ 'train', 'test' ]
    
    for split in splits:
        
        if split == 'train':
            data_dir = train_dir
            dataset = abdominal_dataset( data_dir, split, num_classes, domain = domain, transform = None)
            loader = torch.utils.data.DataLoader( dataset=dataset, batch_size = batch_size, shuffle=True, num_workers=4, )
            
        elif split == 'test':
            data_dir = val_dir
            dataset = abdominal_dataset( data_dir, split, num_classes, domain = domain, transform = None)
            loader = torch.utils.data.DataLoader( dataset=dataset, batch_size = batch_size, shuffle=False, num_workers=0, )  
            
        else:
            print('error')
        
        
        dataloader[ split ] = loader
        
    return dataloader