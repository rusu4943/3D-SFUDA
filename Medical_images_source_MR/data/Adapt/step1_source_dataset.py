import h5py
import torch
import random
import numpy as np
from scipy import ndimage




    
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

        
        data_vol = data_vol * 2 - 1 # min-max normalization

        return data_vol, label_vol

    def __len__(self):
        return self.num_samples
    
def get_dataloader( train_dir, val_dir, num_classes, batch_size, domain = 'source'):
    dataloader = {}
    
    splits = [ 'train', 'test' ]
    
    for split in splits:
        
        if split == 'train':
            data_dir = train_dir
            dataset = abdominal_dataset( data_dir, split, num_classes, domain = domain)
            loader = torch.utils.data.DataLoader( dataset=dataset, batch_size = batch_size, shuffle=False, num_workers=4, )
            
        elif split == 'test':
            data_dir = val_dir
            dataset = abdominal_dataset( data_dir, split, num_classes, domain = domain)
            loader = torch.utils.data.DataLoader( dataset=dataset, batch_size = batch_size, shuffle=False, num_workers=0, )  
            
        else:
            print('error')
        
        
        dataloader[ split ] = loader
        
    return dataloader