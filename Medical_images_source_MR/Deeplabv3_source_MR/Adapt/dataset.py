import h5py
import torch
import random
import numpy as np
from scipy import ndimage




class Custom3DTransform:
    
    def __init__(self):
        self.crop_size = (32, 32, 32)
       

    def random_crop(self, volume, label):
        h, w, z = volume.shape
        ch, cw, cz = self.crop_size

        sx, sy, sz = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1), np.random.randint(0, z - cz + 1)
        
        return volume[sx:sx + ch, sy:sy + cw, sz:sz + cz], label[sx:sx + ch, sy:sy + cw, sz:sz + cz], (sx, sy, sz)

 
    def __call__(self, volume, label):
        volume, label, (sy, sx, sz) = self.random_crop(volume, label)
        
        
        return volume, label, (sy, sx, sz)

    
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
        
        print('data_vol.shape: ', X.shape)
        
        data_vol = X
        label_vol = Y
        
        # right now is [0, 1], we need to perform min-max normalization
        #print(np.min(data_vol), np.max(data_vol), np.mean(data_vol), np.var(data_vol))

        
        if self.transform:
            
            data_vol, label_vol, (sx, sy, sz) = self.transform(data_vol, label_vol)
        
        data_vol = data_vol * 2 - 1 # min-max normalization
        data_vol = np.expand_dims( data_vol.transpose((2, 0, 1)), 0)
        data_vol = torch.from_numpy(data_vol).float()

        label_vol = label_vol.transpose((2, 0, 1))
        label_vol = torch.from_numpy(label_vol.copy()).long()

        if self.transform:
        
            return data_vol, label_vol, (sx, sy, sz), X.shape

        return data_vol, label_vol

    def __len__(self):
        return self.num_samples
    
def get_dataloader( train_dir, val_dir, num_classes, batch_size, domain = 'source'):
    dataloader = {}
    
    splits = [ 'train', 'test' ]
    
    for split in splits:
        
        if split == 'train':
            data_dir = train_dir
            transform = Custom3DTransform()
            dataset = abdominal_dataset( data_dir, split, num_classes, domain = domain, transform = transform)
            loader = torch.utils.data.DataLoader( dataset=dataset, batch_size = batch_size, shuffle=True, num_workers=4, )
            
        elif split == 'test':
            data_dir = val_dir
            dataset = abdominal_dataset( data_dir, split, num_classes, domain = domain, transform = None)
            loader = torch.utils.data.DataLoader( dataset=dataset, batch_size = batch_size, shuffle=False, num_workers=0, )  
            
        else:
            print('error')
        
        
        dataloader[ split ] = loader
        
    return dataloader