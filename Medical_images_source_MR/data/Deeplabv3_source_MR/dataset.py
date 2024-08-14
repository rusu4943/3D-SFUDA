import time
import h5py
import torch
import random
import numpy as np
from scipy import ndimage



class ElasticTransform:
    def __init__(self, alpha=1000, sigma=30, random_state=None):
        self.alpha = alpha
        self.sigma = sigma
        self.random_state = np.random.RandomState(random_state) # constructs a random number generator

    def __call__(self, volume):
        depth, height, width = volume.shape

        dx = self.random_state.rand(depth, height, width) * 2 - 1
        dy = self.random_state.rand(depth, height, width) * 2 - 1
        dz = self.random_state.rand(depth, height, width) * 2 - 1

        dx = ndimage.gaussian_filter(dx, self.sigma) * self.alpha
        dy = ndimage.gaussian_filter(dy, self.sigma) * self.alpha
        dz = ndimage.gaussian_filter(dz, self.sigma) * self.alpha

        z, y, x = np.meshgrid(np.arange(depth), np.arange(height), np.arange(width), indexing='ij')
        indices = np.reshape(z + dz, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        #  interpolation = 1 = order
        transformed_volume = ndimage.map_coordinates(volume, indices, order=1, mode='reflect').reshape(volume.shape)
        
        return transformed_volume
    
    
class CustomColorJetter:
    
    def __init__(self):
        pass
        
    def RandomSaturation(self, volume, saturation_limit=[0.9,1.1]):
        
        saturation = random.uniform(saturation_limit[0], saturation_limit[1])
        
        return np.clip(volume * saturation,0,1)

    def RandomBrightness(self, volume, intensity_limit=[0, 0.1]):
        
        brightness = random.uniform(intensity_limit[0], intensity_limit[1])
        
        return np.clip(volume + brightness,0,1)

    def RandomContrast(self, volume, contrast_limit=[0.9,1.1]):
        
        mean = np.mean(volume, axis=(0,1,2), keepdims=True)
        contrast = random.uniform(contrast_limit[0], contrast_limit[1])
        
        return np.clip(volume * contrast + mean * (1 - contrast),0,1)
        
    def __call__(self, volume):
        
        augs = [self.RandomSaturation, self.RandomBrightness, self.RandomContrast]
        random.shuffle(augs)

        for aug in augs:
            if random.random() > 0.5:
                volume = aug(volume)

        return volume
    
    
class Custom3DTransform:
    
    def __init__(self):
        self.crop_size = (32, 32, 32)
        self.elastic_transform = ElasticTransform(alpha=1000, sigma=30)
        self.ColorJetter = CustomColorJetter()

    def random_crop(self, volume, label):
        w, h, z = volume.shape
        cw, ch, cz = self.crop_size

        sx, sy, sz = np.random.randint(0, w - cw + 1), np.random.randint(0, h - ch + 1), np.random.randint(0, z - cz + 1)
        
        return volume[sx:sx + cw, sy:sy + ch, sz:sz + cz], label[sx:sx + cw, sy:sy + ch, sz:sz + cz]

    def random_rotation(self, volume, label, degrees=(-15, 15)):
        rotation_angle = np.random.uniform(degrees[0], degrees[1])
        
        return (ndimage.rotate(volume, rotation_angle, axes=(1, 2), reshape=False, order=1),
                ndimage.rotate(label, rotation_angle, axes=(1, 2), reshape=False, order=1))

    def random_translation(self, volume, label, translate=(0.1, 0.1)):
        height_shift = int(volume.shape[1] * np.random.uniform(-translate[0], translate[0]))
        width_shift = int(volume.shape[2] * np.random.uniform(-translate[1], translate[1]))
        
        return (ndimage.shift(volume, (0, height_shift, width_shift), order=1),
                ndimage.shift(label, (0, height_shift, width_shift), order=1))

    def random_scale(self, volume, label, scale_limit=[0.9, 1.1], interpolation=1):
        scale = random.uniform(scale_limit[0], scale_limit[1])
        height, width, depth = volume.shape
        new_height, new_width = int(height * scale), int(width * scale)

        return (ndimage.zoom(volume, (scale, scale, 1), order=interpolation),
                ndimage.zoom(label, (scale, scale, 1), order=interpolation))

    def random_flip(self, volume, label, axis=0):
        return np.flip(volume, axis), np.flip(label, axis)
    
    def resize_to_crop_size(self, volume, label):
        h, w, z = volume.shape
        oh, ow, oz = self.crop_size
        scale_factors = (oh / h, ow / w, oz / z)

        return (ndimage.zoom(volume, scale_factors, order=1),
                ndimage.zoom(label, scale_factors, order=1))

    def __call__(self, volume, label):
        volume, label = self.random_crop(volume, label)

        if random.random() > 0.5:
            volume = self.ColorJetter(volume)
            
        if random.random() > 0.5:
            volume, label = self.random_rotation(volume, label)
        
        if random.random() > 0.5:
            volume, label = self.random_scale(volume, label)
            
        volume, label = self.resize_to_crop_size(volume, label)
        
        if random.random() > 0.5:
            volume, label = self.random_translation(volume, label)
        
        if random.random() > 0.2:
            volume = self.elastic_transform(volume)

        if random.random() > 0.5:
            volume, label = self.random_flip(volume, label)
        
        return volume, label

    
class abdominal_dataset( torch.utils.data.Dataset ):

    def __init__( self, data_dir, split, num_classes, domain = 'source', transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.domain = domain
        
        self.h5f = h5py.File(f"{self.data_dir}{self.domain}_{self.split}.h5", 'r')
        self.num_samples = len([key for key in self.h5f.keys() if key.startswith('X_')])
        
        with h5py.File(f"{self.data_dir}{self.domain}_{self.split}.h5", 'r') as h5f:
            self.data = [h5f['X_' + str(i)][()] for i in range(len(h5f.keys())//2)]
            self.labels = [h5f['Y_' + str(i)][()] for i in range(len(h5f.keys())//2)]
        
    def __getitem__(self, idx):
        X = self.data[idx]
        Y = self.labels[idx]
        
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == Y.shape[1] 
        assert X.shape[2] == Y.shape[2]
        
        data_vol = X
        label_vol = Y
        # right now is [0, 1], we need to perform min-max normalization
        #print(np.min(data_vol), np.max(data_vol), np.mean(data_vol), np.var(data_vol))

        if self.transform:
            data_vol, label_vol = self.transform(data_vol, label_vol)
            
            assert data_vol.shape[0] == 32
            assert data_vol.shape[1] == 32
            assert data_vol.shape[2] == 32

        data_vol = data_vol * 2 - 1 # min-max normalization
        data_vol = np.expand_dims( data_vol.transpose((2, 0, 1)), 0)
        data_vol = torch.from_numpy(data_vol).float()

        label_vol = label_vol.transpose((2, 0, 1))
        label_vol = torch.from_numpy(label_vol.copy()).long()

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