#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display as ipy_display, clear_output


# In[2]:


deplabv3 = __import__('Deeplabv3')
import networks


# In[3]:


import dataset


# # parameters

# In[4]:


num_classes = 5
batch_size = 72
suffix = '70000'
continue_run = False


# # CUDA

# In[5]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# # Path

# In[6]:


source_mr_train_dir = "/scratch/rs37890/CARC/Medical_images_source_MR/data/data/h5py/"
source_mr_test_dir = "/scratch/rs37890/CARC/Medical_images_source_MR/data/data/h5py/"


# # sample from dataset

# In[7]:


def sample_batch(dataset, batch_size=20, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    data_dir = dataset.data_dir
    num_samples = len(dataset)
    sample_indices = np.random.choice(num_samples, batch_size, replace=True) # replace=True allow repeat

    images = []
    labels = []

    for idx in sample_indices:
        
        data_vol, label_vol = dataset[idx]
        
        images.append(data_vol)
        labels.append(label_vol)

    images = torch.stack(images)
    labels = torch.stack(labels)

    return images, labels


# # Loss

# In[8]:


# https://gist.github.com/yang-zhang/217dcc6ae9171d7a46ce42e215c1fee0
def masked_ce_loss(num_classes=20, class_to_ignore=None):
    mask = np.ones(num_classes)
    if class_to_ignore is not None:
        mask[class_to_ignore] = 0
    mask = torch.tensor(mask, dtype=torch.float32)

    def masked_loss(y_pred, y_true, from_logits = False, F_corss_entropy_label = True):

        # Preprocess data
        if not from_logits and not F_corss_entropy_label: # False = (True and False) if I want to use F_corss_entropy_label
            print('wrong')
            y_pred = F.softmax(y_pred, dim=1)
        
        if F_corss_entropy_label:
            
            loss = F.cross_entropy(y_pred, y_true, weight = mask.to(y_true.device) , reduction='mean')

            # For cross entropy: y_pred has the shape [batch_size, num_classes, height, width]
            #                    y_true has the shape [batch_size, height, width].

            #The F.cross_entropy function understands this difference in dimensions and 
            #calculates the loss accordingly by comparing the predicted class probabilities 
            #(y_pred) to the ground truth class indices (y_true).
            
        else:
            # not sucess for manually implement don't know why
            # Clip y_pred values to avoid numerical instability
            y_pred = torch.clamp(y_pred, 1e-10, 1)

            # Move mask tensor to the same device as y_true; input need to be (batch, height, weight), channel = 1 squeezed before
            y_true_one_hot  = F.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2)

            # (8, 5, 256, 256) * Shape: (1, 5, 1, 1)
            y_true_one_hot_masked = y_true_one_hot  * mask.to(y_true.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  



            # Calculate cross entropy loss for each class
            ce_loss = -y_true_one_hot_masked * torch.log(y_pred)

            # Sum the loss for each class and take the mean over the batch
            loss = ce_loss.sum(dim=[1, 2, 3]).mean()

        return loss

    return masked_loss


# # Initialize

# In[9]:


dataloader = dataset.get_dataloader( source_mr_train_dir,  source_mr_test_dir, num_classes, batch_size )

train_dataset = dataloader["train"].dataset
test_dataset = dataloader["test"].dataset


# In[10]:


dpv3 = deplabv3.DeepLabV3(num_classes)
classifier = networks.classifier(num_classes)

dpv3 = dpv3.to(device)
classifier = classifier.to(device)

# parallel
dpv3 = torch.nn.DataParallel(dpv3)
classifier = torch.nn.DataParallel(classifier)


# In[11]:


loss_function = masked_ce_loss(num_classes, None)  

optimizer_dpv3 = torch.optim.Adam(dpv3.parameters(), lr=1e-4, eps=1e-6, weight_decay=1e-6)
optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-4, eps=1e-6, weight_decay=1e-6)


# In[12]:


if continue_run:
    
    dpv3_checkpoint = torch.load('./record-data/' + 'dpv3_weights_' + suffix + '.pth')
    classifier_checkpoint = torch.load('./record-data/' + 'classifier_weights_' + suffix + '.pth')

    dpv3.load_state_dict(dpv3_checkpoint)
    classifier.load_state_dict(classifier_checkpoint)
    print("Loaded model weights")


# # Training

# In[13]:


restart_epoch = 0
epochs = 600000
epoch_step = 250

fig, ax = plt.subplots(1, figsize=(10, 7))

if continue_run:
    with open('record-data/' + "loss_history" + "_" + suffix + ".pkl", "rb") as file:
        loss_history = pickle.load(file)
        
    restart_epoch = len(loss_history)
    
else:
    loss_history = []

start_time = time.time()

for epoch in range(restart_epoch, epochs):
    
    dpv3.train()
    
    source_train_data, source_train_labels = sample_batch(train_dataset, batch_size, seed=epoch)
    
    source_train_data = source_train_data.to(device)
    source_train_labels = source_train_labels.to(device)
    
    # zero grad
    optimizer_dpv3.zero_grad()
    optimizer_classifier.zero_grad()
    
    # predicts
    outputs = dpv3(source_train_data) # batch_size, num_classes = 5, 32 = D, 256 = H, 256 = W
    outputs = classifier(outputs) 
    
    # Loss 
    loss = loss_function(outputs, source_train_labels)
    loss.backward()
    
    # step
    optimizer_dpv3.step()
    optimizer_classifier.step()
    
    loss_history.append(loss.item())
    print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    
    if epoch % epoch_step == 0 or epoch < 1000:
        if epoch != 0:
            ax.clear()
            ax.plot(np.log(np.asarray(loss_history))) #not understand why do np.log； <1 will negative

        ax.set_title("Training loss on source domain")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Log Loss")

        clear_output(wait=True)  # Add this line to clear the output
        ipy_display(plt.gcf())  # Use the new name for the display function
        time.sleep(1e-3)  # Add a small delay for smoother updates
        
        
    
    if epoch % 5000 == 0 or epoch == epochs - 1:
        
        torch.save(dpv3.state_dict(), 'record-data/' + 'dpv3_weights' + "_" + str(epoch) + '.pth')
        torch.save(classifier.state_dict(),'record-data/' + 'classifier_weights' + "_" + str(epoch) + '.pth')
        
        with open('record-data/' + "loss_history" + "_" + str(epoch) + ".pkl", "wb") as file:
            pickle.dump(loss_history, file)


# In[ ]:


torch.save(dpv3.state_dict(), 'record-data/' + 'dpv3_weights' + "_" + str(epoch) + '.pth')
torch.save(classifier.state_dict(),'record-data/' + 'classifier_weights' + "_" + str(epoch) + '.pth')

with open('record-data/' + "loss_history" + "_" + str(epoch) + ".pkl", "wb") as file:
    pickle.dump(loss_history, file)


# In[ ]:




