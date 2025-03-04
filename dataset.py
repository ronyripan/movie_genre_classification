# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:17:58 2023

@author: Rony Chowdhury Ripan
"""
import cv2
from torch.utils.data import Dataset
import numpy as np

# Custom dataset class to load images with labels

class MovieDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  #image_path are in the first column
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV reads images in BGR format. converting it to rgb

        if self.transform:
            image = self.transform(image)

        label = self.dataframe.iloc[idx, 1]  #the labels are in the second column
        label = np.array(label)
        return image, label
