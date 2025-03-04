# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:47:07 2023

@author: Rony Chowdhury Ripan
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, input_shape):
        super(ConvNet, self).__init__()
        self.in_shape = input_shape
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5))
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,27) #27 output class
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        
        
    def forward(self, x):
        x = self.dropout1(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.dropout1(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = self.dropout1(self.pool(self.relu(self.bn3(self.conv3(x)))))
        x = self.dropout1(self.pool(self.relu(self.bn4(self.conv4(x)))))

        x = self.flatten(x)
        x = self.dropout2(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        
        return x
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5))
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(9216, 128) #last layer output is 64 x 12 x 12 = 9216
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,27) #27 output class
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        
        
    def forward(self, x):
        x = self.dropout1(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.dropout1(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = self.dropout1(self.pool(self.relu(self.bn3(self.conv3(x)))))
        x = self.dropout1(self.pool(self.relu(self.bn4(self.conv4(x)))))

        x = x.view(-1,9216) #flattening the layers
        x = self.dropout2(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x)) 
        
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc4 = nn.Linear(27, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, 64*16*16)  #upsampling to match 3 x 256 x 256
        self.dropout3 = nn.Dropout(p=0.5)
        self.dropout4 = nn.Dropout2d(p=0.2)
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(16)
        self.deconv4 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = x.view(-1, 64, 16, 16)  # flatten
        
        x = self.dropout4(self.relu(self.bn5(self.deconv1(x))))
        #print(x.shape)
        x = self.dropout4(self.relu(self.bn6(self.deconv2(x))))
        #print(x.shape)
        x = self.dropout4(self.relu(self.bn7(self.deconv3(x))))
        #print(x.shape)
        x = self.deconv4(x)
        
        return x

class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder() 
        self.decoder = Decoder() #

    def forward(self, x):

        encoded = self.encoder(x) 
        decoded = self.decoder(encoded) 

        return encoded, decoded
    
