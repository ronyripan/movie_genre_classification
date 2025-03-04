# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:23:58 2023

@author: Rony Chowdhury Ripan
"""

import pandas as pd
from dataset import MovieDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from train import train, test
from model import AutoEncoder
import torch
import cv2
from torch.utils.data import Dataset
import numpy as np

if __name__ == '__main__':
    dataset_path = "D:\\OneDrive - University of Central Florida\\Course Work\\Fall 23\\CAP-5415-CV\\project\\final_dataset.json"
    
    df = pd.read_json(dataset_path, orient='records', lines=True) #reading the data
    df = df.iloc[:100]
    
    # transformations for image resizing and normalization
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # Convert PIL Image to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize
    ])
    
    # Splitting the dataset into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) #80-20 split
    
    # Creating instances of the MovieDataset for train and test sets
    train_dataset = MovieDataset(dataframe=train_df, transform=transform)
    test_dataset = MovieDataset(dataframe=test_df, transform=transform)
    
    # Create PyTorch DataLoaders for train and test sets
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    model = AutoEncoder().to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) #Adam Optimizer
    best_accuracy = 0.0
        
    epoch_lst = []
    train_loss_lst = []
    train_accuracy_lst = []
    test_loss_lst = []
    test_accuracy_lst = []
    no_epoch = 1
    
    for epoch in range(1, no_epoch+1):
        print("Epoch: {}\n".format(epoch))
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, batch_size)
        test_loss, test_accuracy = test(model, device, test_loader)
        
        epoch_lst.append(epoch) #storing epoch information in a epoch list
        train_loss_lst.append(train_loss) #storing train loss information in a training loss list
        train_accuracy_lst.append(train_accuracy) #storing train accuracy information in a training accuracy list
        test_loss_lst.append(test_loss) #storing test loss information in a test loss list
        test_accuracy_lst.append(test_accuracy) #storing test accuracy information in a test accuracy list
            
        if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
    
        #storing all the epoch, loss and accuracy information in a dataframe
        history = pd.DataFrame({'epoch': epoch_lst, 'train_loss': train_loss_lst, 'train_accuracy': train_accuracy_lst, 'test_loss': test_loss_lst, 'test_accuracy': test_accuracy_lst}) 
    
        #history.to_csv('{}_history.csv'.format(FLAGS.mode), index=False) #downloading history dataframe as csv file
        
        print("accuracy is {:2.2f}".format(best_accuracy))
        print("Training and evaluation finished")
    