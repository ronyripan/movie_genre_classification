# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:46:46 2023

@author: Rony Chowdhury Ripan
"""
import numpy as np
import torch.nn as nn
import torch

def accuracy(pred, target):
    '''
    input: pred and target 2d tensor of shape batch_size x output class
    output: total accuracy of that batch
    '''
    binary_pred = (pred >= 0.5).int() #turning sigmoid probabilites into binary output
    correct = 0
    for i in range(binary_pred.shape[0]):
        if (binary_pred[i] == target[i]).sum() == 27: #27 is the output class
            correct += 1

    return correct


def criterion(encoded, decoded, target, data):
    '''
    input: encoded, decoded, target and data tensors
    output: float value
    '''
    loss1 = nn.BCELoss() #binary cross-entropy loss calculated between prediction (encoded) and target labels
    loss2 = nn.MSELoss() #mean squared error loss calculate between original data and decoder generated data

    final_loss = loss1(encoded,target) + loss2(decoded, data)

    return final_loss

def train(model, device, train_loader, optimizer, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []

    correct = 0
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        encoded, decoded = model(data)
        target = target.type_as(encoded)
        
        loss = criterion(encoded, decoded, target, data)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        #storing all the batch accuracy
        correct += accuracy(encoded, target.int())
        
        
    train_loss = float(np.mean(losses))
    train_acc = (correct / ((batch_idx+1) * batch_size))*100
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(train_loss, correct, (batch_idx+1) * batch_size,train_acc))
    
    return train_loss, train_acc

def test(model, device, test_loader):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            

            # Predict for data by doing forward pass
            encoded, decoded = model(data)
            target = target.type_as(encoded)
           
            loss = criterion(encoded, decoded, target, data)
            # Append loss to overall test loss
            losses.append(loss.item())
            
            correct += accuracy(encoded, target.int())

    test_loss = float(np.mean(losses))
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    
    #log_file.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #   test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, test_acc