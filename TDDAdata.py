# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 12:03:19 2019

@author: Zymieth
"""
import os
import numpy as np
from torch.utils.data import *
from sklearn.model_selection import train_test_split
import torch

class TDDA_loader():
    '''
    :PATH to npy or mat files
    :npy_flag: whether to read npy files or mat files
    :drop_transient: drop first 0.8 of each simulation to obtain steady state data exclusively
    '''
    def __init__(self, PATH, split = 1, npy_flag = True, drop_transient = True):
        self.data = []
        if npy_flag:
            for file in os.listdir(PATH):
                self.data.append(np.load(f'{PATH}'f'{file}'))
        else:
            # if needed, implement loading from mat
            pass
        
        if drop_transient:
            for dataset in self.data:
                dataset = dataset[:,800::]
        
        # assume 100% training if not specified otherwise
        self.split = split
    
    def getData(self, idx):
        return self.data[idx]
        
    def train_to_test_ratio(percentage):
        '''
        Sets a variable that determines the split between train and test.
        Split made inside the same domain (sub-dataset)
        Arg assumed to be between 0 (0% train 100% test) and 1 (100% train and 0% test).
        '''
        self.split = percentage
        
    def getDataLoader(self, sz_sourceDomain = 400, num_workers = 0, shuffle = True, batch_size = 1):
        '''
        returns PyTorch train, test dataloaders
        if split = 1, returns only trainDataLoader. Otherwise, returns trainDataLoader, testDataLoader
        '''
        data_list = []
        data = []
        label= []
        for i,d in enumerate(self.data):
            data = np.concatenate([self.data[i][:,:2000] for i in range(len(self.data))])
            label = np.concatenate([self.data[i][:,2001] for i in range(len(self.data))])
        
        # shift down to 0,1,2,3 instead of 1,2,3,4
        label = label - 1
        
        assert self.split <= 1 and self.split >= 0, 'Split percentage not between 0 and 1'
        # double int label. Second column indicates domain label
        domain_lbl = np.concatenate((np.ones(400),np.zeros(len(data)-sz_sourceDomain)),axis = 0)
        label = np.transpose(np.vstack((label,domain_lbl)))
        # dependency on sklearn.cross_validation / can be eliminated                      
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=1-self.split)
        train = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train[:,0]).to(torch.long),\
                              torch.Tensor(y_train[:,1]).to(torch.long))
        test = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test[:,0]).to(torch.long),\
                             torch.Tensor(y_test[:,1]).to(torch.long))
        trainloader = DataLoader(train, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
        try:
            testloader = DataLoader(test, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers)
            print('Number of examples in train and test dataloaders: train = {}, test = {}.'\
                  .format(len(trainloader), len(testloader)))
            return (trainloader, testloader)
        except:
            print('Number of examples in the dataloader: {}'.format(len(trainloader)))
            return trainloader       