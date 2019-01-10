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
from .utils import sorted_alphanumeric

class TDDA_loader():
    '''
    :PATH to npy or mat files
    :npy_flag: whether to read npy files or mat files
    :drop_transient: drop first 0.8 of each simulation to obtain steady state data exclusively
    '''
    def __init__(self, PATH, split = 1, npy_flag = True, drop_transient = True):
        self.data = []
        if npy_flag:
            for file in sorted_alphanumeric(os.listdir(PATH)):
                self.data.append(np.load(f'{PATH}'f'{file}'))
        else:
            # if needed, implement loading from mat
            pass
        
        if drop_transient:
            for i, dataset in enumerate(self.data):
                self.data[i] = dataset[:,800::]
        
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
        
    def getDataLoader(self, sz_sourceDomain = 1600, num_workers = 0, shuffle = True, batch_size = 1):
        '''
        returns PyTorch train, test dataloaders
        if split = 1, returns only trainDataLoader. Otherwise, returns trainDataLoader, testDataLoader
        '''
        data_list = []
        data = []
        label= []
        # count how many data points in source and target domains
        src_count = {}
        tgt_count = {}
          
        for i,d in enumerate(self.data):
            data = np.concatenate([self.data[i][:,:d.shape[1]-1] for i in range(len(self.data))])
            label = np.concatenate([self.data[i][:,d.shape[1]-1] for i in range(len(self.data))])
        
        # shift down to 0,1,2,3 instead of 1,2,3,4
        label = label - 1
        
        assert self.split <= 1 and self.split >= 0, 'Split percentage not between 0 and 1'
        # double int label. Second column indicates domain label
        domain_lbl = np.concatenate((np.ones(sz_sourceDomain),np.zeros(len(data)-sz_sourceDomain)),axis = 0)
        label = np.transpose(np.vstack((label,domain_lbl)))
        # dependency on sklearn.cross_validation / can be eliminated                      
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=1-self.split)
        train = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train[:,0]).to(torch.long),\
                              torch.Tensor(y_train[:,1]).to(torch.long))
        src_count[0] = np.count_nonzero(y_train[:,1])
        tgt_count[0] = len(train) - src_count[0]
                
        test = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test[:,0]).to(torch.long),\
                             torch.Tensor(y_test[:,1]).to(torch.long))
        src_count[1] = np.count_nonzero(y_test[:,1])
        tgt_count[1] = len(test) - src_count[1]
        
        trainloader = DataLoader(train, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
        try:
            testloader = DataLoader(test, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers)
            print('Number of examples in train and test dataloaders: train = {}, test = {}.'\
                  .format(len(trainloader), len(testloader)))
            print('Source, target domain data points in train dataloader: {}, {}; test dataloader: {}, {}.'\
                 .format(src_count[0], tgt_count[0], src_count[1], tgt_count[1]))
            return (trainloader, testloader)
        except:
            print('Number of examples in the dataloader: {}'.format(len(trainloader)))
            print('Source, target domain data points in train dataloader: {}, {}.'\
                 .format(src_count[0], tgt_count[0]))
            return trainloader    
        
    def getSplitDataLoader(self, sz_sourceDomain = 1600, num_workers = 0, shuffle = True, batch_size = 1):
        data_list = []
        data = []
        label= []
        # count how many data points in source and target domains
        src_count = {}
        tgt_count = {}
         
        assert self.split <= 1 and self.split >= 0, 'Split percentage not between 0 and 1'
        
        sr_train, sr_test, sr_y_train, sr_y_test = train_test_split(self.data[0][:,:self.data[0][0,:].shape[0]-1],\
                 self.data[0][:,self.data[0][0,:].shape[0]-1]-1, test_size=1-self.split)
        
        for i,d in enumerate(self.data):
            data = np.concatenate([self.data[i][:,:d.shape[1]] for i in range(len(self.data))])
            label = np.concatenate([self.data[i][:,d.shape[1]-1] for i in range(len(self.data))])
        
        # shift down to 0,1,2,3 instead of 1,2,3,4
        label = label - 1
                
        # dependency on sklearn.cross_validation / can be eliminated                      
        tr_train, tr_test, tr_y_train, tr_y_test = train_test_split(data, label, test_size=1-self.split)
        
        sr_train = TensorDataset(torch.Tensor(sr_train), torch.Tensor(sr_y_train).to(torch.long))
        sr_test = TensorDataset(torch.Tensor(sr_test), torch.Tensor(sr_y_test).to(torch.long))
        tr_train = TensorDataset(torch.Tensor(tr_train), torch.Tensor(tr_y_train).to(torch.long))
        tr_test = TensorDataset(torch.Tensor(tr_test), torch.Tensor(tr_y_test).to(torch.long)) 
        
        sourceTrainLoader = DataLoader(sr_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        sourceTestLoader = DataLoader(sr_test, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers)
        targetTrainLoader = DataLoader(tr_train, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers)
        targetTestLoader = DataLoader(tr_test, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers)
                
        print('Number of examples in source domain train and test dataloaders: train = {}, test = {}.'\
              .format(len(sourceTrainLoader), len(sourceTestLoader)))
        print('Number of examples in target domain train and test dataloaders: train = {}, test = {}.'\
              .format(len(targetTrainLoader), len(targetTestLoader)))
        return (sourceTrainLoader, sourceTestLoader, targetTrainLoader, targetTestLoader)

       
        