# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 08:28:23 2018

@author: Zymieth
"""
from .imports import *
from .utils import pred_accuracy, split_pred_accuracy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import *
import torch

class GaussianNoise(nn.Module):
    '''
    nn.Module that adds noise
    '''
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din
    
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


class simple_CNN(nn.Module):
    def __init__(self, conv_layers, dense_layers, smax_l = True):
        '''
        smax_l: leave True for softmax applied to ouput
        '''
        super().__init__()
        self.conv_layers = nn.ModuleList([nn.Conv1d(conv_layers[i], conv_layers[i + 1], kernel_size = 10, dilation=1) 
                                     for i in range(len(conv_layers) - 1)])
        self.dense_layers = nn.ModuleList([nn.Linear(dense_layers[i], dense_layers[i + 1]) 
                                     for i in range(len(dense_layers) - 1)])
# =============================================================================
#         self.bn = nn.ModuleList([nn.BatchNorm1d()])
# =============================================================================
        #self.noise = GaussianNoise(0.05)
        self.max = nn.MaxPool1d(2)
        self.smax = smax_l
        self.bn = nn.BatchNorm1d(12)
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        #if self.training():
            #x = self.noise(x)
        for l in self.conv_layers:
            x = l(x)
            x = F.relu(x)
            x = self.max(x)
        x = x.view(x.size(0), -1)
        for l in self.dense_layers:
            l_x = l(x)
            x = F.relu(l_x)
        if self.smax: return F.log_softmax(l_x, dim=-1)
        else: return torch.sigmoid(l_x)

class MLP(nn.Module):
    def __init__(self, dense_layers):
        '''
        smax_l: leave True for softmax applied to ouput
        '''
        super().__init__()
        self.dense_layers = nn.ModuleList([nn.Linear(dense_layers[i], dense_layers[i + 1]) 
                                     for i in range(len(dense_layers) - 1)])
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        for l in self.dense_layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x
 
class DANN(nn.Module):
    def __init__(self, feature_e, label_d, domain_d):
        super().__init__()
        self.feature_extractor = feature_e
        self.label_discriminator = label_d
        self.domain_discriminator = domain_d

    
    def get_parameters(self):
        return None
    
    def forward(self,x):
        x = x.view(x.size(0), 1, -1)
        z = self.feature_extractor(x)
        label = self.label_discriminator(z)
        z1 = grad_reverse(z)
        domain = self.domain_discriminator(z1)
        return  label,domain
    
    def domain_fit(self, optimizer = None, epochs = 3, sourceTrain = None, sourceTest = None,\
                   targetTrain = None, targetTest = None, print_info = True, every_iter = 300, dom_reg_l = [0.1,0.3,0.5]):
        assert epochs == len(dom_reg_l), 'Length of domain regularizer list not equal to number of epochs'
        
        no_domain = False
        opt = optimizer
        iter_len = min(len(targetTrain),len(sourceTrain))
        
        for epoch in range(epochs):
            dom_reg = dom_reg_l[epoch] 
            
            run_loss = 0
            count = 0
            s = iter(sourceTrain)
            t = iter(targetTrain)
            onesDL = DataLoader(torch.Tensor(torch.ones(iter_len)).to(torch.float),\
                            batch_size=sourceTrain.batch_size,shuffle=False,\
                            num_workers=sourceTrain.num_workers)
            one = iter(onesDL)
            zerosDL = DataLoader(torch.Tensor(torch.zeros(iter_len)).to(torch.float),\
                            batch_size=targetTrain.batch_size,shuffle=False,\
                            num_workers=targetTrain.num_workers)
            zero = iter(zerosDL)
                        
            for i in range(iter_len):
                                               
                optimizer.zero_grad()
                
                x, y = s.next()   
                o = one.next()               
                yhat, yhat_domain = self.forward(x)
                loss_s = F.nll_loss(yhat,y) 
                loss_sd = F.l1_loss(yhat_domain,o)
                
                x, y = t.next()
                z = zero.next()
                yhat, yhat_domain = self.forward(x)
                loss_td = F.l1_loss(yhat_domain,z)
                
                loss = loss_s + dom_reg * (loss_sd + loss_td) 
                loss.backward()
                opt.step()
                run_loss += loss.item()

                if print_info == True and i % every_iter == 0 and i != 0:
                    print('Label loss: {}, domain loss (source): {}, domain loss (target): {}'\
                          .format(loss_s, loss_sd, loss_td))
                    print('Train accuracy on source domain: {}'\
                          .format(split_pred_accuracy(self, sourceTrain)))
                    print('Train accuracy on target domains (different rpm): {}'\
                          .format(split_pred_accuracy(self, targetTrain)))
                    try:
                        print('Test accuracy on source domain: {}'\
                              .format(split_pred_accuracy(self, sourceTest)))       
                        print('Test accuracy on target domains (different rpm): {}'\
                              .format(split_pred_accuracy(self, targetTest))) 
                    except:
                        pass                       
        return None
            
            
        

class RDANN(DANN):
    def __init__(self, feature_e, label_d, domain_d):
        super().__init__(feature_e, label_d, domain_d)
    
    def forward(self,x):
        return super().forward(x)
    
    def domain_fit(self,
                   optimizer = None,
                   epochs = 3, 
                   sourceTrain = None,
                   sourceTest = None,
                   targetTrain = None,
                   targetTest = None,
                   rpm_range = None,
                   print_info = True,
                   every_iter = 300,
                   dom_reg_l = [0.1,0.3,0.5],
                   dom_range = 1):
        
        no_domain = False
        opt = optimizer
        iter_len = min(len(targetTrain),len(sourceTrain))
        
        for epoch in range(epochs):
            dom_reg = dom_reg_l[epoch] 
            
            run_loss = 0
            count = 0
            s = iter(sourceTrain)
            t = iter(targetTrain)                        
            for i in range(iter_len):                                              
                optimizer.zero_grad()
                
                x, y, d = s.next()                
                yhat, yhat_domain = self.forward(x)
                loss_s = F.nll_loss(yhat,y) 
                loss_sd = F.l1_loss(yhat_domain,d)/dom_range
                
                x, y, d = t.next()
                yhat, yhat_domain = self.forward(x)
                loss_td = F.l1_loss(yhat_domain,d)/dom_range
                
                loss = loss_s + dom_reg * (loss_sd + loss_td) 
                loss.backward()
                opt.step()
                run_loss += loss.item()

                if print_info == True and i % every_iter == 0 and i != 0:
                    print('Label loss: {}, domain loss (source): {}, domain loss (target): {}'\
                          .format(loss_s, loss_sd, loss_td))
                    print('Train accuracy on source domain: {}'\
                          .format(split_pred_accuracy(self, sourceTrain)))
                    print('Train accuracy on target domains (different rpm): {}'\
                          .format(split_pred_accuracy(self, targetTrain)))
                    try:
                        print('Test accuracy on source domain: {}'\
                              .format(split_pred_accuracy(self, sourceTest)))       
                        print('Test accuracy on target domains (different rpm): {}'\
                              .format(split_pred_accuracy(self, targetTest))) 
                    except:
                        pass                       
        return None
        
                    
        
class VibNet(nn.Module):
    '''
    Multi-head TCN for raw single channel sequential data
    Takes lists of layers to initialize an instance. path 1, path 2, path 3, shared convolutional layers, dense layers.
    e.g m = VibNet([1, 3, 1], [1, 3, 1], [1, 3, 1], [3,3,12,64,256], [256, 4])
    Attention size for the concatenated convolutional paths is hard coded and has to be determined for different choices 
    of dilation
    '''
    def __init__(self, conv_layers1, conv_layers2, conv_layers3, conv_post, layers):
        super().__init__()
        # path 1
        self.conv_layers1 = nn.ModuleList([nn.Conv1d(conv_layers1[i], conv_layers1[i + 1], kernel_size = 10, dilation=1) 
                                     for i in range(len(conv_layers1) - 1)])
        # path 2
        self.conv_layers2 = nn.ModuleList([nn.Conv1d(conv_layers2[i], conv_layers2[i + 1], kernel_size = 10, dilation=10) 
                                     for i in range(len(conv_layers2) - 1)])
        # path 3
        self.conv_layers3 = nn.ModuleList([nn.Conv1d(conv_layers3[i], conv_layers3[i + 1], kernel_size = 10, dilation=20) 
                                     for i in range(len(conv_layers3) - 1)])
        # shared convolutions
        self.conv_post = nn.ModuleList([nn.Conv1d(conv_post[i], conv_post[i + 1], kernel_size = 15, dilation=1) 
                                     for i in range(len(conv_post) - 1)])
        # dense layer
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        
        # gaussian noise for data augmentation at runtime
        self.noise = GaussianNoise(0.05)
        
        # several other useful modules that might or might not be used
        self.max = nn.MaxPool1d(4) 
        self.dropout = nn.Dropout(p=0.33)
        self.gp = nn.AdaptiveMaxPool1d(1)
        self.mp = nn.AdaptiveMaxPool1d(2000)
        # hardcoded attention size. has to be determined for each combination of dilations, depth of conv paths etc.
        # needs to be a multiple of 3. Optional
        self.attention = utils.Attention(5586)
    
    def get_weights(self, layer):
        '''
        Obtain the weights of path 1, path 2, path 3 layers.
        input: layer number. returns (w1, w2, w3), the weights of each path at the specified layer
        '''
        weights1 = self.conv_layers1[layer].weight.data.cpu().numpy()
        weights2 = self.conv_layers2[layer].weight.data.cpu().numpy()
        weights3 = self.conv_layers3[layer].weight.data.cpu().numpy()
        
        return weights1, weights2, weights3
    
    def plot_latent_components(self, val, psd_flag = None):
        '''
        Takes as input a single validation time series
        Plots the latent time series activations (or power spectral density if psd_flag = True)
        for each of the path's final layer
        '''
        val = val.view(1,1,-1)
        if psd_flag:
            p1, p2, p3 = m.evaluate_paths(val)
            fig = plt.figure(figsize=(20,20), facecolor = 'white')
            fig.add_subplot(4,1,1) 
            plt.plot(get_power(val,0)[10::], color = 'black')
            fig.add_subplot(4,1,2)
            plt.plot(get_power(p1.view(1,1,-1),0)[10::], color = 'red')
            fig.add_subplot(4,1,3)
            plt.plot(get_power(p2.view(1,1,-1),0)[10::], color = 'green')
            fig.add_subplot(4,1,4)
            plt.plot(get_power(p3.view(1,1,-1),0)[10::], color = 'blue')    
        else:
            p1, p2, p3 = self.evaluate_paths(val)
            fig = plt.figure(figsize=(20,20), facecolor = 'white')
            fig.add_subplot(4,1,1) 
            plt.plot(val.view(-1).detach().cpu().numpy(), color = 'black')      
            fig.add_subplot(4,1,2)
            plt.plot(p1.view(-1).detach().cpu().numpy(), color = 'red')
            fig.add_subplot(4,1,3)
            plt.plot(p2.view(-1).detach().cpu().numpy(), color = 'green')
            fig.add_subplot(4,1,4)
            plt.plot(p3.view(-1).detach().cpu().numpy(), color = 'blue')
        
    def evaluate_paths(self, x):
        '''
        Used to obtain latent time series at inference time
        Uses the learned weights and the same architecture to manipulate validation data
        '''
        x = x.view(x.size(0), 1, -1)
        if m.training:
            x = self.noise(x)
        #save input
        s = x      
        for l in self.conv_layers1:
            l_x = l(x)
            x = torch.tanh(l_x) 
            #x = self.max(x)
        #x = self.conv_dil1(x)
        x1 = x
        x = s
        for l in self.conv_layers2:
            l_x = l(x)
            x = torch.tanh(l_x)
            #x = self.max(x)
        #x = self.conv_dil3(x)
        x2 = x
        x = s
        for l in self.conv_layers3:
            l_x = l(x)
            x = torch.tanh(l_x)
            #x = self.max(x)
        #x = self.conv_dil5(x)
        x3 = x
        return x1, x2, x3
       
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        if m.training:
            x = self.noise(x)
        #save input
        s = x
        for l in self.conv_layers1:
            l_x = l(x)
            x = torch.tanh(l_x)
            #x = self.max(x)
        #x = self.conv_dil1(x)
        x1 = x
        x = s
        for l in self.conv_layers2:
            l_x = l(x)
            x = torch.tanh(l_x)
            #x = self.max(x)
        #x = self.conv_dil3(x)
        x2 = x
        
        x = s
        for l in self.conv_layers3:
            l_x = l(x)
            x = torch.tanh(l_x)
            #x = self.max(x)
        #x = self.conv_dil5(x)
        x3 = x
        
        x = torch.cat((x1, x2, x3), 2)
        x = self.attention(x)
        x = x.view(s.size(0), 3, -1)
        
        for l in self.conv_post:
            l_x = l(x)
            bn = nn.BatchNorm1d(x.size())
            x = F.relu(l_x)
            #x = self.max(x)
        x = self.gp(x)
        
        x = x.view(x.size(0), -1)   
        for l in self.layers:
            l_x = l(x)
            bn = nn.BatchNorm1d(l_x.size())
            x = F.relu(l_x)
        return F.log_softmax(l_x, dim=-1)