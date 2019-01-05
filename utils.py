# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 22:27:26 2018

@author: Zymieth
"""
from .imports import *

def get_power(time_series, ch):
    '''
    Computes the power spectral density of a 1D time series at a specific channel
    Inputs:
    time_series = 1D time series of size (batch_size, channel, -1). type: torch.Tensor.cuda()
    ch = channel of which to compute the power spectral density
    Ouputs: power spectral density, type: numpy array
    '''
    re = torch.rfft(time_series, 1).detach().cpu().numpy()[0,ch,:,0]
    im = torch.rfft(time_series, 1).detach().cpu().numpy()[0,ch,:,1]
    return np.sqrt(re**2+im**2)

def fourier_split(psd, size):
    '''
    Split a frequency signal into three chunks of size = size
    3*size has to be less or equal than original size
    '''
    return psd[0:size],psd[size:2*size],psd[2*size:3*size]

def pred_accuracy(m, valloader, val_size, domain_adaptation = False):
    '''
    Calculates accuracy from a given model and test DataLoader
    '''
    m.eval()
    count = 0
    y_hat = np.zeros(int(val_size))
    numberSourceDomain = 0
    if not domain_adaptation: dom_flag = 1
    
    for i, data in enumerate(valloader):
        if domain_adaptation: x_val, y_val, dom_flag = data
        else:
            try:
                x_val, y_val = data
            except:
                x_val, y_val, _ = data
            
        if dom_flag == 0:
            # sampled example from source domain, increase counter and ignore rest
            numberSourceDomain += 0
                
        if dom_flag == 1 or domain_adaptation == False:
            pr = m(x_val.view(1,-1))
            val, indx = torch.max(pr,1)
            if y_val - indx == 0:
                count += 1
    return count/int(val_size - numberSourceDomain)

class Attention(nn.Module):
    '''
    Scaled dot product attention
    '''
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = new_parameter(attention_size, 1)
    
    def get_param(self):
        return self.attention

    def forward(self, x_in):
        # after this, we have (batch, dim1) with a diff weight per each cell
        attention_score = torch.matmul(x_in, self.attention).squeeze()
        attention_score = F.softmax(attention_score).view(x_in.size(0),-1)
        scored_x = torch.sum(x_in * attention_score, dim=1)

        # now, sum across dim 1 to get the expected feature vector
        #condensed_x = torch.sum(scored_x, dim=1)
        return scored_x

def tensor_from_dataloader(loader):
    x = {}
    y = {}
    for i, data in enumerate(loader):
        x[i], y[i] = data
    return x, y


def new_parameter(*size):
    '''
    Creates and initializes a cuda FloatTesor parameter
    '''
    out = nn.Parameter(FloatTensor(*size).cuda())
    nn.init.xavier_normal_(out)
    return out


def readFromFolder(PATH):
    '''
    Read mat files from specified PATH
    '''
    list_of_data = []
    for file in os.listdir(PATH):
        x = sio.loadmat(f'{PATH}'+ file, appendmat=True)
        list_of_data.append(x)
    return list_of_data