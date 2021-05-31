"""
Loss functions

Anonymized code submitted alongide 
the manuscript titled 
Learning Granger Causal Feature Representations 

please do not distribute
"""


import torch
import torch.nn.functional as F

def binary_cross_entropy_loss(output, target):
    return F.binary_cross_entropy(output, target, reduction='none')

def hinge_loss(output, target):
    hl = 1 - torch.mul(output, target)
    hl[hl < 0] = 0
    return hl

## x.shape = y.shape = (batch_size, tbp)
def lag_cor(x,y,lag):
    dim = -1
    ## no checks on x,y dimensions
    x = x[:, lag:]
    y = y[:, :-lag]
    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)
    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)
    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)
    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)
    corr = bessel_corrected_covariance / (x_std * y_std)
    return corr.mean()


## h.shape = (batch_size, tbp, nlatents)
## target.shape = (batch_size, tbp)
def granger_loss(h, target, maxlag = 1, idx = 0):
    gl = torch.zeros(()) 
    yy = h[:,maxlag:,idx].flatten() ## skip first maxlag elements 
    xx = torch.zeros(yy.shape + (maxlag * (h.shape[2] + 1)+1,)) 
    for lag in range(maxlag):
        xx[:, lag*h.shape[2]:(lag+1)*h.shape[2]] = h[:, (maxlag - lag - 1):-(lag + 1), :].reshape((xx.shape[0],) +  (h.shape[2],))
    xx[:, -maxlag - 1] = 1
    for lag in range(maxlag):
        xx[:, -(lag+1)] = target[:,  (maxlag - lag - 1):-(lag+1)]
    PP = torch.linalg.pinv(xx[:,0:-maxlag]) 
    yypred = torch.mv(xx[:,0:-maxlag], torch.mv(PP, yy))
    loss = torch.nn.MSELoss()
    loss_base = loss(yy, yypred)
    PP1 = torch.linalg.pinv(xx) 
    yypred1 = torch.mv(xx, torch.mv(PP1, yy))
    loss_with = loss(yy, yypred1)
    return torch.log(loss_with) - torch.log(loss_base) 
