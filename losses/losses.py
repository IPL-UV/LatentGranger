# BSD 3-Clause License (see LICENSE file)
# Copyright (c) Image and Signaling Process Group (ISP) IPL-UV 2021,
# All rights reserved.

"""
Loss functions

"""
import torch


# x.shape = y.shape = (batch_size, tbp)
def lag_cor(x, y, lag):
    dim = -1
    # no checks on x,y dimensions
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


# x.shape = y.shape = (batch_size, tbp)
def cor(x, y, lag):
    dim = -1
    # no checks on x,y dimensions
    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)
    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)
    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)
    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)
    corr = bessel_corrected_covariance / (x_std * y_std)
    return corr.mean()


# h.shape = (batch_size, tbp, nlatents)
# target.shape = (batch_size, tbp)
def granger_loss(h, target, maxlag=1, idx=0):
    yy = h[:, maxlag:, idx].flatten()  # skip first maxlag elements
    xx = torch.zeros(yy.shape + (maxlag * (h.shape[2] + 1)+1,))
    for lag in range(maxlag):
        xx[:, lag*h.shape[2]:(lag+1)*h.shape[2]] = \
            h[:, (maxlag - lag - 1):-(lag + 1), :].reshape((xx.shape[0],) +
                                                           (h.shape[2],))
    xx[:, -maxlag - 1] = 1.0
    for lag in range(maxlag):
        xx[:, -(lag+1)] = target[:,  (maxlag - lag - 1):-(lag+1)]
    PP = torch.linalg.pinv(xx[:, 0:-maxlag])
    yypred = torch.mv(xx[:, 0:-maxlag], torch.mv(PP, yy))
    loss = torch.nn.MSELoss()
    loss_base = loss(yy, yypred)
    PP1 = torch.linalg.pinv(xx)
    yypred1 = torch.mv(xx, torch.mv(PP1, yy))
    loss_with = loss(yy, yypred1)
#   return loss_with / (loss_base - loss_with)
#   return torch.log(loss_with) - torch.log(loss_base)
    return torch.log(loss_base + loss_with) - torch.log(loss_base)
#   return loss_with / loss_base


# h.shape = (batch_size, tbp, nlatents)
# target.shape = (batch_size, tbp)
def granger_simple_loss(h, target, maxlag=1, idx=0):
    yy = h[:, maxlag:, idx].flatten()  # skip first maxlag elements
    xx = torch.zeros(yy.shape + (maxlag * 2 + 1,))
    for lag in range(maxlag):
        xx[:, lag] = h[:, (maxlag - lag - 1):-(lag + 1), idx]
    xx[:, -maxlag - 1] = 1.0
    for lag in range(maxlag):
        xx[:, -(lag+1)] = target[:,  (maxlag - lag - 1): -(lag + 1)]
    PP = torch.linalg.pinv(xx[:, 0:-maxlag])
    yypred = torch.mv(xx[:, 0:-maxlag], torch.mv(PP, yy))
    loss = torch.nn.MSELoss()
    loss_base = loss(yy, yypred)
    PP1 = torch.linalg.pinv(xx)
    yypred1 = torch.mv(xx, torch.mv(PP1, yy))
    loss_with = loss(yy, yypred1)
#   return loss_with / (loss_base - loss_with)
#   return torch.log(loss_with) - torch.log(loss_base)
    return torch.log(loss_base + loss_with) - torch.log(loss_base)
#   return loss_with / loss_base


# h.shape = (batch_size, tbp, nlatents)
def uncor_loss(h):
    if h.shape[2] <= 1:
        return torch.zeros(())
    else:
        cc = torch.zeros(())
        for i in range(h.shape[2]):
            for j in range(h.shape[2] - i - 1):
                cc = cc + cor(h[:, :, i], h[:, :, i + j + 1], 0)**2
        return cc


# ||A*A**t - I ||**2 with A a matrix
def orth_loss(A):
    n = A.shape[0]
    A = torch.matmul(A, torch.transpose(A, 0, 1))
    loss = torch.nn.MSELoss(reduction='mean')(A, torch.eye(n))
    return loss
