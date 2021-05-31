"""
utility function

Anonymized code submitted alongide 
the manuscript titled 
Learning Granger Causal Feature Representations 

please do not distribute
"""


import numpy as np
import matplotlib.pyplot as plt 


def plot_output(x):
    if len(x.shape) > 2:
        figure, axis = plt.subplots(1, x.shape[2])
        for j in range(x.shape[2]):
            axis[j].imshow(x[:,:,j])
    else:
        plt.imshow(x) 
    plt.show()


def plot_latent(h, target, path=None):
    target = np.squeeze(target.detach().numpy())
    target = (target - np.mean(target)) / np.std(target)
    plt.plot(target, color = 'red', label = 'target')
    for i in range(h.shape[1]):
        h0 = np.squeeze(h[:,i].detach().numpy())
        h0 = (h0 - np.mean(h0)) / np.std(h0)
        plt.plot(h0, label = i)
    plt.legend()
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
