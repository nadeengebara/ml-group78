import numpy as np
import matplotlib.pyplot as plt
from costs import *
from ToolBox import *
from plots import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_):
    #test_x=np.zeros(len(k_indices[k]),x.shape[1])
    test_y=y[k_indices[k]]
    test_x=x[k_indices[k]]
    train_y=y
    train_x=x
    train_x=np.delete(train_x,k_indices[k],axis=0)
    train_y=np.delete(train_y,k_indices[k])
    weight=ridge_regression(train_y,train_x, lambda_)
    loss_tr=compute_loss_MSE(train_y, train_x, weight)
    loss_te=compute_loss_MSE(test_y,test_x, weight)
    return weight,loss_tr, loss_te

         
    # ********************************
