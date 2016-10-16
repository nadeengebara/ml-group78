# -*- coding: utf-8 -*-
import numpy as np
import costs
import helpers


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    gradient=-(1/len(y))*((np.transpose(tx)).dot(y-(tx.dot(w))))
    return gradient
    
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    #define w_initial here
    ws = [w_initial]
    losses = []
    for n_iter in range(max_iters):
        gradient=compute_gradient(y,tx,w)
        loss=compute_loss_MSE(y, tx, w)
        w=w-(gamma*gradient)
        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
#output minimum loss w values
    return losses, ws

def stochastic_gradient_descent(y, tx,batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # Internally define initial_w
    ws = [initial_w]
    losses = []
    w = initial_w
    i = 0
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        i = i + 1
        if (i>max_epochs):
            return losses, ws
        gradient=compute_gradient(minibatch_y,minibatch_tx, w)
        loss=compute_loss_MSE(y, tx, w)
        w=w-(gamma*gradient)
        ws.append(np.copy(w))
        losses.append(loss)
        
def compute_subgradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    ones_vector=np.ones(len(y))
    error=y-(tx.dot(w))
    # print(subgradient)
    for i in range(len(error)):
        if (error[i]<0):
            ones_vector[i]=-1  
        if (error[i]==0):
            print("encountered nondifferentiable point")
    subgradient=(1/len(y))*(-1*(np.transpose(tx)).dot(ones_vector))

    
       return subgradient

def subgradient_descent(y, tx, initial_w, max_iters, gamma): 
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient=compute_subgradient(y,tx,w)
        loss=compute_loss_MAE(y, tx, w)
        w=w-(gamma*gradient)
        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
        
        
def least_squares(y, tx):
    phitphi=np.transpose(tx).dot(tx)
    phity=np.transpose(tx).dot(y)
    return np.linalg.solve(phitphi,phity)

def ridge_regression(y, tx, lamb):
    xtx=np.transpose(tx).dot(tx)
    lambdaprime=2*lamb*len(tx)
    lambdaprime_identity=lambdaprime*np.identity(tx.shape[1])
    first_term=xtx+lambdaprime_identity
    first_term_inv=np.linalg.inv(first_term)
    second_term=np.transpose(tx).dot(y)
    weight=first_term_inv.dot(second_term)
    return weight

def split_data(x, y, ratio, seed=1):
    
    # set seed
    np.random.seed(seed)
    # ***************************************************
    shuffle_indices =np.random.permutation(len(x))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]
    train_end_index=int(np.ceil(len(x)*ratio))
    test_start_index=train_end_index+1
    train_data_x=shuffled_x[0:train_end_index]
    train_data_y=shuffled_y[0:train_end_index]
    test_data_x= shuffled_x[test_start_index:len(x)]
    test_data_y= shuffled_y[test_start_index:len(x)]

    return train_data_x,train_data_y,test_data_x,test_data_y
