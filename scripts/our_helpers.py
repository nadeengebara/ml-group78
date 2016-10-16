
import numpy as np
import matplotlib.pyplot as plt

def nb_invalid_data(col_x, invalid_value=-999):
    """Counts the number of invalid data points in a feature"""
    col = tX[:, 0]
    return col_x[col_x == invalid_value].shape[0]
    

def clean_up_invalid_values_mean(col_x, invalid_value=-999):
    col_x
    mean = col_x[col_x != -999].mean()
    col_x[col_x == -999] = mean
    return col_x

def clean_up_invalid_values_median(col_x, invalid_value=-999):
    col_x
    col=col_x[col_x != -999]
    median=np.median(col)
    col_x[col_x == -999] = median
    return col_x


def feature_hist(X,col_x, bins=50, title=''):
    plt.hist(col_x, bins=bins)
    plt.title(title)
    plt.ylim([0,  X.shape[0]])
    plt.grid(True)
    
def feature_plot(X):
    nb_cols_subplot = 2
    nb_rows_subplot = np.ceil(X.shape[1]/nb_cols_subplot)
    plt.figure(figsize=(10,50))
    for i in range(X.shape[1]):
        plt.subplot(nb_rows_subplot, nb_cols_subplot, i + 1)
        feature_hist(X,X[:,i], bins=50, title='feature ' + str(i + 1))
plt.show()

###############Standardizations####################3

def regular_standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    return x

def compute_correlation(X,y):
    corr=np.zeros(X.shape[1])
    for i in (range(X.shape[1])):
        corr[i]=np.corrcoef(X[::,i],y)[0,1]
    return corr


def normalize_data(X):
    maximum=np.max(X,axis=0)
    minimum=np.min(X,axis=0)
    return (X-minimum)/(maximum-minimum)
    
    
    
def correlation_plot(V,length,title=''):
    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.yticks(np.linspace(-0.7,0.7,14))
    plt.ylim(-0.7,0.7)
    plt.grid(True)
    X=np.linspace(0,length,length)
    Y1=V
    plt.bar(X,Y1, facecolor='#ff9999', edgecolor='white')
    plt.show()
    
#def observe_features(X,Y,title=''):
#    i=0
   # plt.figure(figsize=(10,50))
    # plt.title(title)
    #nb_cols_subplot = 2
    #nb_rows_subplot = np.ceil(X.shape[1]/nb_cols_subplot)
    #for i in range(X.shape[1]):
    #plt.subplot(nb_rows_subplot, nb_cols_subplot, i + 1)
    #plt.yticks(np.linspace(-1.2,1.2,14))
    #plt.ylim(-1.2,1.2)
    #plt.grid(True)
    #X=X[::,i]
    #plt.bar(X,Y, facecolor='#ff9999', edgecolor='white')
   
    #plt.show()
    