#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import some_routines as sr

# # 6. Logistic Regression
# 

def sigmoid(X):
    """
    Calculate the sigmoid function for vector X
    """
    return 1/(1+np.exp(-X))

def bincrossent(y, yhat):
    """
    Given yhat and y (prediction and true label).
    Calculate the binary cross entropy
    """
    m = np.shape(yhat)[0]
        
    return -1/m *( np.dot(y.T,np.log(yhat)) + np.dot(
                                        (1-y).T,np.log(1-yhat)) )

def gradient_descent(w, X, b, y, alpha, calcLoss=False):
    """
    Perform one iteration of gradient descent.
    n is amount of features.
    m is amount of examples.
    
    w     -- (nx1) array -- weights  (also called parameters theta)
    X     -- (mxn) array -- features 
    b     -- float       -- bias   
    y     -- (mx1) array -- true labels 
    alpha -- float       -- learning rate
    calcLoss -- bool -- whether to calculate and return the loss value
    
    Returns 
    w -- updated weights
    b -- updated biases
    
    """
    m = np.shape(X)[0]
    invm = 1/m
    
    # prediction
    yhat = sigmoid(np.dot(X,w)+b)
    # error
    err = yhat - y
    # derivative w.r.t the weights
    dw = invm * np.dot(X.T,err)
    # derivative w.r.t. the bias
    db = invm * np.sum(err)
    
    # Update weights and biases
    w -= alpha*dw
    b -= alpha*db
    
    if calcLoss:
        # calculate loss before the update
        loss = bincrossent(y,yhat)
        return w, b, loss
    else:        
        return w, b

def make_labels(T90):
    """
    Reads in the T90 values and assigns a label based on T90
    short (0) if T90 <= 10
    long  (1) if T90 > 10
    """
    labels = np.array(T90>10,dtype='float')[:,np.newaxis] # return the necessary shape
    return labels

def standardize(features, setmean=False):
    """
    Standardize all features such that they have 
    mean 0 and variance 1.
    
    If setmean, set all missing features to the mean of the feature
    and add another column of 1's where it is missing and 0 where
    it is not missing, to track the influence of the missing data.
    """
    for j in range(features.shape[1]):
        # all missing features
        missing = features[:,j] == -1
        # Ignore missing data for calculation of mean and variance
        featcol = features[:,j][np.invert(missing)]
        mean = np.mean(featcol)
        std = np.std(featcol)
        # Standardize the column
        features[:,j] -= mean
        features[:,j] /= std
        if setmean:
            # Put missing on zero, which is the mean of the feature
            features[:,j][missing] = 0
            # add another column
            features = np.append(features,(missing^1)[:,np.newaxis],axis=1)
        else:
            # Put missing back to -1
            features[:,j][missing] = -1
            
    return features

def check_missing_data(features, labels):
    missing = features == -1
    # sum across rows to see how many features are missing
    nummiss = np.sum(missing,axis=0)
    print (f"Number of missing values per column: {nummiss}")
    # decide to drop last three columns
    features = features[:,:-3]
    
    # For the rest of the columns, check how many entries have all features
    missing = features == -1
    
    # Mask array that is true where data is not missing
    mask = np.sum(missing,axis=1) == 0
    # The amount of rows without missing values
    nomiss = np.sum(mask)
    print (f"Amount of datapoints that have all first {features.shape[1]} features: {nomiss}")
    
    return features, labels
    
# Use all except first 2 columns, those contain GRB and some index
# The columns indicate
# redshift, T90, log(M*/Msun), SFR, log(Z/Zsun), SSFR, AV
data = np.loadtxt('./GRBs.txt', comments='#', usecols=(2,3,4,5,6,7,8))
# Feature column labels for plotting purposes
flabels = ['Redshift', 'logM', 'SFR', 'logZ', 'SSFR', 'AV']

# Labels (1 or 0) are based on T90 values. Calculate them
T90 = data[:,1] 
labels = make_labels(T90) # shape (235,1)

# Check the zeroR prediction. This is our baseline. The algorithm should perform better.
print (f"Predicting everything as most abundant class (ZeroR) gives accuracy: {np.sum(labels)/len(labels):.2f}")

# Features are all things that are not T90
features = np.copy(data[:,[0,2,3,4,5,6]]) # shape (235,6)

# First check missing values, decide to drop last three columns
features, labels = check_missing_data(features, labels)

# Append all squared features, track missing values
missing = features == -1
sqfeatures = features**2.
sqfeatures[missing] = -1
features = np.append(features,sqfeatures,axis=1)

# Then standardize the data, setting missing values to the mean of the data
# and add another column with 1's where the value was missing and 0 where it was not
setmean = True
features = standardize(features,setmean)

# Split into training and test set, about 3:1
split = int(len(features)*3/4)

train = features[:split]
test = features[split:]

features = train
testlabels = labels[split:]
labels = labels[:split]

# Initialize weights and biases to zero
w = np.zeros((features.shape[1],1))
b = 0

numit = 10000
loss = np.zeros(numit)
alpha = 0.01
for i in range(numit):
    w, b, loss[i] = gradient_descent(w, features, b, labels, alpha, True)

# This plot is not required, if not interested can comment.
plt.plot(loss)
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.savefig('./plots/q6_loss.png')
plt.close()

# Final prediction
yhat = sigmoid(np.dot(features,w)+b)
# Decision boundary at 0.5
yhat[yhat > 0.5] = 1
yhat[yhat < 0.5] = 0
print ("Training set:")
print ("Amount predict Long", np.sum(yhat))
print ("Amount predict Short", len(yhat)-np.sum(yhat))

print ("Amount Long: ", np.sum(labels))
print ("Amount Short: ", len(labels)-np.sum(labels))

print ('Amount incorrect:', np.sum(np.abs(yhat-labels)))
print (f'Accuracy {1 - np.sum(np.abs(yhat-labels))/len(labels):.2f}')

# Plot predicted and true values
plt.title('Train set')
plt.hist(yhat.flatten(),label='Predictions',alpha=0.5)
plt.hist(labels.flatten(),label='Actual values',alpha=0.5)
plt.legend(frameon=False)
plt.xlabel('Class label')
plt.ylabel('Counts')
plt.savefig('./plots/q6_1.png')
plt.close()

if len(test) > 0:
    print ("Test set:")
    # Final prediction on test set
    yhat = sigmoid(np.dot(test,w)+b)
    # Decision boundary at 0.5
    yhat[yhat > 0.5] = 1
    yhat[yhat < 0.5] = 0
    print ("Amount predict Long", np.sum(yhat))
    print ("Amount predict Short", len(yhat)-np.sum(yhat))

    print ("Amount Long: ", np.sum(testlabels))
    print ("Amount Short: ", len(testlabels)-np.sum(testlabels))

    print ('Amount incorrect:', np.sum(np.abs(yhat-testlabels)))
    print (f'Accuracy {1 - np.sum(np.abs(yhat-testlabels))/len(testlabels):.2f}')
    
    plt.title('Test set')
    plt.hist(yhat.flatten(),label='Predictions',alpha=0.5)
    plt.hist(testlabels.flatten(),label='Actual values',alpha=0.5)
    plt.legend(frameon=False)
    plt.xlabel('Class label')
    plt.ylabel('Counts')
    plt.savefig('./plots/q6_2.png')
    plt.close()

# Merge train and test again
features = np.append(train,test,axis=0)
labels = np.append(labels,testlabels,axis=0)

# Plot all 2D projections of the 3 features we have. 
# Plot the class label (0/1) as color. 0 indicates missing values
yhat = sigmoid(np.dot(features,w)+b)
yhat_predict = yhat>0.5
for i in range(0,3):
    for j in range(0,i+1):
        if i != j:
            fig, axes = plt.subplots(1,2,sharex=True,sharey=True)
            plt.subplots_adjust(wspace=0)
            ax = axes[0]
            ax.set_title("Predicted")
            im = ax.scatter(features[:,i],features[:,j],c=yhat_predict[:,0])
            ax.set_xlabel(flabels[i])
            ax.set_ylabel(flabels[j])

            ax = axes[1]
            ax.set_title("Actual")
            im = ax.scatter(features[:,i],features[:,j],c=labels[:,0])
            plt.xlabel(flabels[i])
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Class')
            plt.savefig(f'./plots/q6_{i}{j}.png')
            plt.close()