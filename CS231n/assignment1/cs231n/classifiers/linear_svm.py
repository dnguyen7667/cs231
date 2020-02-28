from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0


    # forward pass
    for i in range(num_train): 
        # for each sample
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] -= X[i]
                dW[:, j] += X[i]
                             
      

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)


    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW /= num_train # since lose is /num_train, dW is /num_train as well
   
    dW +=2*reg*W # as regularization is added



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    num_classes = W.shape[1]
    num_train  = X.shape[0]
  

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W) # scores matrix of dim num_train by num_classes - 
                        # each row is a score of each train example,
                        # with each col/element is score of that class


    # vector of scores of true classes
    correct_class_scores = scores[np.arange(len(scores)), y] 

    # get the margins 
    margins = scores - correct_class_scores[:, None] + 1 

    # reset the correct class scores as they're supposed to be untouched
    margins[np.arange(len(margins)), y] = correct_class_scores 


    # make a copy of margins
    margins_sub = margins.copy() 

    # set values of <0 at position of correct class
    # so that it won't be added to the loss
    margins_sub[np.arange(len(margins_sub)), y] = -99999 

    # los = sum of margins where margins > 0 
    loss = np.sum(margins_sub[margins_sub > 0]) 


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

   

    # "activate" the columns (each column = each sample) that will be added if it has loss > 0
    act = np.where(margins_sub > 0, 1, 0)

    # count number of time we have to subtract correct_class_scores 
    act_count = np.sum(act, axis = 1) 

    # make a vector act_count_vect to store act_count at
    # the correct position on a margins matrix
    act_count_vect = np.zeros(act.shape)
    act_count_vect[np.arange(len(act_count)), y] = -act_count



    # sum up the samples that caused the loss as dL/dW = X
    dW_ = X.T.dot(act)
    dW_sub = X.T.dot(act_count_vect)

    # also substract samples that causes loss as dL/dW(of correct class) = -X
    # dW = X.T.dot(act)

    dW = dW_ + dW_sub
    # since lose is /num_train, dW is /num_train as well
    dW /= num_train 
   
    # as regularization is added
    dW +=2*reg*W 


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
