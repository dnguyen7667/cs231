from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # num train and dim
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    
    # for each sample"
    for i in range(num_train):
        # scores = X dot W
        score = X[i].dot(W)

        #exponentiate the scores
        score = np.exp(score)

        #score -= score/np.max(score)

        #get correct class score
        correct_class_score = score[y[i]]

        # sum of all score
        sum_score = np.sum(score)

        #loss
        loss += -np.log(correct_class_score/sum_score)

        # get the dW 
        # WRONGGGG LMAO IM STUPID
        # Li = -log(e^s_yi/ sum(e^s_j)) with j is from 1 to num_classes
        # if we work out the loss, it will be Li = sum(s_j) for j != yi
        # Or Li = sum(X_i dot W_j) for j != y[i]
        # thus dW[:, j] = X_i for j != yi
        for j in range(num_classes):
            if (j == y[i]): 
                dW[:, j] -= X[i]

            dW[:, j] += X[i]*score[j]/sum_score


    # average the loss
    loss /= num_train

    # add the regularization
    loss += reg*np.sum(W*W)


    # also divide dW by num_class as well
    dW /= num_train

    # also the regularization
    dW += 2*reg*W



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # number of training samples
    num_train = X.shape[0]

    # number of classes
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Scores are X dot T
    scores = X.dot(W)

    # exponentiate the scores
    scores = np.exp(scores)

    # get sumscores of each sample
    # reshape to get division by row
    sum_scores = np.sum(scores, axis = 1).reshape(-1, 1)

    # correct class - WRONG because of broadcasting
    # correct_class_score = scores[np.arange(len(scores)), y]

    #get matrix proba = scores/sum_scores of each score of each sample
    proba = scores/sum_scores

    #vector proba_correct_class = correct_class_score/sum_scores of score of each correct classs of each sample
    proba_correct_class = proba[np.arange(len(scores)), y]

    # loss = sum of all proba_correct_class

    loss += np.sum(-np.log(proba_correct_class))

    # Calculate dW
    dW = X.T.dot(proba)

    # also, for each correct class in a sample i , subtract Xi from it
    acti = np.zeros_like(scores)
    acti[np.arange(len(scores)), y] = -1

    added_dW = X.T.dot(acti)

    dW = dW + added_dW

    # average the loss
    loss /= num_train

    # add the regularization
    loss += reg*np.sum(W*W)


    # also divide dW by num_class as well
    dW /= num_train

    # also the regularization
    dW += 2*reg*W





    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
