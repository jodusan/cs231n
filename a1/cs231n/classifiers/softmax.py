import numpy as np
from random import shuffle


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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in xrange(num_train):
        scores = X[i].dot(W)
        logc = -np.max(scores)
        correct_class_score = scores[y[i]]
        sum_scores = 0
        sum_scores_grad = 0
        for j in xrange(num_classes):
            sum_scores += np.exp(scores[j] + logc)
            sum_scores_grad += np.exp(scores[j] + logc)

        for j in xrange(num_classes):
            if j == y[i]:
                dW[:, j] += - X[i] * (1 - np.exp(scores[j] + logc) / sum_scores_grad)
            else:
                dW[:, j] += - X[i] * (-np.exp(scores[j] + logc) / sum_scores_grad)

        loss += -correct_class_score - logc + np.log(sum_scores)

    loss /= num_train

    # Regularize
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train

    # Derivative of regularization above
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    sel_rows = range(num_train)
    scores = X.dot(W)

    # We subtract the max val for each sample because of numeric stability
    logc_vec = -np.max(scores, axis=1)

    y_vals_vec = np.vstack(scores[sel_rows, y])

    # Subtract the max val
    scores += np.vstack(logc_vec)

    scores1 = np.sum(np.exp(scores), axis=1)
    scores2 = np.log(scores1)
    scores2 = -np.vstack(logc_vec) - y_vals_vec + np.vstack(scores2)

    # Normalize
    loss = np.sum(scores2) / num_train

    # Regularize
    loss += 0.5 * reg * np.sum(W * W)

    # Gradient #
    # Divide all by their appropriate row-sum
    p = - np.exp(scores) / np.tile(scores1, (num_classes, 1)).T

    # For all correct classes add one
    p[sel_rows, y] += 1

    dW = -X.T.dot(p)

    # Normalize
    dW /= num_train

    # Derivative of regularization
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

