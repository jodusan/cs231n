import numpy as np
from random import shuffle


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        cnt = 0
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                cnt += 1
                loss += margin
                dW[:, j] += X[i]
        dW[:, y[i]] += -cnt * X[i]

    # We do this so it doesn't interfere with learning rate.
    # This will make step size always the same.
    dW /= num_train

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################



    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    delta = 1
    num_samples = X.shape[0]

    # Select all rows
    sel_rows = range(num_samples)

    scores = X.dot(W)

    # Subtract from all scores correct ones
    scores -= np.vstack(scores[sel_rows, y])

    # Add the delta margin
    scores += delta

    # Mark those within margin
    scoresMax = np.maximum(scores, 0)

    # Sum over all scores and subtract
    # -delta For the case when j=yi we included earlier and sum should skip it
    # Note: When we subtracted from all scores correct one, we also subtracted
    # correct score from correct score which just added delta so we delete it now
    scoresMax = np.sum(scoresMax, axis=1) - delta

    # Divide by num_samples so step size is normalized/always same/doesn't
    # interfere with learning rate

    loss = np.sum(scoresMax) / num_samples
    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    # All entries that are greater than zero should be clamped to one and all
    # other should be zero
    zeroOnesFixed = (scores > 0) + 0

    # Since correct classes need to be excluded from addition we can set them to
    # 0
    zeroOnesFixed[sel_rows, y] = 0

    # We need to know how much
    zeroSums = np.sum(zeroOnesFixed, axis=1)

    # Calculate all updates for all incorrect classes for X
    dW = X.T.dot(zeroOnesFixed)

    # Reset all to zero since we only need number of _classes within margin_
    zeroOnesFixed = np.zeros(zeroOnesFixed.shape)

    # in the column of the correct class.
    zeroOnesFixed[sel_rows, y] = zeroSums

    # Calculate and subtract updates for correct classes for X
    dW -= X.T.dot(zeroOnesFixed)

    # Normalize the gradient "step" size so it doesnt interfere with learning
    # rate.
    dW /= num_samples
    dW += reg * W

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
