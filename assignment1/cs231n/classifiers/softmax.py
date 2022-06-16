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
    scores_exp = np.exp(scores)
    softmax_score = scores_exp/np.sum(scores_exp)

    loss += -np.log(softmax_score[y[i]])

    for j in xrange(num_classes):
      if j == y[i]:
        dW[:,j] += (softmax_score[j]-1)*X[i,:]
      else:
        dW[:,j] += (softmax_score[j])*X[i,:]

  
  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += reg*W

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
  dW = np.zeros_like(W) ## DxC

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W) # NxC
  scores_exp = np.exp(scores)
  softmax_score = scores_exp/np.expand_dims(np.sum(scores_exp, axis=1), axis=1) ## NxC
  loss = -np.sum(np.log(softmax_score[np.arange(num_train), y]))
  loss /= num_train
  loss += reg * np.sum(W * W)

  
  
  updated = softmax_score
  updated[np.arange(num_train), y] -= 1
  dW = np.matmul(X.T, updated)
  dW /= num_train
  dW += reg*W
  # raise NotImplementedError

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

