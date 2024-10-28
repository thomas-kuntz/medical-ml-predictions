import numpy as np

def gradient_mse(y, tx, w):
    """
    Computes the gradient at w.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        gradient: An numpy array of shape (D, ) containing the gradient of the mse loss at w.
    """
    e = y - tx @ w
    return -(1 / len(y)) * tx.T @ e

def loss_mse(y, tx, w):
    """Calculate the loss using the MSE cost function.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        loss: a scalar corresponding to the MSE loss.
    """
    e = y - tx @ w
    return 1 / (2 * len(y)) * e.T @ e
