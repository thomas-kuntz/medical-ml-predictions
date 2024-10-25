import numpy as np
from helpers import batch_iter
from utilities import loss_mse, gradient_mse
from utilities import gradient_logistic, loss_logistic

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * gradient_mse(y, tx, w)
    return w, loss_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    w = initial_w
    for _ in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
            w = w - gamma * gradient_mse(minibatch_y, minibatch_tx, w)
    return w, loss_mse(y, tx, w)


def least_squares(y, tx):
    """Calculate the least squares solution using the normal equations.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, loss_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    n, d = tx.shape
    w = np.linalg.solve(
        tx.T @ tx + 2 * n * lambda_ * np.identity(d),
        tx.T @ y
    )
    # TODO if there’s a problem here, check that we return the correct loss
    return w, loss_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y ∈ {0, 1})"""
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * gradient_logistic(y, tx, w)
    return w, loss_logistic(y, tx, w)


def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y ∈ {0, 1}, with regularization term λ/2 * ∥w∥^2 )"""
    w = initial_w
    n = len(tx)
    for _ in range(max_iters):
        gradient = gradient_logistic(y, tx, w) + n * lambda_ * w
        w = w - gamma * gradient
    return w, loss_logistic(y, tx, w)
