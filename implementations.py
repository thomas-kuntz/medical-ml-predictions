import numpy as np
from helpers import batch_iter


def loss_mse(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    return 1/(2 * len(y)) * e.T @ e


def gradient_mse(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx @ w
    return -(1/len(y)) * tx.T @ e


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for i in range(max_iters):
        w = w - gamma * gradient_mse(y, tx, w)
    return w, loss_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    w = initial_w
    # TODO which batch size to pick
    batch_size = 42
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
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


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    return 1/(1+np.exp(-t))


def loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(loss_logistic(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    sigm = sigmoid(tx @ w)
    log_l = y * np.log(sigm) + (1 - y) * np.log(1 - sigm)
    return -1 * log_l.mean()


def gradient_logistic(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> gradient_logistic(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    return (1/len(tx)) * tx.T @ (sigmoid(tx @ w) - y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y ∈ {0, 1})"""
    w = initial_w
    for i in range(max_iters):
        w = w - gamma * gradient_logistic(y, tx, w)
    return w, loss_logistic(y, tx, w)


def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y ∈ {0, 1}, with regularization term λ∥w∥^2 )"""
    w = initial_w
    n = len(tx)
    for i in range(max_iters):
        gradient = gradient_logistic(y, tx, w) * n * lambda_ * w
        w = w - gamma * gradient
    return w, loss_logistic(y, tx, w)
