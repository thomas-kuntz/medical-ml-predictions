import numpy as np
from helpers import batch_iter
from utilities import loss_mse, gradient_mse
from utilities import gradient_logistic, loss_logistic


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent with a MSE cost function.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape (D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: numpy array of shape (D, ), optimal weight vector at the end of GD.
        loss: a scalar denoting the mse loss value with the optimal weight vector w.
    """

    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * gradient_mse(y, tx, w)
    return w, loss_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent with a MSE cost function.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape (D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: numpy array of shape (D, ), optimal weight vector at the end of GD.
        loss: a scalar denoting the mse loss value with the optimal weight vector w.
    """
    BATCH_SIZE_SGD = 1

    w = initial_w
    for _ in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(
            y, tx, batch_size=BATCH_SIZE_SGD, num_batches=1, shuffle=True
        ):
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
        loss: a scalar denoting the mse loss value with the optimal weight vector w.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, loss_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar. #todo préciser ? (ils disent juste scalar dans les exos)

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: a scalar denoting the mse loss value with the optimal weight vector w. # TODO check ed mse vs rmse


    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    n, d = tx.shape
    w = np.linalg.solve(tx.T @ tx + 2 * n * lambda_ * np.identity(d), tx.T @ y)

    return w, loss_mse(y, tx, w)  # TODO check ed mse vs rmse


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent (y ∈ {0, 1})
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape (D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: numpy array of shape (D, ), optimal weight vector at the end of GD.
        loss: a scalar denoting the logistic loss value with the optimal weight vector w.
    """
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * gradient_logistic(y, tx, w)
    return w, loss_logistic(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y ∈ {0, 1}, assuming that the regularization term of the loss is λ * ∥w∥^2)

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar. #todo préciser ? (ils disent juste scalar dans les exos)
        initial_w: numpy array of shape (D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: numpy array of shape (D, ), optimal weight vector at the end of GD.
        loss: a scalar denoting the logistic loss value with the optimal weight vector w.
    """

    w = initial_w
    n = len(tx)
    for _ in range(max_iters):
        gradient = gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    return w, loss_logistic(y, tx, w)
