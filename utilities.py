import numpy as np


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


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar

    Returns:
        image: scalar corresponding to sigmoid(t)

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    return np.exp(t) / (1 + np.exp(t))


def loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood.
    Assuming y values in {0, 1}. Our data has y values in {-1, 1} but the
    tests assume {0, 1}, so we transform our data to {0, 1} and then back
    to {-1, 1} when training the model and making predictions

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        loss: the logistic loss given y, tx and w.

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(loss_logistic(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]  # todo useful ?
    assert tx.shape[1] == w.shape[0]

    # Using the formula from lab05 of the negative log likelihood loss
    sigm = sigmoid(tx @ w)
    return -1 * np.mean(y * np.log(sigm) + (1 - y) * np.log(1 - sigm))


def gradient_logistic(y, tx, w):
    """compute the gradient of the logistic loss.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        gradient: An numpy array of shape (D, ) containing the gradient of the logistic loss at w.

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> gradient_logistic(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    return (1 / len(tx)) * tx.T @ (sigmoid(tx @ w) - y)


def compute_f1_score(y, preds, threshold=0.5):
    """
    todo proper doc

    """
    preds = (preds >= threshold).astype(int)

    TP = np.sum((y == 1) & (preds == 1))
    FP = np.sum((y == 0) & (preds == 1))
    FN = np.sum((y == 1) & (preds == 0))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return f1
