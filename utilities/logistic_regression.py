import numpy as np
from helpers import batch_iter

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

def compute_f1_score(y, preds):
    """Computes the f1 score for the given labels and predictions
    Args:
        y:      shape=(N,) the correct labels in {0, 1}
        preds:  shape=(N,) the predicted labels in {0, 1}

    Returns:
        f1:     scalar in [0; 1], the f1 score
    """

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

def compute_score_logistic(y, x, w):
    """Compute test scores in the context of logistic regression
    Args:
        y:      shape=(N,) the correct labels
        x:      shape=(N, D) the model inputs
        w:      shape=(D,) the optimal weights found through logistic regression
    
    Returns:
        scores: dict, keys are score names (f1, accuracy, loss), values are the values of said scores 
    """
    preds = np.round(sigmoid(x @ w))
    f1 = compute_f1_score(y, preds)
    accuracy = (preds == y).mean()
    loss = loss_logistic(y, x, w)
    scores = dict(f1=f1, accuracy=accuracy, loss=loss) 
    return scores


def logistic_regression_sgd(y, tx, initial_w, max_iters, gamma, balanced=False, batch_size=64):
    """
    Logistic regression using stochastic gradient descent (y ∈ {0, 1})
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
    N, D = tx.shape
    w = initial_w
    for _ in range(max_iters):
        idxs = np.random.randint(low=0, high=N+1, size=batch_size)
        for minibatch_y, minibatch_tx in batch_iter(
            y, tx, batch_size=batch_size, num_batches=1, shuffle=True
        ):
            w = w - gamma * gradient_logistic(minibatch_y, minibatch_tx, w)
    return w, loss_logistic(y, tx, w)