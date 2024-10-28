from utilities.logistic_regression import compute_f1_score

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

def compute_score_linear(y, x, w):
    preds = (x @ w > 0.5).astype(int)
    f1 = compute_f1_score(y, preds)
    accuracy = (preds == y).mean()
    loss = loss_mse(y, x, w)
    return dict(f1=f1, accuracy=accuracy, loss=loss)

