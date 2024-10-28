import numpy as np
from itertools import product
from implementations import reg_logistic_regression, loss_logistic

def build_k_indices(y, k_fold, seed=0):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def k_th_cross_validation(y, x, k_indices, k, train_fn, loss_fn, *args):
    """Perform one fold of cross validation and compute both training and testing losses.

    Args:
        y:          shape=(N,)
        x:          shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        train_fn:   training function, first two positional parameters must be y (training output) and x (training input)
        loss_fn:    loss function, parameters must be y (test output), x (test input) and w (training weights)
        *args:      any additional parameters to be passed to the training function


    Returns:
        train and test losses
    """

    # Split data
    te_mask = k_indices[k]
    x_te, y_te = x[te_mask], y[te_mask]
    x_tr, y_tr = x[~te_mask], y[~te_mask]

    # Train
    weights, loss_tr = train_fn(y_tr, x_tr, *args)

    # Compute test loss
    loss_te = loss_fn(y_te, x_te, weights)

    return loss_tr, loss_te

def cross_validation_reg_logistic_regression(y, x, lambdas, initial_w, max_its, gammas, fold_nums, verbose=False):
    """Perform cross validation for regularized logistic regression.
    Args:
        y:          shape=(N,)
        x:          shape=(N,D)
        lambdas:    list of values of lambda to cross-validate
        initial_w:  shape=(D,) initial weights
        max_its:    list of values of max_iters to cross-validate
        gammas:     list of values of gamma to cross-validate
        fold_nums:  int, number of folds to perform
        verbose:    bool, whether to print progress reports or not 


    Returns:
        train and test losses
    """
    k_indices = build_k_indices(y, fold_nums)

    # Keep track of the best loss and the best hyperparameters
    best_loss = np.inf
    best_hps = dict(lambda_=None, max_it=None, gamma=None)

    # Keep track of all losses
    losses = [[[None for _ in gammas] for _ in max_its] for _ in lambdas]
    losses_per_hp = dict(
        lambda_={l: [] for l in lambdas},
        max_it={m: [] for m in max_its},
        gamma={g: [] for g in gammas}
    )

    # Consider each combination of hyperparameters
    for ilambda_, lambda_ in enumerate(lambdas):
        for imax_it, max_it in enumerate(max_its):
            for igamma, gamma in enumerate(gammas):
                if verbose:
                    print(f"\nComputing cross validation for lambda={lambda_}\tmax_it={max_it}\tgamma={gamma}")

                # Cross validate the loss
                loss = 0
                for k in range(fold_nums):
                    if verbose:
                        print(f"k={k}", end="\t")
                    loss_tr, loss_te = k_th_cross_validation(y, x, k_indices, k, reg_logistic_regression, loss_logistic, lambda_, initial_w, max_it, gamma)
                    loss += loss_te
                loss /= fold_nums
                if verbose:
                    print("\n\tAverage Loss = {:.6f}".format(loss))

                if loss < best_loss:
                    best_loss = loss
                    best_hps = dict(lambda_=lambda_, max_it=max_it, gamma=gamma)

                loss[ilambda_][imax_it][igamma] = loss
                losses_per_hp['lambda_'][lambda_] += [loss]
                losses_per_hp['max_it'][max_it] += [loss]
                losses_per_hp['gamma'][gamma] += [loss]

    return best_loss, best_hps, losses_per_hp, losses