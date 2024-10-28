import numpy as np
from implementations import reg_logistic_regression, logistic_regression, ridge_regression, mean_squared_error_gd
from utilities.logistic_regression import compute_score_logistic
from utilities.linear_regression import compute_score_linear

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

def k_th_cross_validation(y, x, k_indices, k, kind, **kwargs): #lambda_, initial_w, max_its, gamma):
    # TODO document properly
    """Perform one fold of cross validation and compute both training and testing losses.

    Args:
        y:          shape=(N,)
        x:          shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        train_fn:   training function, first two positional parameters must be y (training output) and x (training input)
        score_fn:   score function, parameters must be y (test output), x (test input) and w (training weights)
        *args:      any additional parameters to be passed to the training function


    Returns:
        train and test losses for each of max iters
    """
    if kind not in ['reg_logistic_regression', 'logistic_regression', 'ridge_regression', 'linear_regression']:
        raise ValueError("Invalid kind")

    # Split data
    te_mask = k_indices[k]
    x_te, y_te = x[te_mask], y[te_mask]
    x_tr, y_tr = x[~te_mask], y[~te_mask]

    losses_tr = []
    scores_te = dict(f1=[], accuracy=[], loss=[])

    w = kwargs['initial_w']
    max_its = kwargs['max_its']

    # Train
    for i in range(len(max_its)):
        # Train starting with the previous weight
        # Use delta between current max_it and previous max_it, except for the first iteration
        max_it = max_its[i]
        if i > 0:
            max_it -= max_its[i - 1]
        if kind == 'reg_logistic_regression':
            w, loss_tr = reg_logistic_regression(y_tr, x_tr, kwargs['lambda_'], w, max_it, kwargs['gamma'])
        if kind == 'logistic_regression':
            w, loss_tr = logistic_regression(y_tr, x_tr, w, max_it, kwargs['gamma'])
        if kind == 'ridge_regression':
            w, loss_tr = ridge_regression(y_tr, x_tr, kwargs['lambda_'], w, max_it, kwargs['gamma'])
        if kind == 'linear_regression':
            w, loss_tr = mean_squared_error_gd(y_tr, x_tr, w, max_it, kwargs['gamma'])
        losses_tr += [loss_tr]

        # Compute test scores
        if kind in ['reg_logistic_regression','logistic_regression']:
            score = compute_score_logistic(y_te, x_te, w)
        if kind in ['ridge_regression', 'linear_regression']:
            score = compute_score_linear(y_te, x_te, w)
        for score_name in scores_te.keys():
            scores_te[score_name] += [score[score_name]]

    return losses_tr, scores_te

def cross_validation_reg_logistic_regression(y, x, lambdas, initial_w, max_its, gammas, fold_nums, verbose=False):
    # TODO proper documentation
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

    # Keep track of the best f1 score and the best hyperparameters for f1 score
    best_f1 = 0
    best_hps = dict(lambda_=None, max_it=None, gamma=None)

    # Keep track of all scores
    all_scores = [[None for _ in gammas] for _ in lambdas]

    # Consider each combination of hyperparameters
    for ilambda_, lambda_ in enumerate(lambdas):
        for igamma, gamma in enumerate(gammas):
            if verbose:
                print(f"\nComputing cross validation for\tlambda={lambda_}\tgamma={gamma}")

            # Compute the sum of the scores across the k folds
            scores = dict(
                f1=[0 for _ in max_its],
                accuracy=[0 for _ in max_its],
                loss=[0 for _ in max_its]
            )
            for k in range(fold_nums):
                if verbose:
                    print(f"k={k}", end="\t")
                # scores_te here is a dictionary of lists
                # keys are the name of the score (f1, accuracy, etc)
                # values are a list of the values said score takes during training (polled at each value of max_its)
                losses_tr, scores_te = k_th_cross_validation(
                    y, x, k_indices, k, 'reg_logistic_regression', lambda_=lambda_, initial_w=initial_w, max_its=max_its, gamma=gamma
                )
                for score_name in scores.keys():
                    scores[score_name] = [agg + v for agg, v in zip(scores[score_name], scores_te[score_name])]
            # Average each score across the k folds
            for score_name in scores.keys():
                scores[score_name] = [val / fold_nums for val in scores[score_name]]
            # We want to avoid models that overfit, so, as a simple heuristic, we will take the last f1 score
            local_f1_score = scores['f1'][-1]
            if verbose:
                print("\n\tF1 score after {} iterations: {:.6f}".format(max_its[-1], local_f1_score))

            # Keep track of the best f1 score
            if local_f1_score > best_f1:
                best_f1 = local_f1_score
                best_hps = dict(lambda_=lambda_, gamma=gamma)

            # Keep track of all scores for each value of each hyperparameter
            all_scores[ilambda_][igamma] = scores

    return best_f1, best_hps, all_scores