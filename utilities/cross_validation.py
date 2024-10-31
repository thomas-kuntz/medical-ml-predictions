import numpy as np
from implementations import reg_logistic_regression, logistic_regression, ridge_regression, mean_squared_error_gd, least_squares
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

def k_th_cross_validation(y, x, k_indices, k, kind, **kwargs):
    """Perform one fold of cross validation and compute both training and testing losses.

    Args:
        y:          shape=(N,)
        x:          shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        kind:       string, either 'logistic', 'linear' or 'least_squares'
        **kwargs:   additional parameters for the specific kind. See below.
                    


    Returns:
        train and test losses for each of max iters
    
    List of possible kwargs:
        lambda_:    scalar, regularization coefficient. If used with kind='logistic'
                    or kind='least_squares', the regression will be regularized. Will be ignored otherwise.
        max_its:    list of int, indices of the iteration numbers at which to compute test scores. Required for
                    kind='logistic' and kind='linear'. Will be ignored otherwise.
        initial_w:  initial weights. Required for kind='linear' or kind='logistic', ignored otherwise.
        gamma:      learning rate for regression. Required for kind='linear' or kind='logistic', ignored otherwise.
        
        
    """
    if kind not in ['logistic', 'linear', 'least_squares']:
        raise ValueError("Invalid kind, must be one of 'logistic', 'linear', or 'least_squares'.")

    # Split data
    te_mask = k_indices[k]
    x_te, y_te = x[te_mask], y[te_mask]
    x_tr, y_tr = x[~te_mask], y[~te_mask]

    losses_tr = []
    scores_te = dict(f1=[], accuracy=[], loss=[])


    # Handle least squares separately
    if kind == 'least_squares':
        if 'lambda_' in kwargs:
            w, loss_tr = ridge_regression(y_tr, x_tr, kwargs['lambda_'])
        else:
            w, loss_tr = least_squares(y_tr, x_tr)
        losses_tr += [loss_tr]
        score = compute_score_linear(y_te, x_te, w)
        for score_name in scores_te.keys():
            scores_te[score_name] += [score[score_name]]
        return losses_tr, scores_te

    w = kwargs['initial_w']
    max_its = kwargs['max_its']
    # Train
    for i in range(len(max_its)):
        # Train starting with the previous weights
        # Use delta between current max_it and previous max_it, except for the first iteration
        max_it = max_its[i]
        if i > 0:
            max_it -= max_its[i - 1]
        if kind == 'logistic':
            if 'lambda_' in kwargs:
                w, loss_tr = reg_logistic_regression(y_tr, x_tr, kwargs['lambda_'], w, max_it, kwargs['gamma'])
            else:
                w, loss_tr = logistic_regression(y_tr, x_tr, w, max_it, kwargs['gamma'])
            score = compute_score_logistic(y_te, x_te, w)
        if kind == 'linear':
            w, loss_tr = mean_squared_error_gd(y_tr, x_tr, w, max_it, kwargs['gamma'])
            score = compute_score_linear(y_te, x_te, w)
        losses_tr += [loss_tr]
        for score_name in scores_te.keys():
            scores_te[score_name] += [score[score_name]]

    return losses_tr, scores_te

def cross_validation(y, x, kind, k_indices, fold_nums, verbose=False, **kwargs):
    # TODO proper docs
    """
    """
    if kind not in ['logistic', 'linear', 'least_squares']:
        raise ValueError("Invalid kind, must be one of 'logistic', 'linear', or 'least_squares'.")
    if verbose:
        print("\nCross validating for {:20s}".format(kind) + "\t".join([f"{k}={v}" for k, v in kwargs.items() if k not in ['initial_w', 'max_its']]))
    if kind == 'least_squares':
        max_its = [0]
    else:
        if 'max_its' in kwargs:
            max_its = kwargs['max_its']
        else:
            max_its = [100]
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
            y, x, k_indices, k, kind=kind, **kwargs
        )
        for score_name in scores.keys():
            scores[score_name] = [agg + v for agg, v in zip(scores[score_name], scores_te[score_name])]
    # Average each score across the k folds (but keep separate values for each max_it)
    for score_name in scores.keys():
        scores[score_name] = [val / fold_nums for val in scores[score_name]]
    # We want to avoid models that overfit, so, as a simple heuristic, we will take the last f1 score
    f1_score = scores['f1'][-1]
    if verbose:
        print("\n\tF1 score{}: {:.6f}".format("" if kind=='least_squares' else f" after {max_its[-1]} iterations", f1_score))
    return scores, f1_score


def cross_validation_for_parameter_selection(y, x, kind, fold_nums, verbose=False, **kwargs):
    """Perform cross validation for regularized logistic regression.
    Args:
        y:          shape=(N,)
        x:          shape=(N,D)
        kind:       string, one of 'logistic', 'linear' or 'least_squares'
        fold_nums:  int, number of folds to perform
        verbose:    bool, whether to print progress reports or not 
        **kwargs:   parameters or hyperparameters to pass to the training function. See below.


    Returns:
        best_f1:    scalar, best f1 score for all combinations of hyperparameters
        best_hps:   dict, keys are the hyperparameter names, values are those that give the best f1 score
        scores:     list of dicts if there's one kind of hyperparameter, list of list of dicts if there's two,
                    in which case the rows correspond to a fixed value of lambda, and the columns to a fixed
                    value of gamma. Each entry contains a dict. The keys of the dictionary are the three kind of
                    scores (f1, accuracy, loss), and the values are lists of losses, one for each value of
                    max_its. If kind='least_squares' there are no iters, so said lists are of size 1.                 
    
    kwargs for kind='logistic' 
        lambdas:    (optional) list of values of lambda to cross-validate. If provided, regression will be regularized.
        initial_w:  shape=(D,) initial weights, required
        max_its:    list of values of max_iters to cross-validate, required
        gammas:     list of values of gamma to cross-validate, required

    kwargs for kind='linear'
        initial_w:  shape=(D,) initial weights, required
        max_its:    list of values of max_iters to cross-validate, required
        gammas:     list of values of gamma to cross-validate, required
        
    kwargs for kind='least_squares' 
        lambdas:    (optional) list of values of lambda to cross-validate. If provided, regression will be regularized.
    """
    if kind not in ['logistic', 'linear', 'least_squares']:
        raise ValueError("Invalid kind, must be one of 'logistic', 'linear', or 'least_squares'.")

    k_indices = build_k_indices(y, fold_nums)
    
    best_f1 = 0
    best_hps = None

    # Create score boards for each combination of hyperparameters
    # If there are no hyperparameters, we use a list for consistency
    all_scores = [None]
    if 'lambdas' in kwargs and 'gammas' in kwargs:
        all_scores = [[None for _ in kwargs['gammas']] for _ in kwargs['lambdas']]
    else:
        if 'lambdas' in kwargs:
            all_scores = [None for _ in kwargs['lambdas']]
        if 'gammas' in kwargs:
            all_scores = [None for _ in kwargs['gammas']]

    ##===========================##
    #   LEAST SQUARES
    ##===========================##
    if kind == 'least_squares':
        if 'lambdas' in kwargs:
            for ilambda_, lambda_ in enumerate(kwargs['lambdas']):
                params = dict(lambda_=lambda_)
                scores, f1_score = cross_validation(y, x, kind, k_indices, fold_nums, verbose, **params)

                # Keep track of the best f1 score
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_hps = dict(lambda_=lambda_)

                # Keep track of all scores for each value of lambda
                all_scores[ilambda_] = scores
        else:
            scores, f1_score = cross_validation(y, x, kind, k_indices, fold_nums, verbose)

            # Keep track of the best f1 score
            if f1_score > best_f1:
                best_f1 = f1_score

            # Keep track of all scores
            all_scores = [scores]
    
    ##===========================##
    #   LINEAR REGRESSION
    ##===========================##
    if kind == 'linear':
        if any([k not in kwargs for k in ['initial_w', 'max_its', 'gammas']]):
            raise ValueError("All of 'initial_w', 'max_its' and 'gammas' must be passed to this function when kind='linear'.")
        for igamma, gamma in enumerate(kwargs['gammas']):
            params=dict(initial_w=kwargs['initial_w'], max_its=kwargs['max_its'], gamma=gamma)
            scores, f1_score = cross_validation(y, x, kind, k_indices, fold_nums, verbose, **params)

            # Keep track of the best f1 score
            if f1_score > best_f1:
                best_f1 = f1_score
                best_hps = dict(gamma=gamma)

            # Keep track of all scores for each value of gamma
            all_scores[igamma] = scores

    ##===========================##
    #   LOGISTIC REGRESSION
    ##===========================##
    if kind == 'logistic':
        if any([k not in kwargs for k in ['initial_w', 'max_its', 'gammas']]):
            raise ValueError("All of 'initial_w', 'max_its' and 'gammas' must be passed to this function when kind='logistic'")
        if 'lambdas' in kwargs:
            for ilambda_, lambda_ in enumerate(kwargs['lambdas']):
                for igamma, gamma in enumerate(kwargs['gammas']):
                    params=dict(lambda_=lambda_, initial_w=kwargs['initial_w'], max_its=kwargs['max_its'], gamma=gamma)
                    scores, f1_score = cross_validation(y, x, kind, k_indices, fold_nums, verbose, **params)

                    # Keep track of the best f1 score
                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_hps = dict(lambda_=lambda_, gamma=gamma)

                    # Keep track of all scores for each value of gamma and lambda
                    all_scores[ilambda_][igamma] = scores
        else:
            for igamma, gamma in enumerate(kwargs['gammas']):
                params=dict(initial_w=kwargs['initial_w'], max_its=kwargs['max_its'], gamma=gamma)
                scores, f1_score = cross_validation(y, x, kind, k_indices, fold_nums, verbose, **params)

                # Keep track of the best f1 score
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_hps = dict(gamma=gamma)

                # Keep track of all scores for each value of gamma
                all_scores[igamma] = scores

    return best_f1, best_hps, all_scores

def cross_validation_for_model_selection(y, x, fold_nums, models, verbose=False):
    # TODO proper docs
    scores_by_model = {}
    overall_best_f1 = 0
    best_model = None
    for model in models:
        if verbose:
            print(f"Performing cross validation for {model['name']}")
        best_f1, best_hps, all_scores = cross_validation_for_parameter_selection(y, x, kind=model['kind'], fold_nums=fold_nums, verbose=verbose, **model['parameters'])
        scores_by_model[model['name']] = (best_f1, best_hps, all_scores)

        if best_f1 > overall_best_f1:
            overall_best_f1 = best_f1
            best_model = model['name']
    return overall_best_f1, best_model, scores_by_model