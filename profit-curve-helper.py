def one_d_confusion_matrix(y_hat, y_true):
    """
    Calculates and returns a one-dimensional confusion matrix

    Input:
    y_hat = np.array of length m of predicted labels (dtype = bool or int)
    y_true = np.array of length m of true labels (dtype = bool or int)

    Output:
    np.array of shape (4,) in form (true positive, false negative, false positive, true negative)

    Pro-tip:
    Use one_d_confusion_matrix(y_hat, y_true).reshape(2,2) to display as a 2x2 array as follows:
    +----+----+
    | tp | fn |
    +----+----+
    | fp | tn |
    +----+----+
    """
    tp = np.sum(y_hat & y_true)
    fn = np.sum(~y_hat & y_true)
    fp = np.sum(y_hat & ~y_true)
    tn = y_hat.shape[0] - tp - fp - fn
    return np.array([tp, fn, fp, tn])

def confusion_matrices(bootstraps, X_test, y_test, n_thresholds=250):
    """
    Calculates confusion matrices for a set bootstraped models using
    test data.

    Input:
    bootstraps: a list of models of
    """
    yhats = np.array([model.predict_proba(X_test) for model in bootstraps])
    yhats_pos_class = yhats[:,:,1]
    test_thresholds = np.linspace(0, 1, n_thresholds).reshape(-1,1,1)
    yhat_by_threshold = yhats_pos_class > test_thresholds
    confusion_matrices = (np.apply_along_axis(lambda x: one_d_confusion_matrix(x, y_test),
                                     2, yhat_by_threshold))
    return confusion_matrices

def bootstrap_revenues(confusion_matrices, revenue_matrix):
    """
    Returns estimated costs/profits/revenues for an array of confusion matrices.

    Intended for quantifying uncertainty

    Parameters:
    Input:
    confusion_matrices: np.array in shape [m,n,4] where the 3rd dimension
    is a confusion matrix in the form [tp, fn, fp, tn]

    cost_matrix: np.array in shape [2,2] or [4,] where the values represent
    the costs/profits associated of true positive, false negative,
    false positive, true negative in the form

    [tp, fn, fp, tn]

    or

    +----+----+
    | tp | fn |
    +----+----+
    | fp | tn |
    +----+----+

    Output:
    revenues_matrix: np.array in shape [m,n,4,1] where the 4th dimension
    is the calcuated cost
    """
    revenues = confusion_matrices.dot(revenue_matrix)
    fifth_percentile = np.percentile(revenues, 5, axis = 1)
    ninety_fifth_percentile = np.percentile(revenues, 95, axis = 1)
