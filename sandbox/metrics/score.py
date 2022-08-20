from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

# ToDo: 他のスコアリングの計算式を紐解き, 下記のエイリアスおよび関数に追加する.
# 参考: https://lightgbm.readthedocs.io/en/latest/Parameters.html
SCORING_ALIASES = {
    "mean_squared_error": [
        "regression",
        "regression_l2",
        "l2",
        "mean_squared_error",
        "mse",
    ],
    "root_mean_squared_error": ["l2_root", "root_mean_squared_error", "rmse"],
    "mean_absolute_error": ["regression_l1", "l1", "mean_absolute_error", "mae"],
    "mean_absolute_percentage_error": ["mape", "mean_absolute_percentage_error"],
    "r2": ["r2", "r2_score"],
}


def manage_scoring_alias(scoring):
    """Manage and convert the string expressing which score is used.

    Parameters
    ----------
    scoring : str
        The string expressing which score is used.

    Returns
    -------
    str
        Converted string.
    """
    standard_scoring = None
    for key, value in SCORING_ALIASES.items():
        if scoring in value:
            standard_scoring = key
            break
    return standard_scoring


def mean_absolute_error(
    y_true, y_pred, sample_weight=None, multioutput="uniform_average"
):
    """Mean absolute error regression loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.

        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute error is returned
        for each output separately.
        If multioutput is 'uniform_average' or a ndarray of weights, then the
        weighted average of all output errors is returned.
        MAE output is non-negative floating point. The best value is 0.0.

    Examples
    --------
    >>> from sandbox.metrics.score import mean_absolute_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_absolute_error(y_true, y_pred)
    0.5
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> mean_absolute_error(y_true, y_pred)
    0.75
    >>> mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    array([0.5, 1. ])
    >>> mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.85...
    """
    return mae(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )


def mean_absolute_percentage_error(
    y_true, y_pred, sample_weight=None, multioutput="uniform_average"
):
    """Mean absolute percentage error (MAPE) regression loss from ``scikit-learn``

    Note here that the output is not a percentage in the range [0, 100]
    and a value of 100 does not mean 100% but 1e2. Furthermore, the output
    can be arbitrarily high when `y_true` is small (which is specific to the
    metric) or when `abs(y_true - y_pred)` is large (which is common for most
    regression metrics).

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weight : {array-like}, optional
        Sample weights.
    multioutput : {str{'raw_values', 'uniform_average'}, array-like}
        Defines aggregating of multiple output values.

        Array-like value defines weights used to average errors.

        If input is list then the shape must be (n_outputs,).

        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : {float, ndarray[float]}
        If multioutput is 'raw_values', then mean absolute percentage error
        is returned for each output separately.

        If multioutput is 'uniform_average' or a ndarray of weights, then the
        weighted average of all output errors is returned.

        MAPE output is non-negative floating point. The best value is 0.0.
        But note that bad predictions can lead to arbitrarily large
        MAPE values, especially if some `y_true` values are very close to zero.

        Note that we return a large value instead of `inf` when `y_true` is zero.

    Examples
    --------
    >>> from sandbox.metrics.score import mean_absolute_percentage_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.3273...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.5515...
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.6198...
    >>> # the value when some element of the y_true is zero is arbitrarily high because
    >>> # of the division by epsilon
    >>> y_true = [1., 0., 2.4, 7.]
    >>> y_pred = [1.2, 0.1, 2.4, 8.]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    112589990684262.48
    """
    return mape(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )


def mape_score(y_true, y_pred, sample_weight=None, multioutput="uniform_average"):
    """1 - Mean absolute percentage error (MAPE) regression.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weight : array-like, optional
        Sample weights.
    multioutput : {str{'raw_values', 'uniform_average'}, -like}
        Defines aggregating of multiple output values.

        Array-like value defines weights used to average errors.

        If input is list then the shape must be (n_outputs,).

        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    z : {float, ndarray[float]}
        1 - `mean_absolute_percentage_error`

    """
    mape = mean_absolute_percentage_error(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )
    mape = 1 if 1 < mape or mape < 0 else mape
    return 1 - mape


def mean_squared_error(
    y_true, y_pred, sample_weight=None, multioutput="uniform_average", squared=True
):
    """Mean squared error regression loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    squared : bool, default=True
        If True returns MSE value, if False returns RMSE value.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.

    Examples
    --------
    >>> from sandbox.metrics.score import mean_squared_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred)
    0.375
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred, squared=False)
    0.612...
    >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
    >>> y_pred = [[0, 2],[-1, 2],[8, -5]]
    >>> mean_squared_error(y_true, y_pred)
    0.708...
    >>> mean_squared_error(y_true, y_pred, squared=False)
    0.822...
    >>> mean_squared_error(y_true, y_pred, multioutput='raw_values')
    array([0.41666667, 1.        ])
    >>> mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.825...
    """
    return mse(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
        squared=squared,
    )


def r2_score(
    y_true,
    y_pred,
    sample_weight=None,
    multioutput="uniform_average",
    force_finite=True,
):
    r""":math:`R^2` (coefficient of determination) regression score function from ``scikit-learn``

    Best possible score is 1.0, and it can be negative (because the
    model can be arbitrarily worse). In the general case when the true y is
    non-constant, a constant model that always predicts the average y
    disregarding the input features would get a :math:`R^2` score of 0.0.

    In the particular case when ``y_true`` is constant, the :math:`R^2` score
    is not finite: it is either ``NaN`` (perfect predictions) or ``-Inf``
    (imperfect predictions). To prevent such non-finite numbers to pollute
    higher-level experiments such as a grid search cross-validation, by default
    these cases are replaced with 1.0 (perfect predictions) or 0.0 (imperfect
    predictions) respectively. You can set ``force_finite`` to ``False`` to
    prevent this fix from happening.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weight : array-like, optional
        Sample weights.
    multioutput : {str{'raw_values', 'uniform_average', 'variance_weighted'}, array-like}, optional
        Defines aggregating of multiple output scores.

        Array-like value defines weights used to average scores.

        Default is "uniform_average".

        'raw_values' :
            Returns a full set of scores in case of multioutput input.
        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.
        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.
    force_finite : bool, optional
        Flag indicating if ``NaN`` and ``-Inf`` scores resulting from constant
        data should be replaced with real numbers (``1.0`` if prediction is
        perfect, ``0.0`` otherwise). Default is ``True``, a convenient setting
        for hyperparameters' search procedures (e.g. grid search
        cross-validation).

    Returns
    -------
    z : float or ndarray of floats
        The :math:`R^2` score or ndarray of scores if 'multioutput' is
        'raw_values'.

    Notes
    -----
    This is not a symmetric function.

    Unlike most other scores, :math:`R^2` score may be negative (it need not
    actually be the square of a quantity R).

    This metric is not well-defined for single samples and will return a NaN
    value if n_samples is less than two.

    Examples
    --------
    >>> from sklearn.metrics import r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)
    0.948...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> r2_score(y_true, y_pred,
    ...          multioutput='variance_weighted')
    0.938...
    >>> y_true = [1, 2, 3]
    >>> y_pred = [1, 2, 3]
    >>> r2_score(y_true, y_pred)
    1.0
    >>> y_true = [1, 2, 3]
    >>> y_pred = [2, 2, 2]
    >>> r2_score(y_true, y_pred)
    0.0
    >>> y_true = [1, 2, 3]
    >>> y_pred = [3, 2, 1]
    >>> r2_score(y_true, y_pred)
    -3.0
    >>> y_true = [-2, -2, -2]
    >>> y_pred = [-2, -2, -2]
    >>> r2_score(y_true, y_pred)
    1.0
    >>> r2_score(y_true, y_pred, force_finite=False)
    nan
    >>> y_true = [-2, -2, -2]
    >>> y_pred = [-2, -2, -2 + 1e-8]
    >>> r2_score(y_true, y_pred)
    0.0
    >>> r2_score(y_true, y_pred, force_finite=False)
    -inf
    """
    return r2(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
        force_finite=force_finite,
    )


def score(y_true, y_pred, scoring="r2", **score_kwargs):
    """Score function.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    scoring : str
        The string expressing which score is used.
    score_kwargs : dict
        Keyword arguments for internal score function.

    Returns
    -------
    score : float or ndarray of floats
        The result of score function.
    """
    scoring = manage_scoring_alias(scoring)
    if scoring == "mean_squared_error":
        score_val = mean_squared_error(y_true, y_pred, squared=True, **score_kwargs)
    elif scoring == "root_mean_squared_error":
        score_val = mean_squared_error(y_true, y_pred, squared=False, **score_kwargs)
    elif scoring == "mean_absolute_error":
        score_val = mean_absolute_error(y_true, y_pred, **score_kwargs)
    elif scoring == "mean_absolute_percentage_error":
        score_val = mean_absolute_percentage_error(y_true, y_pred, **score_kwargs)
    elif scoring == "r2":
        score_val = r2_score(y_true, y_pred, **score_kwargs)
    else:
        msg = "Specified scoring is not supported, {}".format(scoring)
        raise ValueError(msg)
    return score_val
