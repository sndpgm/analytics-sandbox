from sandbox.ensemble.base import _BaseEnsembleModelMeta, _NotInstalledModel
from sandbox.metrics.score import score

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = _NotInstalledModel

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = _NotInstalledModel


class XGBoostRegressor(XGBRegressor, metaclass=_BaseEnsembleModelMeta):
    """Implementation of the scikit-learn API for XGBoost regression.
    Here is the details of version 1.5.0 xgboost arguments.

    .. attention::

        Not explicitly defining, the methods of :class:`xgboost.XGBRegressor` including
        :obj:`~xgboost.XGBRegressor.fit` and :obj:`~xgboost.XGBRegressor.predict` methods are inheritance.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of gradient boosted trees. Equivalent to number of boosting rounds.
    max_depth : int or None, default=None
        Maximum tree depth for base learners.
    max_leaves : int or None, default=None
        Maximum number of leaves; 0 indicates no limit.
    max_bin : int or None, default=None
        If using histogram-based algorithm, maximum number of bins per feature
    grow_policy : str or None, default=None
        Tree growing policy. 0: favor splitting at nodes closest to the node, i.e. grow
        depth-wise. 1: favor splitting at nodes with highest loss change.
    learning_rate : float or None, default=None
        Boosting learning rate (xgb's "eta")
    verbosity : int or None, default=None
        The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
    objective : default=None
        Specify the learning task and the corresponding learning objective or
        a custom objective function to be used (see note below).
    booster: str or None, default=None
        Specify which booster to use: gbtree, gblinear or dart.
    tree_method: str or None, default=None
        Specify which tree method to use.  Default to auto.  If this parameter is set to
        default, XGBoost will choose the most conservative option available.  It's
        recommended to study this option from the parameters document `tree method (XGBoost official API document)
        <https://xgboost.readthedocs.io/en/stable/treemethod.html>`_
    n_jobs : int or None, default=None
        Number of parallel threads used to run xgboost. When used with other
        Scikit-Learn algorithms like grid search, you may choose which algorithm to
        parallelize and balance the threads. Creating thread contention will
        significantly slow down both algorithms.
    gamma : float or None, default=None
        (min_split_loss) Minimum loss reduction required to make a further partition on a
        leaf node of the tree.
    min_child_weight : float or None, default=None
        Minimum sum of instance weight(hessian) needed in a child.
    max_delta_step : float or None, default=None
        Maximum delta step we allow each tree's weight estimation to be.
    subsample : float or None, default=None
        Subsample ratio of the training instance.
    sampling_method : str or None, default=None
        Sampling method. Used only by `gpu_hist` tree method.
          - `uniform`: select random training instances uniformly.
          - `gradient_based` select random training instances with higher probability when
            the gradient and hessian are larger. (cf. CatBoost)
    colsample_bytree : float or None, default=None
        Subsample ratio of columns when constructing each tree.
    colsample_bylevel : float or None, default=None
        Subsample ratio of columns for each level.
    colsample_bynode : float or None, default=None
        Subsample ratio of columns for each split.
    reg_alpha : float or None, default=None
        L1 regularization term on weights (xgb's alpha).
    reg_lambda : float or None, default=None
        L2 regularization term on weights (xgb's lambda).
    scale_pos_weight : float or None, default=None
        Balancing of positive and negative weights.
    base_score : float or None, default=None
        The initial prediction score of all instances, global bias.
    random_state : Optional[Union[numpy.random.RandomState, int]] or None, default=None
        Random number seed.

        .. note::
           Using gblinear booster with shotgun updater is nondeterministic as
           it uses Hogwild algorithm.

    missing : float, default=np.nan
        Value in the data which needs to be present as a missing value.
    num_parallel_tree: int or None, default=None
        Used for boosting random forest.
    monotone_constraints : Optional[Union[Dict[str, int], str]] or None, default=None
        Constraint of variable monotonicity.  See
        `XGBoost official tutorial <https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html>`_
        for more information.
    interaction_constraints : Optional[Union[str, List[Tuple[str]]]] or None, default=None
        Constraints for interaction representing permitted interactions.  The
        constraints must be specified in the form of a nested list, e.g. ``[[0, 1], [2,
        3, 4]]``, where each inner list is a group of indices of features that are
        allowed to interact with each other.  See `XGBoost official tutorial
        <https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html>`_
        for more information
    importance_type: str or None, default=None
        The feature importance type for the feature_importances\\_ property:

        * For tree model, it's either "gain", "weight", "cover", "total_gain" or "total_cover".
        * For linear model, only "weight" is defined and it's the normalized coefficients without bias.

    gpu_id : int or None, default=None
        Device ordinal.
    validate_parameters : bool or None, default=None
        Give warnings for unknown parameter.
    predictor : str or None, default=None
        Force XGBoost to use specific predictor, available choices are [cpu_predictor,
        gpu_predictor].
    enable_categorical : bool or None, default=None
        Experimental support for categorical data.  When enabled, cudf/pandas.DataFrame
        should be used to specify categorical data type.  Also, JSON/UBJSON
        serialization format is required.
    eval_metric : Optional[Union[str, List[str], Callable]], default=None
        Metric used for monitoring the training result and early stopping.  It can be a
        string or list of strings as names of predefined metric in XGBoost (See
        doc/parameter.rst), one of the metrics in :py:mod:`sklearn.metrics`, or any other
        user defined metric that looks like `sklearn.metrics`.
        If custom objective is also provided, then custom metric should implement the
        corresponding reverse link function.
        Unlike the `scoring` parameter commonly used in scikit-learn, when a callable
        object is provided, it's assumed to be a cost function and by default XGBoost will
        minimize the result during early stopping.
        For advanced usage on Early stopping like directly choosing to maximize instead of
        minimize, see :py:obj:`xgboost.callback.EarlyStopping`.
        See :doc:`Custom Objective and Evaluation Metric </tutorials/custom_metric_obj>`
        for more.

        .. highlight:: python
        .. code-block:: python

            from sandbox.ensemble.boost import XGBoostRegressor
            from sklearn.datasets import load_diabetes
            from sklearn.metrics import mean_absolute_error
            X, y = load_diabetes(return_X_y=True)
            reg = XGBoostRegressor(
                tree_method="hist",
                eval_metric=mean_absolute_error,
            )
            reg.fit(X, y, eval_set=[(X, y)])

    early_stopping_rounds : int or None, default=None
        Activates early stopping. Validation metric needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training. Requires at least
        one item in **eval_set** in :py:meth:`fit`.
        The method returns the model from the last iteration (not the best one). If
        there's more than one item in **eval_set**, the last entry will be used for early
        stopping. If there's more than one metric in **eval_metric**, the last metric
        will be used for early stopping.
        If early stopping occurs, the model will have three additional fields:
        :py:attr:`best_score`, :py:attr:`best_iteration` and
        :py:attr:`best_ntree_limit`.

    callbacks : Optional[List[TrainingCallback]] or None, default=None
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using
        `Callback API <https://xgboost.readthedocs.io/en/stable/python/python_api.html#callback-api>`_.

        .. note::

           States in callback are not preserved during training, which means callback
           objects can not be reused for multiple training sessions without
           reinitialization or deepcopy.

        .. highlight:: python
        .. code-block:: python

            for params in parameters_grid:
                # be sure to (re)initialize the callbacks before each run
                callbacks = [xgb.callback.LearningRateScheduler(custom_rates)]
                xgboost.train(params, Xy, callbacks=callbacks)

    kwargs : Any, optional
        Keyword arguments for XGBoost Booster object.  Full documentation of parameters
        can be found `here <https://xgboost.readthedocs.io/en/stable/parameter.html>`_.
        Attempting to set a parameter via the constructor args and \\*\\*kwargs
        dict simultaneously will result in a TypeError.

        .. note::

            \\*\\*kwargs is unsupported by scikit-learn. We do not guarantee
            that parameters passed via this argument will interact properly
            with scikit-learn.

    See Also
    --------
    xgboost.XGBRegressor : Implementation of the scikit-learn API for XGBoost regression.
    xgboost.XGBRegressor.fit : Fit gradient boosting model.
    xgboost.XGBRegressor.predict : Predict with `X`
    """

    _required_package = "xgboost"

    def score(self, X, y, scoring="r2", **score_kwargs):
        """Return score metric.

        Parameters
        ----------
        X :
            Feature matrix.
        y :
            Labels.
        scoring : str, default="r2"
            Which metric to use.
        score_kwargs : dict
            Parameters passed to the `score` method of the estimator.

        See Also
        --------
        sandbox.metrics.score.score : Score function.

        """
        y_pred = self.predict(X)
        return score(y_true=y, y_pred=y_pred, scoring=scoring, **score_kwargs)


class LightGBMRegressor(LGBMRegressor, metaclass=_BaseEnsembleModelMeta):
    """Construct a gradient boosting model.
    Here is the details of version 3.2.0 lightgbm arguments.

    .. attention::

        Not explicitly defining, the methods of :class:`lightgbm.LGBMRegressor` including
        :obj:`~lightgbm.LGBMRegressor.fit` and :obj:`~lightgbm.LGBMRegressor.predict` methods are inheritance.

    Parameters
    ----------
    boosting_type : str, default='gbdt'
        - 'gbdt', traditional Gradient Boosting Decision Tree.
        - 'dart', Dropouts meet Multiple Additive Regression Trees.
        - 'goss', Gradient-based One-Side Sampling.
        - 'rf', Random Forest.
    num_leaves : int, default=31
        Maximum tree leaves for base learners.
    max_depth : int, default=-1
        Maximum tree depth for base learners, <=0 means no limit.
    learning_rate : float, default=0.1
        Boosting learning rate.
        You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
        in training using ``reset_parameter`` callback.
        Note, that this will ignore the ``learning_rate`` argument in training.
    n_estimators : int, default=100
        Number of boosted trees to fit.
    subsample_for_bin : int, default=200000
        Number of samples for constructing bins.
    objective : str, callable or None, default=None
        Specify the learning task and the corresponding learning objective or
        a custom objective function to be used (see note below).
        Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass'
        for LGBMClassifier, 'lambdarank' for LGBMRanker.
    class_weight : dict, 'balanced' or None, default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        Use this parameter only for multi-class classification task;
        for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
        Note, that the usage of all these parameters will result in poor estimates of
        the individual class probabilities. You may want to consider performing probability calibration
        (https://scikit-learn.org/stable/modules/calibration.html) of your model.
        The 'balanced' mode uses the values of y to automatically adjust weights
        inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
        If None, all classes are supposed to have weight one.
        Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
        if ``sample_weight`` is specified.
    min_split_gain : float, default=0.
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    min_child_weight : float, default=1e-3
        Minimum sum of instance weight (Hessian) needed in a child (leaf).
    min_child_samples : int, default=20
        Minimum number of data needed in a child (leaf).
    subsample : float, default=1.
        Subsample ratio of the training instance.
    subsample_freq : int, default=0
        Frequency of subsample, <=0 means no enable.
    colsample_bytree : float, default=1.
        Subsample ratio of columns when constructing each tree.
    reg_alpha : float, default=0.
        L1 regularization term on weights.
    reg_lambda : float, default=0.
        L2 regularization term on weights.
    random_state : int, RandomState object or None, default=None
        Random number seed.
        If int, this number is used to seed the C++ code.
        If RandomState object (numpy), a random integer is picked based on its state to seed the C++ code.
        If None, default seeds in C++ code are used.
    n_jobs : int or None, default=None
        Number of parallel threads to use for training (can be changed at prediction time by
        passing it as an extra keyword argument).

        For better performance, it is recommended to set this to the number of physical cores
        in the CPU.

        Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
        scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
        threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
        to using the number of physical cores in the system (its correct detection requires
        either the ``joblib`` or the ``psutil`` util libraries to be installed).
    importance_type : str, default='split'
        The type of feature importance to be filled into ``feature_importances_``.
        If 'split', result contains numbers of times the feature is used in a model.
        If 'gain', result contains total gains of splits which use the feature.
    **kwargs
        Other parameters for the model.
        Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

        .. warning::

           \\*\\*kwargs is not supported in sklearn, it may cause unexpected issues.

    Note
    ----
    A custom objective function can be provided for the ``objective`` parameter.
    In this case, it should have the signature
    ``objective(y_true, y_pred) -> grad, hess``,
    ``objective(y_true, y_pred, weight) -> grad, hess``
    or ``objective(y_true, y_pred, weight, group) -> grad, hess``:

        y_true : numpy 1-D array of shape = [n_samples]
            The target values.
        y_pred : numpy 1-D array of shape = [n_samples] or
        numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
            The predicted values.
            Predicted values are returned before any transformation,
            e.g. they are raw margin instead of probability of positive class for binary task.
        weight : numpy 1-D array of shape = [n_samples]
            The weight of samples. Weights should be non-negative.
        group : numpy 1-D array
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``,
            that means that you have 6 groups, where the first 10 records are in the first group,
            records 11-30 are in the second group, records 31-70 are in the third group, etc.
        grad : numpy 1-D array of shape = [n_samples]
        or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
            The value of the first order derivative (gradient) of the loss
            with respect to the elements of y_pred for each sample point.
        hess : numpy 1-D array of shape = [n_samples]
        or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
            The value of the second order derivative (Hessian) of the loss
            with respect to the elements of y_pred for each sample point.

    For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],
    and grad and hess should be returned to the same format.

    See Also
    --------
    lightgbm.LGBMRegressor : Construct a gradient boosting model.
    lightgbm.LGBMRegressor.fit : Build a gradient boosting model from the training set (X, y).
    lightgbm.LGBMRegressor.predict : Return the predicted value for each sample.

    """

    _required_package = "lightgbm"

    def score(self, X, y, scoring="r2", **score_kwargs):
        """Return score metric.

        Parameters
        ----------
        X :
            Feature matrix.
        y :
            Labels.
        scoring : str, default="r2"
            Which metric to use.
        score_kwargs : dict
            Parameters passed to the `score` method of the estimator.

        See Also
        --------
        sandbox.metrics.score.score : Score function.

        """
        y_pred = self.predict(X)
        return score(y_true=y, y_pred=y_pred, scoring=scoring, **score_kwargs)
