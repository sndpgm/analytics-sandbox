:py:mod:`sandbox.ensemble.boost`
================================

.. py:module:: sandbox.ensemble.boost


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.ensemble.boost.XGBoostRegressor
   sandbox.ensemble.boost.LightGBMRegressor




Attributes
~~~~~~~~~~

.. autoapisummary::

   sandbox.ensemble.boost.XGBRegressor
   sandbox.ensemble.boost.LGBMRegressor


.. py:data:: XGBRegressor
   

   

.. py:data:: LGBMRegressor
   

   

.. py:class:: XGBoostRegressor(*, objective: _SklObjective = 'reg:squarederror', **kwargs: Any)

   Bases: :py:obj:`xgboost.XGBRegressor`

   
   Implementation of the scikit-learn API for XGBoost regression.
   Here is the details of version 1.5.0 xgboost arguments.

   .. attention::

       Not explicitly defining, the methods of :class:`xgboost.XGBRegressor` including
       :obj:`~xgboost.XGBRegressor.fit` and :obj:`~xgboost.XGBRegressor.predict` methods are inheritance.

   :param n_estimators: Number of gradient boosted trees. Equivalent to number of boosting rounds.
   :type n_estimators: int, default=100
   :param max_depth: Maximum tree depth for base learners.
   :type max_depth: int or None, default=None
   :param max_leaves: Maximum number of leaves; 0 indicates no limit.
   :type max_leaves: int or None, default=None
   :param max_bin: If using histogram-based algorithm, maximum number of bins per feature
   :type max_bin: int or None, default=None
   :param grow_policy: Tree growing policy. 0: favor splitting at nodes closest to the node, i.e. grow
                       depth-wise. 1: favor splitting at nodes with highest loss change.
   :type grow_policy: str or None, default=None
   :param learning_rate: Boosting learning rate (xgb's "eta")
   :type learning_rate: float or None, default=None
   :param verbosity: The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
   :type verbosity: int or None, default=None
   :param objective: Specify the learning task and the corresponding learning objective or
                     a custom objective function to be used (see note below).
   :type objective: default=None
   :param booster: Specify which booster to use: gbtree, gblinear or dart.
   :type booster: str or None, default=None
   :param tree_method: Specify which tree method to use.  Default to auto.  If this parameter is set to
                       default, XGBoost will choose the most conservative option available.  It's
                       recommended to study this option from the parameters document `tree method (XGBoost official API document)
                       <https://xgboost.readthedocs.io/en/stable/treemethod.html>`_
   :type tree_method: str or None, default=None
   :param n_jobs: Number of parallel threads used to run xgboost. When used with other
                  Scikit-Learn algorithms like grid search, you may choose which algorithm to
                  parallelize and balance the threads. Creating thread contention will
                  significantly slow down both algorithms.
   :type n_jobs: int or None, default=None
   :param gamma: (min_split_loss) Minimum loss reduction required to make a further partition on a
                 leaf node of the tree.
   :type gamma: float or None, default=None
   :param min_child_weight: Minimum sum of instance weight(hessian) needed in a child.
   :type min_child_weight: float or None, default=None
   :param max_delta_step: Maximum delta step we allow each tree's weight estimation to be.
   :type max_delta_step: float or None, default=None
   :param subsample: Subsample ratio of the training instance.
   :type subsample: float or None, default=None
   :param sampling_method:
                           Sampling method. Used only by `gpu_hist` tree method.
                             - `uniform`: select random training instances uniformly.
                             - `gradient_based` select random training instances with higher probability when
                               the gradient and hessian are larger. (cf. CatBoost)
   :type sampling_method: str or None, default=None
   :param colsample_bytree: Subsample ratio of columns when constructing each tree.
   :type colsample_bytree: float or None, default=None
   :param colsample_bylevel: Subsample ratio of columns for each level.
   :type colsample_bylevel: float or None, default=None
   :param colsample_bynode: Subsample ratio of columns for each split.
   :type colsample_bynode: float or None, default=None
   :param reg_alpha: L1 regularization term on weights (xgb's alpha).
   :type reg_alpha: float or None, default=None
   :param reg_lambda: L2 regularization term on weights (xgb's lambda).
   :type reg_lambda: float or None, default=None
   :param scale_pos_weight: Balancing of positive and negative weights.
   :type scale_pos_weight: float or None, default=None
   :param base_score: The initial prediction score of all instances, global bias.
   :type base_score: float or None, default=None
   :param random_state: Random number seed.

                        .. note::
                           Using gblinear booster with shotgun updater is nondeterministic as
                           it uses Hogwild algorithm.
   :type random_state: Optional[Union[numpy.random.RandomState, int]] or None, default=None
   :param missing: Value in the data which needs to be present as a missing value.
   :type missing: float, default=np.nan
   :param num_parallel_tree: Used for boosting random forest.
   :type num_parallel_tree: int or None, default=None
   :param monotone_constraints: Constraint of variable monotonicity.  See
                                `XGBoost official tutorial <https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html>`_
                                for more information.
   :type monotone_constraints: Optional[Union[Dict[str, int], str]] or None, default=None
   :param interaction_constraints: Constraints for interaction representing permitted interactions.  The
                                   constraints must be specified in the form of a nested list, e.g. ``[[0, 1], [2,
                                   3, 4]]``, where each inner list is a group of indices of features that are
                                   allowed to interact with each other.  See `XGBoost official tutorial
                                   <https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html>`_
                                   for more information
   :type interaction_constraints: Optional[Union[str, List[Tuple[str]]]] or None, default=None
   :param importance_type: The feature importance type for the feature_importances\_ property:

                           * For tree model, it's either "gain", "weight", "cover", "total_gain" or "total_cover".
                           * For linear model, only "weight" is defined and it's the normalized coefficients without bias.
   :type importance_type: str or None, default=None
   :param gpu_id: Device ordinal.
   :type gpu_id: int or None, default=None
   :param validate_parameters: Give warnings for unknown parameter.
   :type validate_parameters: bool or None, default=None
   :param predictor: Force XGBoost to use specific predictor, available choices are [cpu_predictor,
                     gpu_predictor].
   :type predictor: str or None, default=None
   :param enable_categorical: Experimental support for categorical data.  When enabled, cudf/pandas.DataFrame
                              should be used to specify categorical data type.  Also, JSON/UBJSON
                              serialization format is required.
   :type enable_categorical: bool or None, default=None
   :param callbacks: List of callback functions that are applied at end of each iteration.
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
   :type callbacks: Optional[List[TrainingCallback]] or None, default=None
   :param kwargs: Keyword arguments for XGBoost Booster object.  Full documentation of parameters
                  can be found `here <https://xgboost.readthedocs.io/en/stable/parameter.html>`_.
                  Attempting to set a parameter via the constructor args and \*\*kwargs
                  dict simultaneously will result in a TypeError.

                  .. note::

                      \*\*kwargs is unsupported by scikit-learn. We do not guarantee
                      that parameters passed via this argument will interact properly
                      with scikit-learn.
   :type kwargs: Any, optional

   .. seealso::

      :obj:`xgboost.XGBRegressor`
          Implementation of the scikit-learn API for XGBoost regression.

      :obj:`xgboost.XGBRegressor.fit`
          Fit gradient boosting model.

      :obj:`xgboost.XGBRegressor.predict`
          Predict with `X`















   ..
       !! processed by numpydoc !!
   .. py:method:: score(X, y, scoring='r2', **score_kwargs)

      
      Return score metric.

      :param X: Feature matrix.
      :param y: Labels.
      :param scoring: Which metric to use.
      :type scoring: str, default="r2"
      :param score_kwargs: Parameters passed to the `score` method of the estimator.
      :type score_kwargs: dict

      .. seealso::

         :obj:`sandbox.metrics.score.score`
             Score function.















      ..
          !! processed by numpydoc !!


.. py:class:: LightGBMRegressor(boosting_type: str = 'gbdt', num_leaves: int = 31, max_depth: int = -1, learning_rate: float = 0.1, n_estimators: int = 100, subsample_for_bin: int = 200000, objective: Optional[Union[str, Callable]] = None, class_weight: Optional[Union[Dict, str]] = None, min_split_gain: float = 0.0, min_child_weight: float = 0.001, min_child_samples: int = 20, subsample: float = 1.0, subsample_freq: int = 0, colsample_bytree: float = 1.0, reg_alpha: float = 0.0, reg_lambda: float = 0.0, random_state: Optional[Union[int, numpy.random.RandomState]] = None, n_jobs: int = -1, silent: Union[bool, str] = 'warn', importance_type: str = 'split', **kwargs)

   Bases: :py:obj:`lightgbm.LGBMRegressor`

   
   Construct a gradient boosting model.
   Here is the details of version 3.2.0 lightgbm arguments.

   .. attention::

       Not explicitly defining, the methods of :class:`lightgbm.LGBMRegressor` including
       :obj:`~lightgbm.LGBMRegressor.fit` and :obj:`~lightgbm.LGBMRegressor.predict` methods are inheritance.

   :param boosting_type:
                         - 'gbdt', traditional Gradient Boosting Decision Tree.
                         - 'dart', Dropouts meet Multiple Additive Regression Trees.
                         - 'goss', Gradient-based One-Side Sampling.
                         - 'rf', Random Forest.
   :type boosting_type: str, default='gbdt'
   :param num_leaves: Maximum tree leaves for base learners.
   :type num_leaves: int, default=31
   :param max_depth: Maximum tree depth for base learners, <=0 means no limit.
   :type max_depth: int, default=-1
   :param learning_rate: Boosting learning rate.
                         You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
                         in training using ``reset_parameter`` callback.
                         Note, that this will ignore the ``learning_rate`` argument in training.
   :type learning_rate: float, default=0.1
   :param n_estimators: Number of boosted trees to fit.
   :type n_estimators: int, default=100
   :param subsample_for_bin: Number of samples for constructing bins.
   :type subsample_for_bin: int, default=200000
   :param objective: Specify the learning task and the corresponding learning objective or
                     a custom objective function to be used (see note below).
                     Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass'
                     for LGBMClassifier, 'lambdarank' for LGBMRanker.
   :type objective: str, callable or None, default=None
   :param class_weight: Weights associated with classes in the form ``{class_label: weight}``.
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
   :type class_weight: dict, 'balanced' or None, default=None
   :param min_split_gain: Minimum loss reduction required to make a further partition on a leaf node of the tree.
   :type min_split_gain: float, default=0.
   :param min_child_weight: Minimum sum of instance weight (Hessian) needed in a child (leaf).
   :type min_child_weight: float, default=1e-3
   :param min_child_samples: Minimum number of data needed in a child (leaf).
   :type min_child_samples: int, default=20
   :param subsample: Subsample ratio of the training instance.
   :type subsample: float, default=1.
   :param subsample_freq: Frequency of subsample, <=0 means no enable.
   :type subsample_freq: int, default=0
   :param colsample_bytree: Subsample ratio of columns when constructing each tree.
   :type colsample_bytree: float, default=1.
   :param reg_alpha: L1 regularization term on weights.
   :type reg_alpha: float, default=0.
   :param reg_lambda: L2 regularization term on weights.
   :type reg_lambda: float, default=0.
   :param random_state: Random number seed.
                        If int, this number is used to seed the C++ code.
                        If RandomState object (numpy), a random integer is picked based on its state to seed the C++ code.
                        If None, default seeds in C++ code are used.
   :type random_state: int, RandomState object or None, default=None
   :param n_jobs: Number of parallel threads to use for training (can be changed at prediction time by
                  passing it as an extra keyword argument).

                  For better performance, it is recommended to set this to the number of physical cores
                  in the CPU.

                  Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
                  scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
                  threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
                  to using the number of physical cores in the system (its correct detection requires
                  either the ``joblib`` or the ``psutil`` util libraries to be installed).
   :type n_jobs: int or None, default=None
   :param importance_type: The type of feature importance to be filled into ``feature_importances_``.
                           If 'split', result contains numbers of times the feature is used in a model.
                           If 'gain', result contains total gains of splits which use the feature.
   :type importance_type: str, default='split'
   :param \*\*kwargs: Other parameters for the model.
                      Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

                      .. warning::

                         \*\*kwargs is not supported in sklearn, it may cause unexpected issues.

   .. note::

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

   .. seealso::

      :obj:`lightgbm.LGBMRegressor`
          Construct a gradient boosting model.

      :obj:`lightgbm.LGBMRegressor.fit`
          Build a gradient boosting model from the training set (X, y).

      :obj:`lightgbm.LGBMRegressor.predict`
          Return the predicted value for each sample.















   ..
       !! processed by numpydoc !!
   .. py:method:: score(X, y, scoring='r2', **score_kwargs)

      
      Return score metric.

      :param X: Feature matrix.
      :param y: Labels.
      :param scoring: Which metric to use.
      :type scoring: str, default="r2"
      :param score_kwargs: Parameters passed to the `score` method of the estimator.
      :type score_kwargs: dict

      .. seealso::

         :obj:`sandbox.metrics.score.score`
             Score function.















      ..
          !! processed by numpydoc !!


