:py:mod:`sandbox.model_selection`
=================================

.. py:module:: sandbox.model_selection


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.model_selection.BaseOptunaSearchCV
   sandbox.model_selection.BaseOptunaStudyInitializer
   sandbox.model_selection.LightGBMOptunaStepwiseSearchCV
   sandbox.model_selection.XGBoostOptunaSearchCV
   sandbox.model_selection.GroupTimeSeriesSplit
   sandbox.model_selection.PurgedGroupTimeSeriesSplit



Functions
~~~~~~~~~

.. autoapisummary::

   sandbox.model_selection.plot_cv_indices



.. py:class:: BaseOptunaSearchCV(estimator, scoring, cv=None, n_jobs=None, pre_dispatch='2*n_jobs', storage=None, study_name=None, direction='minimize', load_if_exists=False, sampler=None, sampler_seed=42)

   Bases: :py:obj:`BaseOptunaStudyInitializer`, :py:obj:`sklearn.base.BaseEstimator`

   
   Base class for hyperparameter search using `Optuna`.

   .. rubric:: Examples

   Inheriting to this base class, it is easy to create
   the custom class on hyperparameter tuning for your estimator.

   .. highlight:: python
   .. code-block:: python

       from sandbox.model_selection import BaseOptunaSearchCV

       class SVMOptunaSearchCV(BaseOptunaSearchCV):
           def __init__(
               self,
               scoring="mse",
               cv=None,
               n_jobs=None,
               pre_dispatch="2*n_jobs",
               direction="minimize",
           ):
               estimator = SVR()
               super(SVMOptunaSearchCV, self).__init__(
                   estimator=estimator,
                   scoring=scoring,
                   cv=cv,
                   n_jobs=n_jobs,
                   pre_dispatch=pre_dispatch,
                   direction=direction,
               )

           # you have to override params method, and define the parameters to be searched.
           def params(self, trial):
               return {
                   "C": trial.suggest_loguniform(
                       'C',
                       1e0, 1e2
                   )
                   ,
                   "epsilon": trial.suggest_loguniform(
                       'epsilon',
                       1e-1, 1e1
                   ),
               }

   :param estimator: The estimator class compatible with scikit-learn
   :type estimator: estimator object
   :param scoring: Which metric to use in evaluating the precision of cross validated estimator using `Optuna`.
   :type scoring: str
   :param cv:
              Determines the cross-validation splitting strategy. Possible inputs for cv are:

                  - None, to use the default 5-fold cross validation,
                  - integer, to specify the number of folds in a `KFold`,
                  - CV splitter,
                  - An iterable yielding (train, test) splits as arrays of indices.
   :type cv: {None, int, cross-validation generator or and iterable}, default=None
   :param n_jobs: Number of jobs to run in parallel.
                  ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
                  ``-1`` means using all processors.
   :type n_jobs: int, default=None
   :param pre_dispatch: Controls the number of jobs that get dispatched during parallel
                        execution. Reducing this number can be useful to avoid an
                        explosion of memory consumption when more jobs get dispatched
                        than CPUs can process. This parameter can be:

                            - None, in which case all the jobs are immediately
                              created and spawned. Use this for lightweight and
                              fast-running jobs, to avoid delays due to on-demand
                              spawning of the jobs
                            - An int, giving the exact number of total jobs that are
                              spawned
                            - A str, giving an expression as a function of n_jobs,
                              as in '2*n_jobs'
   :type pre_dispatch: {int, str}, default='2*n_jobs'
   :param storage: Database URL. If this argument is set to None, in-memory storage is used, and the
                   :class:`optuna.study.Study` will not be persistent.

                   .. note::
                       When a database URL is passed, Optuna internally uses `SQLAlchemy`_ to handle
                       the database. Please refer to `SQLAlchemy's document`_ for further details.
                       If you want to specify non-default options to `SQLAlchemy Engine`_, you can
                       instantiate :class:`~optuna.storages.RDBStorage` with your desired options and
                       pass it to the ``storage`` argument instead of a URL.

                    .. _SQLAlchemy: https://www.sqlalchemy.org/
                    .. _SQLAlchemy's document:
                        https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
                    .. _SQLAlchemy Engine: https://docs.sqlalchemy.org/en/latest/core/engines.html
   :type storage: {None, str}, default=None
   :param study_name: Study's name. If this argument is set to None, a unique name is generated automatically.
   :type study_name: {None, str}, default=None
   :param direction: Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for maximization.
                     You can also pass the corresponding :class:`optuna.study.StudyDirection` object.
   :type direction: str, default=minimize
   :param load_if_exists: Flag to control the behavior to handle a conflict of study names.
                          In the case where a study named ``study_name`` already exists in the ``storage``,
                          a :class:`optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
                          set to :obj:`False`. Otherwise, the creation of the study is skipped, and the existing one is returned.
   :type load_if_exists: bool, default=False
   :param sampler: A sampler object that implements background algorithm for value suggestion.
                   If :obj:`None` is specified, :class:`optuna.samplers.TPESampler` is used.
   :type sampler: {optuna.samplers, None}, default=None
   :param sampler_seed: Seed for random number generator.
   :type sampler_seed: int, default=42















   ..
       !! processed by numpydoc !!
   .. py:method:: study()
      :property:

      
      This has the all results of searching hyperparameter in the instance.

      .. seealso::

         :obj:`optuna.study.Study`
             A study corresponds to an optimization task, i.e., a set of trials.















      ..
          !! processed by numpydoc !!

   .. py:method:: params(trial)
      :abstractmethod:

      
      This returns the hyperparameter search space for your defined estimator.

      .. rubric:: Examples

      When you want to search the hyperparameter on support vector machine,
      the following two parameters are to be searched: `C` and `epsilon`

      .. highlight:: python
      .. code-block:: python

          ...
          def params(self, trial):
              return {
                  "C": trial.suggest_loguniform(
                      'C',
                      1e0, 1e2
                  )
                  ,
                  "epsilon": trial.suggest_loguniform(
                      'epsilon',
                      1e-1, 1e1
                  ),
              }

      .. seealso::

         :obj:`optuna.trial.Trial.suggest_categorical`
             Suggest a value for the categorical parameter.

         :obj:`optuna.trial.Trial.suggest_discrete_uniform`
             Suggest a value for the discrete parameter.

         :obj:`optuna.trial.Trial.suggest_float`
             Suggest a value for the floating point parameter.

         :obj:`optuna.trial.Trial.suggest_int`
             Suggest a value for the integer parameter.

         :obj:`optuna.trial.Trial.suggest_loguniform`
             Suggest a value for the continuous parameter.

         :obj:`optuna.trial.Trial.suggest_uniform`
             Suggest a value for the continuous parameter.















      ..
          !! processed by numpydoc !!

   .. py:method:: fit(X, y, groups=None, n_trials=10, show_progress_bar=False, optuna_verbosity=1, **fit_params)

      
      Execute hyperparameter tuning.

      :param X: The input samples.
      :param y: Target values (strings or integers in classification, real numbers in regression).
                For classification, labels must correspond to classes.
      :param groups: Group labels for the samples used while splitting the dataset into train/test set.
      :param n_trials: The number of trials.
      :type n_trials: int
      :param show_progress_bar: Flag to show progress bars or not. To disable progress bar, set this :obj:`False`.
                                Currently, progress bar is experimental feature and disabled when ``n_jobs`` :math:`\ne 1`.
      :type show_progress_bar: bool, default=False
      :param optuna_verbosity: The degree of verbosity in `Optuna` optimization. Valid values are 0 (silent) - 3 (debug).
      :type optuna_verbosity: int, default=1
      :param fit_params: Parameters passed to the `fit` method of the estimator.
      :type fit_params: dict















      ..
          !! processed by numpydoc !!


.. py:class:: BaseOptunaStudyInitializer(storage=None, study_name=None, direction='minimize', load_if_exists=False, sampler=None, **sampler_params)

   
   Base initializer class for study instance.

   :param storage: Database URL. If this argument is set to None, in-memory storage is used, and the
                   :class:`optuna.study.Study` will not be persistent.

                   .. note::
                       When a database URL is passed, Optuna internally uses `SQLAlchemy`_ to handle
                       the database. Please refer to `SQLAlchemy's document`_ for further details.
                       If you want to specify non-default options to `SQLAlchemy Engine`_, you can
                       instantiate :class:`~optuna.storages.RDBStorage` with your desired options and
                       pass it to the ``storage`` argument instead of a URL.

                    .. _SQLAlchemy: https://www.sqlalchemy.org/
                    .. _SQLAlchemy's document:
                        https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
                    .. _SQLAlchemy Engine: https://docs.sqlalchemy.org/en/latest/core/engines.html
   :type storage: {None, str}, default=None
   :param study_name: Study's name. If this argument is set to None, a unique name is generated automatically.
   :type study_name: {None, str}, default=None
   :param direction: Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for maximization.
                     You can also pass the corresponding :class:`optuna.study.StudyDirection` object.
   :type direction: str, default=minimize
   :param load_if_exists: Flag to control the behavior to handle a conflict of study names.
                          In the case where a study named ``study_name`` already exists in the ``storage``,
                          a :class:`optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
                          set to :obj:`False`. Otherwise, the creation of the study is skipped, and the existing one is returned.
   :type load_if_exists: bool, default=False
   :param sampler: A sampler object that implements background algorithm for value suggestion.
                   If :obj:`None` is specified, :class:`optuna.samplers.TPESampler` is used.
   :type sampler: {optuna.samplers, None}, default=None
   :param sampler_params: Parameters passed to the specified `optuna.samplers`.
   :type sampler_params: dict















   ..
       !! processed by numpydoc !!
   .. py:method:: optuna_sampler(sampler=None, **sampler_params)
      :staticmethod:

      
      Return your specified `optuna.samplers`

      :param sampler: A sampler object that implements background algorithm for value suggestion.
                      If :obj:`None` is specified, :class:`optuna.samplers.TPESampler` is used.
      :type sampler: {optuna.samplers, None}, default=None
      :param sampler_params: Parameters passed to the specified `optuna.samplers`.
      :type sampler_params: dict

      :returns: **sampler**
      :rtype: optuna.samplers















      ..
          !! processed by numpydoc !!

   .. py:method:: create_study()

      
      Create `optuna.study.Study` instance.

      :returns: **study**
      :rtype: optuna.study.Study















      ..
          !! processed by numpydoc !!

   .. py:method:: params(**kwargs)
      :abstractmethod:

      
      The parameter search space which should be implemented in the subclass which is inheritance to this class.

      :param kwargs:
      :type kwargs: dict

      :rtype: The parameter space to be searched.















      ..
          !! processed by numpydoc !!


.. py:class:: LightGBMOptunaStepwiseSearchCV(n_estimators=1000, boosting_type='gbdt', objective='regression', metric='rmse', early_stopping_rounds=100, random_state=42, verbosity=1, cv=None, storage=None, study_name=None, direction='minimize', load_if_exists=False, sampler=None, sampler_seed=42)

   Bases: :py:obj:`BaseOptunaStudyInitializer`, :py:obj:`sklearn.base.BaseEstimator`

   
   Hyperparameter stepwise search for LightGBM with cross-validation

   :param n_estimators: Number of gradient boosted trees. Equivalent to number of boosting rounds.
   :type n_estimators: int, default=1000
   :param boosting_type:
                         - 'gbdt', traditional Gradient Boosting Decision Tree.
                         - 'dart', Dropouts meet Multiple Additive Regression Trees.
                         - 'goss', Gradient-based One-Side Sampling.
                         - 'rf', Random Forest.
   :type boosting_type: str, default='gbdt'
   :param objective: Specify the learning task and the corresponding learning objective or
                     a custom objective function to be used (see note below).
                     Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass'
                     for LGBMClassifier, 'lambdarank' for LGBMRanker.
   :type objective: {str, callable, None}, default="regression"
   :param metric: Metric(s) to be evaluated on the evaluation set(s).
   :type metric: {str, callable, None}, default="rmse"
   :param early_stopping_rounds: Activates early stopping.
   :type early_stopping_rounds: int
   :param random_state: Random number seed.
   :type random_state: int
   :param verbosity:
                     Controls the level of LightGBM’s verbosity.

                         - ``< 0``: Fatal
                         - ``= 0``: Error (Warning)
                         - ``= 1``: Info
                         - ``> 1``: Debug
   :type verbosity: int, default=1
   :param cv:
              Determines the cross-validation splitting strategy. Possible inputs for cv are:

                  - None, to use the default 5-fold cross validation,
                  - integer, to specify the number of folds in a `KFold`,
                  - CV splitter,
                  - An iterable yielding (train, test) splits as arrays of indices.
   :type cv: {None, int, cross-validation generator or and iterable}, default=None
   :param storage: Database URL. If this argument is set to None, in-memory storage is used, and the
                   :class:`optuna.study.Study` will not be persistent.

                   .. note::
                       When a database URL is passed, Optuna internally uses `SQLAlchemy`_ to handle
                       the database. Please refer to `SQLAlchemy's document`_ for further details.
                       If you want to specify non-default options to `SQLAlchemy Engine`_, you can
                       instantiate :class:`~optuna.storages.RDBStorage` with your desired options and
                       pass it to the ``storage`` argument instead of a URL.

                    .. _SQLAlchemy: https://www.sqlalchemy.org/
                    .. _SQLAlchemy's document:
                        https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
                    .. _SQLAlchemy Engine: https://docs.sqlalchemy.org/en/latest/core/engines.html
   :type storage: {None, str}, default=None
   :param study_name: Study's name. If this argument is set to None, a unique name is generated automatically.
   :type study_name: {None, str}, default=None
   :param direction: Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for maximization.
                     You can also pass the corresponding :class:`optuna.study.StudyDirection` object.
   :type direction: str, default=minimize
   :param load_if_exists: Flag to control the behavior to handle a conflict of study names.
                          In the case where a study named ``study_name`` already exists in the ``storage``,
                          a :class:`optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
                          set to :obj:`False`. Otherwise, the creation of the study is skipped, and the existing one is returned.
   :type load_if_exists: bool, default=False
   :param sampler: A sampler object that implements background algorithm for value suggestion.
                   If :obj:`None` is specified, :class:`optuna.samplers.TPESampler` is used.
   :type sampler: {optuna.samplers, None}, default=None
   :param sampler_seed: Seed for random number generator.
   :type sampler_seed: int, default=42

   .. seealso::

      :obj:`optuna.integration.lightgbm.LightGBMTunerCV`
          Hyperparameter tuner for LightGBM with cross-validation.















   ..
       !! processed by numpydoc !!
   .. py:method:: study()
      :property:

      
      This has the all results of searching hyperparameter in the instance.

      .. seealso::

         :obj:`optuna.study.Study`
             A study corresponds to an optimization task, i.e., a set of trials.















      ..
          !! processed by numpydoc !!

   .. py:method:: fit(X, y, groups=None, show_progress_bar=False, optuna_verbosity=1, optuna_seed=42, **fit_params)

      
      Execute hyperparameter tuning.

      :param X: The input samples.
      :param y: Target values (strings or integers in classification, real numbers in regression).
                For classification, labels must correspond to classes.
      :param groups: Group labels for the samples used while splitting the dataset into train/test set.
      :param show_progress_bar: Flag to show progress bars or not. To disable progress bar, set this :obj:`False`.
                                Currently, progress bar is experimental feature and disabled when ``n_jobs`` :math:`\ne 1`.
      :type show_progress_bar: bool, default=False
      :param optuna_verbosity: The degree of verbosity in `Optuna` optimization. Valid values are 0 (silent) - 3 (debug).
      :type optuna_verbosity: int, default=1
      :param optuna_seed: ``seed`` of :class:`optuna.samplers.TPESampler` for random number generator
                          that affects sampling for ``num_leaves``, ``bagging_fraction``, ``bagging_freq``,
                          ``lambda_l1``, and ``lambda_l2``.

                          .. note::
                              The `deterministic`_ parameter of LightGBM makes training reproducible.
                              Please enable it when you use this argument.
      :type optuna_seed: int,default=42
      :param fit_params: Parameters passed to the `fit` method of the estimator of
                         :class:`~sandbox.ensemble.boost.XGBoostRegressor`.
      :type fit_params: dict
      :param .. _deterministic:
      :type .. _deterministic: https://lightgbm.readthedocs.io/en/latest/Parameters.html#deterministic















      ..
          !! processed by numpydoc !!


.. py:class:: XGBoostOptunaSearchCV(n_estimators=1000, scoring='mse', early_stopping_rounds=None, verbosity=1, cv=None, n_jobs=None, pre_dispatch='2*n_jobs', storage=None, study_name=None, direction='minimize', load_if_exists=False, sampler=None, sampler_seed=42)

   Bases: :py:obj:`BaseOptunaSearchCV`

   
   Hyperparameter search for XGBoost with cross-validation

   :param n_estimators: Number of gradient boosted trees. Equivalent to number of boosting rounds.
   :type n_estimators: int, default=1000
   :param scoring: Which metric to use in evaluating the precision of cross validated estimator.
   :type scoring: str, default="mse"
   :param early_stopping_rounds: Activates early stopping. Validation metric needs to improve at least once in
                                 every **early_stopping_rounds** round(s) to continue training. Requires at least
                                 one item in **eval_set** in :py:meth:`fit`.
                                 The method returns the model from the last iteration (not the best one). If
                                 there's more than one item in **eval_set**, the last entry will be used for early
                                 stopping. If there's more than one metric in **eval_metric**, the last metric
                                 will be used for early stopping.
                                 If early stopping occurs, the model will have three additional fields:
                                 :py:attr:`best_score`, :py:attr:`best_iteration` and
                                 :py:attr:`best_ntree_limit`.
   :type early_stopping_rounds: int or None, default=None
   :param verbosity: The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
   :type verbosity: int or None, default=None
   :param cv:
              Determines the cross-validation splitting strategy. Possible inputs for cv are:

                  - None, to use the default 5-fold cross validation,
                  - integer, to specify the number of folds in a `KFold`,
                  - CV splitter,
                  - An iterable yielding (train, test) splits as arrays of indices.
   :type cv: {None, int, cross-validation generator or and iterable}, default=None
   :param n_jobs: Number of jobs to run in parallel.
                  ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
                  ``-1`` means using all processors.
   :type n_jobs: int, default=None
   :param pre_dispatch: Controls the number of jobs that get dispatched during parallel
                        execution. Reducing this number can be useful to avoid an
                        explosion of memory consumption when more jobs get dispatched
                        than CPUs can process. This parameter can be:

                            - None, in which case all the jobs are immediately
                              created and spawned. Use this for lightweight and
                              fast-running jobs, to avoid delays due to on-demand
                              spawning of the jobs
                            - An int, giving the exact number of total jobs that are
                              spawned
                            - A str, giving an expression as a function of n_jobs,
                              as in '2*n_jobs'
   :type pre_dispatch: {int, str}, default='2*n_jobs'
   :param storage: Database URL. If this argument is set to None, in-memory storage is used, and the
                   :class:`optuna.study.Study` will not be persistent.

                   .. note::
                       When a database URL is passed, Optuna internally uses `SQLAlchemy`_ to handle
                       the database. Please refer to `SQLAlchemy's document`_ for further details.
                       If you want to specify non-default options to `SQLAlchemy Engine`_, you can
                       instantiate :class:`~optuna.storages.RDBStorage` with your desired options and
                       pass it to the ``storage`` argument instead of a URL.

                    .. _SQLAlchemy: https://www.sqlalchemy.org/
                    .. _SQLAlchemy's document:
                        https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
                    .. _SQLAlchemy Engine: https://docs.sqlalchemy.org/en/latest/core/engines.html
   :type storage: {None, str}, default=None
   :param study_name: Study's name. If this argument is set to None, a unique name is generated automatically.
   :type study_name: {None, str}, default=None
   :param direction: Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for maximization.
                     You can also pass the corresponding :class:`optuna.study.StudyDirection` object.
   :type direction: str, default=minimize
   :param load_if_exists: Flag to control the behavior to handle a conflict of study names.
                          In the case where a study named ``study_name`` already exists in the ``storage``,
                          a :class:`optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
                          set to :obj:`False`. Otherwise, the creation of the study is skipped, and the existing one is returned.
   :type load_if_exists: bool, default=False
   :param sampler: A sampler object that implements background algorithm for value suggestion.
                   If :obj:`None` is specified, :class:`optuna.samplers.TPESampler` is used.
   :type sampler: {optuna.samplers, None}, default=None
   :param sampler_seed: Seed for random number generator.
   :type sampler_seed: int, default=42















   ..
       !! processed by numpydoc !!
   .. py:method:: params(trial)

      
      :param trial:
      :type trial: optuna.trial.Trial

      :returns:

                - reg_lambda :
                    - L2 regularization term on weights (xgb's lambda).
                    - The value is sampled from the range :math:`[0.001, 10.0)` in the log domain
                - reg_alpha :
                    - L1 regularization term on weights (xgb's alpha).
                    - The value is sampled from the range :math:`[0.001, 10.0)` in the log domain
                - gamma :
                    - Minimum loss reduction required to make a further partition on a leaf node of the tree.
                    - The value is sampled from the integers in :math:`[0, 20]`
                - colsample_bytree :
                    - Subsample ratio of columns when constructing each tree.
                    - Suggest a value for the categorical parameter: :math:`\{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0\}`
                - subsample :
                    - Subsample ratio of the training instance.
                    - Suggest a value for the categorical parameter: :math:`\{0.4, 0.5, 0.6, 0.7, 0.8, 1.0\}`
                - learning_rate :
                    - Boosting learning rate (xgb's "eta")
                    - Suggest a value for the categorical parameter:
                    :math:`\{0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02\}`
                - max_depth :
                    - Maximum tree depth for base learners.
                    - Suggest a value for the categorical parameter: :math:`\{5, 7, 9, 11, 13, 15, 17\}`
                - min_child_weight :
                    - Minimum sum of instance weight(hessian) needed in a child.
                    - The value is sampled from the integers in :math:`[1, 300]`
      :rtype: The following search space on hyperparameters.















      ..
          !! processed by numpydoc !!

   .. py:method:: fit(X, y, groups=None, n_trials=10, show_progress_bar=False, eval_verbosity=1, optuna_verbosity=1, **fit_params)

      
      Execute hyperparameter tuning.

      :param X: The input samples.
      :param y: Target values (strings or integers in classification, real numbers in regression).
                For classification, labels must correspond to classes.
      :param groups: Group labels for the samples used while splitting the dataset into train/test set.
      :param n_trials: The number of trials.
      :type n_trials: int
      :param show_progress_bar: Flag to show progress bars or not. To disable progress bar, set this :obj:`False`.
                                Currently, progress bar is experimental feature and disabled when ``n_jobs`` :math:`\ne 1`.
      :type show_progress_bar: bool, default=False
      :param eval_verbosity: The degree of verbosity in cross-validation evaluation. Valid values are 0 (silent) - 3 (debug).
      :type eval_verbosity: int, default=1
      :param optuna_verbosity: The degree of verbosity in `Optuna` optimization. Valid values are 0 (silent) - 3 (debug).
      :type optuna_verbosity: int, default=1
      :param fit_params: Parameters passed to the `fit` method of the estimator of
                         :class:`~sandbox.ensemble.boost.XGBoostRegressor`.
      :type fit_params: dict















      ..
          !! processed by numpydoc !!


.. py:class:: GroupTimeSeriesSplit(n_splits=5, *, max_train_size=None, sort_groups=True)

   Bases: :py:obj:`sklearn.model_selection._split._BaseKFold`, :py:obj:`abc.ABC`

   
   Time Series cross-validator variant with non-overlapping groups.
   Provides train/test indices to split time series data samples
   that are observed at fixed time intervals according to a
   third party provided group.
   In each split, test indices must be higher than before, and thus shuffling
   in cross validator is inappropriate.
   This cross-validation object is a variation of :class:`KFold`.
   In the kth split, it returns first k folds as train set and the
   (k+1)th fold as test set.
   The same group will not appear in two different folds (the number of
   distinct groups has to be at least equal to the number of folds).
   Note that unlike standard cross-validation methods, successive
   training sets are supersets of those that come before them.
   Read more in the :ref:`User Guide <cross_validation>`.

   :param n_splits: Number of splits. Must be at least 2.
   :type n_splits: int, default=5
   :param max_train_size: Maximum size for a single training set.
   :type max_train_size: int, default=None
   :param sort_groups: Whether to sort the order of groups. Default is True.
   :type sort_groups: bool

   .. rubric:: Examples

   >>> import numpy as np
   >>> from sandbox.model_selection import GroupTimeSeriesSplit
   >>> groups = np.array(['a', 'a', 'a', 'a', 'a', 'a',
   ...                    'b', 'b', 'b', 'b', 'b',
   ...                    'c', 'c', 'c', 'c',
   ...                    'd', 'd', 'd'])
   >>> gtss = GroupTimeSeriesSplit(n_splits=3)
   >>> for train_idx, test_idx in gtss.split(groups, groups=groups):
   ...     print("TRAIN:", train_idx, "TEST:", test_idx)
   ...     print("TRAIN GROUP:", groups[train_idx], "TEST GROUP:", groups[test_idx])
   TRAIN: [0, 1, 2, 3, 4, 5] TEST: [6, 7, 8, 9, 10]
   TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a']
   TEST GROUP: ['b' 'b' 'b' 'b' 'b']
   TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] TEST: [11, 12, 13, 14]
   TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b']
   TEST GROUP: ['c' 'c' 'c' 'c']
   TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
   TEST: [15, 16, 17]
   TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b' 'c' 'c' 'c' 'c']
   TEST GROUP: ['d' 'd' 'd']















   ..
       !! processed by numpydoc !!
   .. py:method:: split(X, y=None, groups=None)

      
      Generate indices to split data into training and test set.

      :param X: Training data, where n_samples is the number of samples
                and n_features is the number of features.
      :type X: array-like of shape (n_samples, n_features)
      :param y: Always ignored, exists for compatibility.
      :type y: array-like of shape (n_samples,)
      :param groups: Group labels for the samples used while splitting the dataset into
                     train/test set.
      :type groups: array-like of shape (n_samples,)

      :Yields: * **train** (*numpy.ndarray*) -- The training set indices for that split.
               * **test** (*numpy.ndarray*) -- The testing set indices for that split.















      ..
          !! processed by numpydoc !!


.. py:class:: PurgedGroupTimeSeriesSplit(n_splits=5, *, max_train_group_size=np.inf, max_test_group_size=np.inf, group_gap=None, sort_groups=True, verbose=False)

   Bases: :py:obj:`sklearn.model_selection._split._BaseKFold`, :py:obj:`abc.ABC`

   
   Time Series cross-validator variant with non-overlapping groups.
   Allows for a gap in groups to avoid potentially leaking info from
   train into test if the model has windowed or lag features.
   Provides train/test indices to split time series data samples
   that are observed at fixed time intervals according to a
   third party provided group.
   In each split, test indices must be higher than before, and thus shuffling
   in cross validator is inappropriate.
   This cross-validation object is a variation of :class:`KFold`.
   In the kth split, it returns first k folds as train set and the
   (k+1)th fold as test set.
   The same group will not appear in two different folds (the number of
   distinct groups has to be at least equal to the number of folds).
   Note that unlike standard cross-validation methods, successive
   training sets are supersets of those that come before them.
   Read more in the :ref:`User Guide <cross_validation>`.

   :param n_splits: Number of splits. Must be at least 2.
   :type n_splits: int, default=5
   :param max_train_group_size: Maximum group size for a single training set.
   :type max_train_group_size: int, default=Inf
   :param group_gap: Gap between train and test
   :type group_gap: int, default=None
   :param sort_groups: Whether to sort the order of groups. Default is True.
   :type sort_groups: bool
   :param max_test_group_size: We discard this number of groups from the end of each train split
   :type max_test_group_size: int, default=Inf















   ..
       !! processed by numpydoc !!
   .. py:method:: split(X, y=None, groups=None)

      
      Generate indices to split data into training and test set.

      :param X: Training data, where n_samples is the number of samples
                and n_features is the number of features.
      :type X: array-like of shape (n_samples, n_features)
      :param y: Always ignored, exists for compatibility.
      :type y: array-like of shape (n_samples,)
      :param groups: Group labels for the samples used while splitting the dataset into
                     train/test set.
      :type groups: array-like of shape (n_samples,)

      :Yields: * **train** (*ndarray*) -- The training set indices for that split.
               * **test** (*ndarray*) -- The testing set indices for that split.















      ..
          !! processed by numpydoc !!


.. py:function:: plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10)

   
   Create a sample plot for indices of a cross-validation object.
















   ..
       !! processed by numpydoc !!

