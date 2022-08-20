:orphan:

:py:mod:`sandbox.model_selection._search`
=========================================

.. py:module:: sandbox.model_selection._search

.. autoapi-nested-parse::

   The :mod:`sandbox.model_selection._search` module includes classes and
   functions to fine-tune the parameters of an estimator.

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.model_selection._search.BaseOptunaSearchCV
   sandbox.model_selection._search.XGBoostOptunaSearchCV
   sandbox.model_selection._search.LightGBMOptunaStepwiseSearchCV




.. py:class:: BaseOptunaSearchCV(estimator, scoring, cv=None, n_jobs=None, pre_dispatch='2*n_jobs', storage=None, study_name=None, direction='minimize', load_if_exists=False)

   Bases: :py:obj:`sklearn.base.BaseEstimator`

   
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

   .. py:method:: fit(X, y, groups=None, n_trials=10, **fit_params)

      
      Execute hyperparameter tuning.

      :param X: The input samples.
      :param y: Target values (strings or integers in classification, real numbers in regression).
                For classification, labels must correspond to classes.
      :param groups: Group labels for the samples used while splitting the dataset into train/test set.
      :param n_trials: The number of trials.
      :type n_trials: int
      :param fit_params: Parameters passed to the `fit` method of the estimator.
      :type fit_params: dict















      ..
          !! processed by numpydoc !!


.. py:class:: XGBoostOptunaSearchCV(n_estimators=1000, scoring='mse', cv=None, n_jobs=None, pre_dispatch='2*n_jobs', storage=None, study_name=None, direction='minimize', load_if_exists=False)

   Bases: :py:obj:`BaseOptunaSearchCV`

   
   Hyperparameter search for XGBoost with cross-validation

   :param n_estimators: Number of gradient boosted trees. Equivalent to number of boosting rounds.
   :type n_estimators: int, default=1000
   :param scoring: Which metric to use in evaluating the precision of cross validated estimator using `Optuna`.
   :type scoring: str, default="mse"
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
                - random_state :
                    - Random number seed.
                    - Seed is fixed as 2020.
                - min_child_weight :
                    - Minimum sum of instance weight(hessian) needed in a child.
                    - The value is sampled from the integers in :math:`[1, 300]`
      :rtype: The following search space on hyperparameters.















      ..
          !! processed by numpydoc !!

   .. py:method:: fit(X, y, groups=None, n_trials=10, early_stopping_rounds=100, **fit_params)

      
      Execute hyperparameter tuning.

      :param X: The input samples.
      :param y: Target values (strings or integers in classification, real numbers in regression).
                For classification, labels must correspond to classes.
      :param groups: Group labels for the samples used while splitting the dataset into train/test set.
      :param n_trials: The number of trials.
      :type n_trials: int
      :param early_stopping_rounds: Activates early stopping.
      :type early_stopping_rounds: int
      :param fit_params: Parameters passed to the `fit` method of the estimator of :class:`~sandbox.ensemble.boost.XGBoostRegressor`.
      :type fit_params: dict















      ..
          !! processed by numpydoc !!


.. py:class:: LightGBMOptunaStepwiseSearchCV(n_estimators=1000, boosting_type='gbdt', objective='regression', metric='rmse', early_stopping_rounds=100, random_state=2022, cv=None, storage=None, study_name=None, direction='minimize', load_if_exists=False)

   Bases: :py:obj:`sklearn.base.BaseEstimator`

   
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

   .. py:method:: fit(X, y, groups=None)

      
      Execute hyperparameter tuning.

      :param X: The input samples.
      :param y: Target values (strings or integers in classification, real numbers in regression).
                For classification, labels must correspond to classes.
      :param groups: Group labels for the samples used while splitting the dataset into train/test set.















      ..
          !! processed by numpydoc !!


