"""
The :mod:`sandbox.model_selection._search` module includes classes and
functions to fine-tune the parameters of an estimator.
"""
from abc import abstractmethod

import numpy as np
import optuna
from joblib import Parallel
from optuna.logging import DEBUG, ERROR, INFO, WARNING, set_verbosity
from sklearn.base import BaseEstimator
from sklearn.utils.fixes import delayed

from sandbox.datamodel.base import BaseData, SupervisedModelDataset
from sandbox.ensemble.boost import XGBoostRegressor
from sandbox.metrics.score import score

from ._split import KFold

__all__ = [
    "BaseOptunaSearchCV",
    "XGBoostOptunaSearchCV",
    "LightGBMOptunaStepwiseSearchCV",
]


class BaseOptunaStudyInitializer:
    """Base initializer class for study instance.

    Parameters
    ----------
    storage : {None, str}, default=None
        Database URL. If this argument is set to None, in-memory storage is used, and the
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

    study_name: {None, str}, default=None
        Study's name. If this argument is set to None, a unique name is generated automatically.
    direction: str, default=minimize
        Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for maximization.
        You can also pass the corresponding :class:`optuna.study.StudyDirection` object.
    load_if_exists: bool, default=False
        Flag to control the behavior to handle a conflict of study names.
        In the case where a study named ``study_name`` already exists in the ``storage``,
        a :class:`optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
        set to :obj:`False`. Otherwise, the creation of the study is skipped, and the existing one is returned.
    sampler : {optuna.samplers, None}, default=None
        A sampler object that implements background algorithm for value suggestion.
        If :obj:`None` is specified, :class:`optuna.samplers.TPESampler` is used.
    sampler_params : dict
        Parameters passed to the specified `optuna.samplers`.
    """

    def __init__(
        self,
        storage=None,
        study_name=None,
        direction="minimize",
        load_if_exists=False,
        sampler=None,
        **sampler_params,
    ):
        self.storage = storage
        self.study_name = study_name
        self.direction = direction
        self.load_if_exists = load_if_exists

        # setting for sampler of optuna.study.Study
        self.sampler = self.optuna_sampler(sampler, **sampler_params)

        self.create_study_params = {
            "storage": self.storage,
            "sampler": self.sampler,
            "study_name": self.study_name,
            "direction": self.direction,
            "load_if_exists": self.load_if_exists,
        }

    @staticmethod
    def optuna_sampler(sampler=None, **sampler_params):
        """Return your specified `optuna.samplers`

        Parameters
        ----------
        sampler : {optuna.samplers, None}, default=None
            A sampler object that implements background algorithm for value suggestion.
            If :obj:`None` is specified, :class:`optuna.samplers.TPESampler` is used.
        sampler_params : dict
            Parameters passed to the specified `optuna.samplers`.

        Returns
        -------
        sampler : optuna.samplers
        """
        if sampler is None:
            return optuna.samplers.TPESampler(**sampler_params)
        else:
            return sampler

    def create_study(self):
        """Create `optuna.study.Study` instance.

        Returns
        -------
        study : optuna.study.Study
        """
        return optuna.create_study(**self.create_study_params)

    def params(self, **kwargs):
        """The parameter search space which should be implemented in the subclass which is inheritance to this class.

        Parameters
        ----------
        kwargs : dict

        Returns
        -------
        The parameter space to be searched.
        """
        raise NotImplementedError


class BaseOptunaSearchCV(BaseOptunaStudyInitializer, BaseEstimator):
    """Base class for hyperparameter search using `Optuna`.

    Examples
    --------
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


    Parameters
    ----------
    estimator : estimator object
        The estimator class compatible with scikit-learn
    scoring : str
        Which metric to use in evaluating the precision of cross validated estimator using `Optuna`.
    cv : {None, int, cross-validation generator or and iterable}, default=None
        Determines the cross-validation splitting strategy. Possible inputs for cv are:

            - None, to use the default 5-fold cross validation,
            - integer, to specify the number of folds in a `KFold`,
            - CV splitter,
            - An iterable yielding (train, test) splits as arrays of indices.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    pre_dispatch : {int, str}, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
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

    storage : {None, str}, default=None
        Database URL. If this argument is set to None, in-memory storage is used, and the
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

    study_name: {None, str}, default=None
        Study's name. If this argument is set to None, a unique name is generated automatically.
    direction: str, default=minimize
        Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for maximization.
        You can also pass the corresponding :class:`optuna.study.StudyDirection` object.
    load_if_exists: bool, default=False
        Flag to control the behavior to handle a conflict of study names.
        In the case where a study named ``study_name`` already exists in the ``storage``,
        a :class:`optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
        set to :obj:`False`. Otherwise, the creation of the study is skipped, and the existing one is returned.
    sampler : {optuna.samplers, None}, default=None
        A sampler object that implements background algorithm for value suggestion.
        If :obj:`None` is specified, :class:`optuna.samplers.TPESampler` is used.
    sampler_seed : int, default=42
        Seed for random number generator.
    """

    def __init__(
        self,
        estimator,
        scoring,
        cv=None,
        n_jobs=None,
        pre_dispatch="2*n_jobs",
        storage=None,
        study_name=None,
        direction="minimize",
        load_if_exists=False,
        sampler=None,
        sampler_seed=42,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch

        if cv is None:
            self.cv = KFold(n_splits=5, shuffle=True)
        elif isinstance(cv, int):
            self.cv = KFold(n_splits=int(cv), shuffle=True)
        else:
            self.cv = cv

        # initialize by BaseOptunaStudyInitializer.
        super(BaseOptunaSearchCV, self).__init__(
            storage=storage,
            study_name=study_name,
            direction=direction,
            load_if_exists=load_if_exists,
            sampler=sampler,
            seed=sampler_seed,
        )
        self.sampler_seed = sampler_seed
        self._study = self.create_study()

    @property
    def study(self):
        """This has the all results of searching hyperparameter in the instance.

        See Also
        --------
        optuna.study.Study : A study corresponds to an optimization task, i.e., a set of trials.
        """
        return self._study

    @abstractmethod
    def params(self, trial):
        """This returns the hyperparameter search space for your defined estimator.

        Examples
        --------
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

        See Also
        --------
        optuna.trial.Trial.suggest_categorical : Suggest a value for the categorical parameter.
        optuna.trial.Trial.suggest_discrete_uniform : Suggest a value for the discrete parameter.
        optuna.trial.Trial.suggest_float : Suggest a value for the floating point parameter.
        optuna.trial.Trial.suggest_int : Suggest a value for the integer parameter.
        optuna.trial.Trial.suggest_loguniform : Suggest a value for the continuous parameter.
        optuna.trial.Trial.suggest_uniform : Suggest a value for the continuous parameter.
        """
        raise NotImplementedError

    def _objective(self, trial, X, y, groups=None, **fit_params):
        if groups is not None:
            groups = BaseData(groups).values

        base_estimator = self.estimator.set_params(**self.params(trial))

        def evaluate(estimator, X, y, scoring, train, test, **fit_params):
            # ToDo: dask形式の場合を検討.
            data = SupervisedModelDataset(X, y)
            X_train = data.X.to_pandas().iloc[train]
            y_train = data.y.to_pandas().iloc[train]
            X_test = data.X.to_pandas().iloc[test]
            y_test = data.y.to_pandas().iloc[test]

            if ("early_stopping_rounds" in fit_params.keys()) or (
                hasattr(estimator, "early_stopping_rounds")
                and getattr(estimator, "early_stopping_rounds", None) is not None
            ):
                fit_params["eval_set"] = [(X_test, y_test)]

            estimator.fit(
                X_train,
                y_train,
                **fit_params,
            )

            try:
                ret = estimator.score(X_test, y_test, scoring=scoring)
            except TypeError:
                y_pred = estimator.predict(X_test)
                ret = score(y_test, y_pred, scoring=scoring)

            return ret

        out = []
        if self.n_jobs is None:
            for split_idx, (train, test) in enumerate(
                self.cv.split(X, y, groups=groups)
            ):
                ret = evaluate(
                    estimator=base_estimator,
                    X=X,
                    y=y,
                    scoring=self.scoring,
                    train=train,
                    test=test,
                    **fit_params,
                )
                out.append(ret)
        else:
            parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)
            with parallel:
                out = parallel(
                    delayed(evaluate)(
                        estimator=base_estimator,
                        X=X,
                        y=y,
                        scoring=self.scoring,
                        train=train,
                        test=test,
                        **fit_params,
                    )
                    for split_idx, (train, test) in enumerate(
                        self.cv.split(X, y, groups=groups)
                    )
                )

        return np.average(out)

    def fit(
        self,
        X,
        y,
        groups=None,
        n_trials=10,
        show_progress_bar=False,
        optuna_verbosity=1,
        **fit_params,
    ):
        """Execute hyperparameter tuning.

        Parameters
        ----------
        X :
            The input samples.
        y :
            Target values (strings or integers in classification, real numbers in regression).
            For classification, labels must correspond to classes.
        groups :
            Group labels for the samples used while splitting the dataset into train/test set.
        n_trials : int
            The number of trials.
        show_progress_bar : bool, default=False
            Flag to show progress bars or not. To disable progress bar, set this :obj:`False`.
            Currently, progress bar is experimental feature and disabled when ``n_jobs`` :math:`\\ne 1`.
        optuna_verbosity : int, default=1
            The degree of verbosity in `Optuna` optimization. Valid values are 0 (silent) - 3 (debug).
        fit_params : dict
            Parameters passed to the `fit` method of the estimator.
        """
        if optuna_verbosity == 0:
            OPTUNA_VERBOSE_LEVEL = ERROR
        elif optuna_verbosity == 1:
            OPTUNA_VERBOSE_LEVEL = WARNING
        elif optuna_verbosity == 2:
            OPTUNA_VERBOSE_LEVEL = INFO
        elif optuna_verbosity == 3:
            OPTUNA_VERBOSE_LEVEL = DEBUG
        else:
            msg = "Specified optuna_verbosity must be [0, 1, 2, 3], not {}".format(
                optuna_verbosity
            )
            raise ValueError(msg)

        set_verbosity(OPTUNA_VERBOSE_LEVEL)

        self._study.optimize(
            lambda trial: self._objective(
                trial,
                X=X,
                y=y,
                groups=groups,
                **fit_params,
            ),
            n_trials=n_trials,
            show_progress_bar=show_progress_bar,
        )
        return self


class XGBoostOptunaSearchCV(BaseOptunaSearchCV):
    """Hyperparameter search for XGBoost with cross-validation

    Parameters
    ----------
    n_estimators : int, default=1000
        Number of gradient boosted trees. Equivalent to number of boosting rounds.
    scoring : str, default="mse"
        Which metric to use in evaluating the precision of cross validated estimator.
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
    verbosity : int or None, default=None
        The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
    cv : {None, int, cross-validation generator or and iterable}, default=None
        Determines the cross-validation splitting strategy. Possible inputs for cv are:

            - None, to use the default 5-fold cross validation,
            - integer, to specify the number of folds in a `KFold`,
            - CV splitter,
            - An iterable yielding (train, test) splits as arrays of indices.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    pre_dispatch : {int, str}, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
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
    storage : {None, str}, default=None
        Database URL. If this argument is set to None, in-memory storage is used, and the
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

    study_name: {None, str}, default=None
        Study's name. If this argument is set to None, a unique name is generated automatically.
    direction: str, default=minimize
        Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for maximization.
        You can also pass the corresponding :class:`optuna.study.StudyDirection` object.
    load_if_exists: bool, default=False
        Flag to control the behavior to handle a conflict of study names.
        In the case where a study named ``study_name`` already exists in the ``storage``,
        a :class:`optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
        set to :obj:`False`. Otherwise, the creation of the study is skipped, and the existing one is returned.
    sampler : {optuna.samplers, None}, default=None
        A sampler object that implements background algorithm for value suggestion.
        If :obj:`None` is specified, :class:`optuna.samplers.TPESampler` is used.
    sampler_seed : int, default=42
        Seed for random number generator.
    """

    def __init__(
        self,
        n_estimators=1000,
        scoring="mse",
        early_stopping_rounds=None,
        verbosity=1,
        cv=None,
        n_jobs=None,
        pre_dispatch="2*n_jobs",
        storage=None,
        study_name=None,
        direction="minimize",
        load_if_exists=False,
        sampler=None,
        sampler_seed=42,
    ):
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.verbosity = verbosity
        estimator = XGBoostRegressor(
            n_estimators=self.n_estimators,
            early_stopping_rounds=self.early_stopping_rounds,
            verbosity=self.verbosity,
            random_state=2020,
        )
        super(XGBoostOptunaSearchCV, self).__init__(
            estimator=estimator,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            storage=storage,
            study_name=study_name,
            direction=direction,
            load_if_exists=load_if_exists,
            sampler=sampler,
            sampler_seed=sampler_seed,
        )

    def params(self, trial):
        """

        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        The following search space on hyperparameters.
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
                - Suggest a value for the categorical parameter: :math:`\\{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0\\}`
            - subsample :
                - Subsample ratio of the training instance.
                - Suggest a value for the categorical parameter: :math:`\\{0.4, 0.5, 0.6, 0.7, 0.8, 1.0\\}`
            - learning_rate :
                - Boosting learning rate (xgb's "eta")
                - Suggest a value for the categorical parameter:
                :math:`\\{0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02\\}`
            - max_depth :
                - Maximum tree depth for base learners.
                - Suggest a value for the categorical parameter: :math:`\\{5, 7, 9, 11, 13, 15, 17\\}`
            - min_child_weight :
                - Minimum sum of instance weight(hessian) needed in a child.
                - The value is sampled from the integers in :math:`[1, 300]`
        """
        return {
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
            "gamma": trial.suggest_int("gamma", 0, 20),
            "colsample_bytree": trial.suggest_categorical(
                "colsample_bytree", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            "subsample": trial.suggest_categorical(
                "subsample", [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            ),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth", [5, 7, 9, 11, 13, 15, 17]
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
        }

    def fit(
        self,
        X,
        y,
        groups=None,
        n_trials=10,
        show_progress_bar=False,
        eval_verbosity=1,
        optuna_verbosity=1,
        **fit_params,
    ):
        """Execute hyperparameter tuning.

        Parameters
        ----------
        X :
            The input samples.
        y :
            Target values (strings or integers in classification, real numbers in regression).
            For classification, labels must correspond to classes.
        groups :
            Group labels for the samples used while splitting the dataset into train/test set.
        n_trials : int
            The number of trials.
        show_progress_bar : bool, default=False
            Flag to show progress bars or not. To disable progress bar, set this :obj:`False`.
            Currently, progress bar is experimental feature and disabled when ``n_jobs`` :math:`\\ne 1`.
        eval_verbosity : int, default=1
            The degree of verbosity in cross-validation evaluation. Valid values are 0 (silent) - 3 (debug).
        optuna_verbosity : int, default=1
            The degree of verbosity in `Optuna` optimization. Valid values are 0 (silent) - 3 (debug).
        fit_params : dict
            Parameters passed to the `fit` method of the estimator of
            :class:`~sandbox.ensemble.boost.XGBoostRegressor`.
        """
        fit_params["verbose"] = eval_verbosity
        super(XGBoostOptunaSearchCV, self).fit(
            X=X,
            y=y,
            groups=groups,
            n_trials=n_trials,
            show_progress_bar=show_progress_bar,
            optuna_verbosity=optuna_verbosity,
            **fit_params,
        )
        return self


class LightGBMOptunaStepwiseSearchCV(BaseOptunaStudyInitializer, BaseEstimator):
    """Hyperparameter stepwise search for LightGBM with cross-validation

    Parameters
    ----------
    n_estimators : int, default=1000
        Number of gradient boosted trees. Equivalent to number of boosting rounds.
    boosting_type : str, default='gbdt'
        - 'gbdt', traditional Gradient Boosting Decision Tree.
        - 'dart', Dropouts meet Multiple Additive Regression Trees.
        - 'goss', Gradient-based One-Side Sampling.
        - 'rf', Random Forest.
    objective : {str, callable, None}, default="regression"
        Specify the learning task and the corresponding learning objective or
        a custom objective function to be used (see note below).
        Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass'
        for LGBMClassifier, 'lambdarank' for LGBMRanker.
    metric : {str, callable, None}, default="rmse"
        Metric(s) to be evaluated on the evaluation set(s).
    early_stopping_rounds : int
         Activates early stopping.
    random_state : int
        Random number seed.
    cv : {None, int, cross-validation generator or and iterable}, default=None
        Determines the cross-validation splitting strategy. Possible inputs for cv are:

            - None, to use the default 5-fold cross validation,
            - integer, to specify the number of folds in a `KFold`,
            - CV splitter,
            - An iterable yielding (train, test) splits as arrays of indices.

    storage : {None, str}, default=None
        Database URL. If this argument is set to None, in-memory storage is used, and the
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

    study_name: {None, str}, default=None
        Study's name. If this argument is set to None, a unique name is generated automatically.
    direction: str, default=minimize
        Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for maximization.
        You can also pass the corresponding :class:`optuna.study.StudyDirection` object.
    load_if_exists: bool, default=False
        Flag to control the behavior to handle a conflict of study names.
        In the case where a study named ``study_name`` already exists in the ``storage``,
        a :class:`optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
        set to :obj:`False`. Otherwise, the creation of the study is skipped, and the existing one is returned.
    sampler : {optuna.samplers, None}, default=None
        A sampler object that implements background algorithm for value suggestion.
        If :obj:`None` is specified, :class:`optuna.samplers.TPESampler` is used.
    sampler_seed : int, default=42
        Seed for random number generator.

    See Also
    --------
    optuna.integration.lightgbm.LightGBMTunerCV : Hyperparameter tuner for LightGBM with cross-validation.
    """

    def __init__(
        self,
        n_estimators=1000,
        boosting_type="gbdt",
        objective="regression",
        metric="rmse",
        early_stopping_rounds=100,
        random_state=42,
        cv=None,
        storage=None,
        study_name=None,
        direction="minimize",
        load_if_exists=False,
        sampler=None,
        sampler_seed=42,
    ):
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds

        self.boosting_type = boosting_type
        self.objective = objective
        self.metric = metric
        self.random_state = random_state
        self.tuner_params = {
            "boosting_type": boosting_type,
            "objective": objective,
            "metric": metric,
            "random_state": random_state,
        }

        if cv is None:
            self.cv = KFold(n_splits=5, shuffle=True)
        elif isinstance(cv, int):
            self.cv = KFold(n_splits=int(cv), shuffle=True)
        else:
            self.cv = cv

        # initialize by BaseOptunaStudyInitializer.
        super(LightGBMOptunaStepwiseSearchCV, self).__init__(
            storage=storage,
            study_name=study_name,
            direction=direction,
            load_if_exists=load_if_exists,
            sampler=sampler,
            seed=sampler_seed,
        )
        self.sampler_seed = sampler_seed
        self.tuner = None

    @property
    def study(self):
        """This has the all results of searching hyperparameter in the instance.

        See Also
        --------
        optuna.study.Study : A study corresponds to an optimization task, i.e., a set of trials.
        """
        if self.tuner is None:
            return None
        else:
            return self.tuner.study

    def _define_tuner(self, X, y, groups=None, eval_verbosity=1, **fit_params):
        from lightgbm import early_stopping, log_evaluation
        from optuna.integration.lightgbm import Dataset, LightGBMTunerCV

        if groups is not None:
            groups = BaseData(groups).values

        # ToDo: dask形式の場合を検討.
        data = SupervisedModelDataset(X, y)
        X = data.X.to_pandas()
        y = data.y.to_pandas()
        train_set = Dataset(X, y)
        self.tuner = LightGBMTunerCV(
            self.tuner_params,
            train_set,
            num_boost_round=self.n_estimators,
            folds=list(self.cv.split(X, y, groups=groups)),
            study=self.create_study(),
            callbacks=[
                early_stopping(
                    stopping_rounds=self.early_stopping_rounds, verbose=False
                ),
                log_evaluation(eval_verbosity),
            ],
            **fit_params,
        )

    def fit(
        self,
        X,
        y,
        groups=None,
        show_progress_bar=False,
        eval_verbosity=1,
        optuna_verbosity=1,
        **fit_params,
    ):
        """Execute hyperparameter tuning.

        Parameters
        ----------
        X :
            The input samples.
        y :
            Target values (strings or integers in classification, real numbers in regression).
            For classification, labels must correspond to classes.
        groups :
            Group labels for the samples used while splitting the dataset into train/test set.
        show_progress_bar : bool, default=False
            Flag to show progress bars or not. To disable progress bar, set this :obj:`False`.
            Currently, progress bar is experimental feature and disabled when ``n_jobs`` :math:`\\ne 1`.
        eval_verbosity : int, default=1
            The degree of verbosity in cross-validation evaluation. Valid values are 0 (silent) - 3 (debug).
        optuna_verbosity : int, default=1
            The degree of verbosity in `Optuna` optimization. Valid values are 0 (silent) - 3 (debug).
        fit_params : dict
            Parameters passed to the `fit` method of the estimator of
            :class:`~sandbox.ensemble.boost.XGBoostRegressor`.
        """
        if optuna_verbosity == 0:
            OPTUNA_VERBOSE_LEVEL = ERROR
        elif optuna_verbosity == 1:
            OPTUNA_VERBOSE_LEVEL = WARNING
        elif optuna_verbosity == 2:
            OPTUNA_VERBOSE_LEVEL = INFO
        elif optuna_verbosity == 3:
            OPTUNA_VERBOSE_LEVEL = DEBUG
        else:
            msg = "Specified optuna_verbosity must be [0, 1, 2, 3], not {}".format(
                optuna_verbosity
            )
            raise ValueError(msg)

        set_verbosity(OPTUNA_VERBOSE_LEVEL)

        fit_params["show_progress_bar"] = show_progress_bar
        self._define_tuner(X, y, groups, eval_verbosity, **fit_params)
        self.tuner.run()
        return self
