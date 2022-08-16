:orphan:

:py:mod:`sandbox.model_selection._split`
========================================

.. py:module:: sandbox.model_selection._split

.. autoapi-nested-parse::

   The :mod:`sandbox.model_selection._split` module includes classes and
   functions to split the data based on a preset strategy.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.model_selection._split.GroupTimeSeriesSplit
   sandbox.model_selection._split.PurgedGroupTimeSeriesSplit



Functions
~~~~~~~~~

.. autoapisummary::

   sandbox.model_selection._split.plot_cv_indices



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



.. py:function:: plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10)

   Create a sample plot for indices of a cross-validation object.


