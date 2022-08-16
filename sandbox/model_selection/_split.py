"""
The :mod:`sandbox.model_selection._split` module includes classes and
functions to split the data based on a preset strategy.
"""
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import (
    BaseCrossValidator,
    BaseShuffleSplit,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    PredefinedSplit,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    check_cv,
    train_test_split,
)
from sklearn.model_selection._split import _BaseKFold, _num_samples, indexable
from sklearn.utils.validation import _deprecate_positional_args

__all__ = [
    "BaseCrossValidator",
    "BaseShuffleSplit",
    "GroupKFold",
    "GroupShuffleSplit",
    "GroupTimeSeriesSplit",
    "KFold",
    "LeaveOneGroupOut",
    "LeaveOneOut",
    "LeavePGroupsOut",
    "LeavePOut",
    "plot_cv_indices",
    "PredefinedSplit",
    "PurgedGroupTimeSeriesSplit",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "ShuffleSplit",
    "StratifiedGroupKFold",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
    "TimeSeriesSplit",
    "check_cv",
    "train_test_split",
]


class GroupTimeSeriesSplit(_BaseKFold, ABC):
    """Time Series cross-validator variant with non-overlapping groups.
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

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum size for a single training set.
    sort_groups : bool
        Whether to sort the order of groups. Default is True.

    Examples
    --------
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
    """

    @_deprecate_positional_args
    def __init__(self, n_splits=5, *, max_train_size=None, sort_groups=True):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.sort_groups = sort_groups

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : numpy.ndarray
            The training set indices for that split.
        test : numpy.ndarray
            The testing set indices for that split.
        """
        if groups is None:
            msg = "The 'groups' parameter should not be None"
            raise ValueError(msg)

        groups = _maybe_convert_groups(groups)

        X, y, groups = indexable(X, y, groups)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_dict = {}

        if self.sort_groups:
            unique_groups = np.unique(groups)
        else:
            u, ind = np.unique(groups, return_index=True)
            unique_groups = u[np.argsort(ind)]

        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if groups[idx] in group_dict:
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            msg = (
                "Cannot have number of folds={0} greater than the number of groups={1}"
            )
            raise ValueError(msg.format(n_folds, n_groups))

        group_test_size = n_groups // n_folds
        group_test_starts = range(
            n_groups - n_splits * group_test_size, n_groups, group_test_size
        )
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []
            for train_group_idx in unique_groups[:group_test_start]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(
                    np.unique(
                        np.concatenate((train_array, train_array_tmp)), axis=None
                    ),
                    axis=None,
                )
            train_end = train_array.size
            if self.max_train_size and self.max_train_size < train_end:
                train_array = train_array[train_end - self.max_train_size : train_end]
            for test_group_idx in unique_groups[
                group_test_start : group_test_start + group_test_size
            ]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(
                    np.unique(np.concatenate((test_array, test_array_tmp)), axis=None),
                    axis=None,
                )
            yield [int(i) for i in train_array], [int(i) for i in test_array]


class PurgedGroupTimeSeriesSplit(_BaseKFold, ABC):
    """Time Series cross-validator variant with non-overlapping groups.
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

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    sort_groups : bool
        Whether to sort the order of groups. Default is True.
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(
        self,
        n_splits=5,
        *,
        max_train_group_size=np.inf,
        max_test_group_size=np.inf,
        group_gap=None,
        sort_groups=True,
        verbose=False
    ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.sort_groups = sort_groups
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            msg = "The 'groups' parameter should not be None"
            raise ValueError(msg)

        groups = _maybe_convert_groups(groups)

        X, y, groups = indexable(X, y, groups)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}

        if self.sort_groups:
            unique_groups = np.unique(groups)
        else:
            u, ind = np.unique(groups, return_index=True)
            unique_groups = u[np.argsort(ind)]

        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if groups[idx] in group_dict:
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            msg = (
                "Cannot have number of folds={0} greater than the number of groups={1}"
            )
            raise ValueError(msg.format(n_folds, n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(
            n_groups - n_splits * group_test_size, n_groups, group_test_size
        )
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[
                group_st : (group_test_start - group_gap)
            ]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(
                    np.unique(
                        np.concatenate((train_array, train_array_tmp)), axis=None
                    ),
                    axis=None,
                )

            for test_group_idx in unique_groups[
                group_test_start : group_test_start + group_test_size
            ]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(
                    np.unique(np.concatenate((test_array, test_array_tmp)), axis=None),
                    axis=None,
                )
            test_array = test_array[group_gap:]
            if self.verbose > 0:
                pass
            yield [int(i) for i in train_array], [int(i) for i in test_array]


def _maybe_convert_groups(groups=None):
    """pandasフォーマットのTimestampはnumpyと互換性があるが, 逆の互換性がないため
    pandas は numpy 形式に全て変更する.
    """
    if isinstance(groups, (pd.Series, pd.Index)):
        return groups.values
    if isinstance(groups, pd.DataFrame):
        return groups[groups.columns[0]].values
    return groups


def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    cmap_cv = plt.cm.coolwarm

    jet = plt.cm.get_cmap("jet", 256)
    seq = np.linspace(0, 1, 256)
    cmap_data = ListedColormap(jet(seq))

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=plt.cm.Set3
    )
    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["target", "day"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, len(y)],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax
