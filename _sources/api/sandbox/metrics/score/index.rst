:py:mod:`sandbox.metrics.score`
===============================

.. py:module:: sandbox.metrics.score


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   sandbox.metrics.score.mean_absolute_percentage_error
   sandbox.metrics.score.mape_score
   sandbox.metrics.score.r2_score



.. py:function:: mean_absolute_percentage_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')

   Mean absolute percentage error (MAPE) regression loss from ``scikit-learn``

   Note here that the output is not a percentage in the range [0, 100]
   and a value of 100 does not mean 100% but 1e2. Furthermore, the output
   can be arbitrarily high when `y_true` is small (which is specific to the
   metric) or when `abs(y_true - y_pred)` is large (which is common for most
   regression metrics).

   :param y_true: Ground truth (correct) target values.
   :type y_true: array-like
   :param y_pred: Estimated target values.
   :type y_pred: array-like
   :param sample_weight: Sample weights.
   :type sample_weight: {array-like}, optional
   :param multioutput: Defines aggregating of multiple output values.

                       Array-like value defines weights used to average errors.

                       If input is list then the shape must be (n_outputs,).

                       'raw_values' :
                           Returns a full set of errors in case of multioutput input.
                       'uniform_average' :
                           Errors of all outputs are averaged with uniform weight.
   :type multioutput: {str{'raw_values', 'uniform_average'}, array-like}

   :returns: **loss** -- If multioutput is 'raw_values', then mean absolute percentage error
             is returned for each output separately.

             If multioutput is 'uniform_average' or an ndarray of weights, then the
             weighted average of all output errors is returned.

             MAPE output is non-negative floating point. The best value is 0.0.
             But note that bad predictions can lead to arbitrarily large
             MAPE values, especially if some `y_true` values are very close to zero.

             Note that we return a large value instead of `inf` when `y_true` is zero.
   :rtype: {float, ndarray[float]}

   .. rubric:: Examples

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


.. py:function:: mape_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average')

   1 - Mean absolute percentage error (MAPE) regression.

   :param y_true: Ground truth (correct) target values.
   :type y_true: array-like
   :param y_pred: Estimated target values.
   :type y_pred: array-like
   :param sample_weight: Sample weights.
   :type sample_weight: array-like, optional
   :param multioutput: Defines aggregating of multiple output values.

                       Array-like value defines weights used to average errors.

                       If input is list then the shape must be (n_outputs,).

                       'raw_values' :
                           Returns a full set of errors in case of multioutput input.
                       'uniform_average' :
                           Errors of all outputs are averaged with uniform weight.
   :type multioutput: {str{'raw_values', 'uniform_average'}, -like}

   :returns: **z** -- 1 - `mean_absolute_percentage_error`
   :rtype: {float, ndarray[float]}


.. py:function:: r2_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average', force_finite=True)

   :math:`R^2` (coefficient of determination) regression score function from ``scikit-learn``

   Best possible score is 1.0 and it can be negative (because the
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

   :param y_true: Ground truth (correct) target values.
   :type y_true: array-like
   :param y_pred: Estimated target values.
   :type y_pred: array-like
   :param sample_weight: Sample weights.
   :type sample_weight: array-like, optional
   :param multioutput: Defines aggregating of multiple output scores.

                       Array-like value defines weights used to average scores.

                       Default is "uniform_average".

                       'raw_values' :
                           Returns a full set of scores in case of multioutput input.
                       'uniform_average' :
                           Scores of all outputs are averaged with uniform weight.
                       'variance_weighted' :
                           Scores of all outputs are averaged, weighted by the variances
                           of each individual output.
   :type multioutput: {str{'raw_values', 'uniform_average', 'variance_weighted'}, array-like}, optional
   :param force_finite: Flag indicating if ``NaN`` and ``-Inf`` scores resulting from constant
                        data should be replaced with real numbers (``1.0`` if prediction is
                        perfect, ``0.0`` otherwise). Default is ``True``, a convenient setting
                        for hyperparameters' search procedures (e.g. grid search
                        cross-validation).
   :type force_finite: bool, optional

   :returns: **z** -- The :math:`R^2` score or ndarray of scores if 'multioutput' is
             'raw_values'.
   :rtype: float or ndarray of floats

   .. rubric:: Notes

   This is not a symmetric function.

   Unlike most other scores, :math:`R^2` score may be negative (it need not
   actually be the square of a quantity R).

   This metric is not well-defined for single samples and will return a NaN
   value if n_samples is less than two.

   .. rubric:: Examples

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


