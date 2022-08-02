:py:mod:`sandbox.graphics.ts_grapher`
=====================================

.. py:module:: sandbox.graphics.ts_grapher

.. autoapi-nested-parse::

   The grapher module on time series modeling.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.graphics.ts_grapher.TimeSeriesGrapher
   sandbox.graphics.ts_grapher.TimeSeriesGrapherMixin




.. py:class:: TimeSeriesGrapher(model)

   The time series graphing class.

   :param model: The time series modeling instance.
   :type model: model

   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: plot_prediction(X=None, y=None, colors=('lightblue', 'navy', 'red'), linestyles=(None, 'dotted', 'solid'), labels=('Observed', 'Fitted', 'Predicted'), title='Prediction Plot', fig=None, ax=None, figsize=(16, 4))

      Plot in-sample and out-of-sample prediction.

      :param X: Design matrix expressing the regression dummies or variables in
                the period to be predicted. If no regression is defined in the model,
                the index expressing the period or the period steps to be predicted
                must be set.
      :type X: {array-like, int, None}, optional
      :param y: True values for `X`.
      :type y: {array-like, None}, optional
      :param colors: Graph colors. The first element is the color of bar plot of observed data.
                     The second is the one of line plot of fitted values. The last is line plot of
                     predicted values.
      :type colors: tuple[str, str, str], optional
      :param linestyles: Graph line style. The observed data is expressed in bar plot, and the line style
                         information is unnecessary, then the first element is None as default. The second
                         is the line style of line plot of fitted values. The last is line plot of
                         predicted values.
      :type linestyles: tuple[None, str, str], optional
      :param labels: Label information on observed, fitted and predicted values graph. If label expression
                     is omitted, set ``labels = None``.
      :type labels: {tuple[str, str, str], None}, optional
      :param title: Graph title. If title expression is omitted in the graph, set ``title = None``.
      :type title: {str, None}, optional
      :param fig: The matplotlib top container for graphs.
      :type fig: matplot.figure.Figure, optional
      :param ax: The axes object.
      :type ax: matplot.axes.Axes, optional
      :param figsize: Figure dimension (width, height) in inches.
      :type figsize: tuple[float, float], optional

      :returns: * **fig** (*matplot.figure.Figure*) -- The matplotlib container including prediction plot graph.
                * **ax** (*matplot.axes.Axes*) -- The axes object of prediction plot graph.


   .. py:method:: plot_components(X=None, y=None, figsize=None)

      Plot in-sample and out-of-sample component predictions.

      :param X: Design matrix expressing the regression dummies or variables in
                the period to be predicted. If no regression is defined in the model,
                the index expressing the period or the period steps to be predicted
                must be set.
      :type X: {array-like, int, None}, optional
      :param y: True values for `X`.
      :type y: {array-like, None}, optional
      :param figsize: Figure dimension (width, height) in inches. As default, ``figsize``
                      is calculated as follows: (16, 3 * (number of components + 1)).
      :type figsize: {tuple[float, float], None}, optional

      :returns: * **fig** (*matplot.figure.Figure*) -- The matplotlib container including components plot graph.
                * **ax** (*matplot.axes.Axes*) -- The axes object of components plot graph.



.. py:class:: TimeSeriesGrapherMixin

   Mixin class for graphing of time series modeling.

   .. seealso:: :obj:`sandbox.graphics.ts_grapher.TimeSeriesGrapher`

   .. rubric:: Examples

   >>> from sandbox.tsa.ssm import LinearGaussianStateSpaceModel
   >>> model = LinearGaussianStateSpaceModel(
   >>>     level=True,
   >>>     trend=True,
   >>>     freq_seasonal=[{"period": 7, "harmonics": 2}, {"period": 30, "harmonics": 4}],
   >>>     mle_regression=False,
   >>> )
   >>> model.fit(X_train, y_train)
   >>> model.graph.plot_prediction(X_test, y_test)
   >>> model.graph.plot_components(X_test, y_test)

   .. py:method:: graph()
      :property:



