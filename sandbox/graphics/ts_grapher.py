"""The grapher module on time series modeling."""
from sandbox.datamodel.base import get_1d_arr
from sandbox.graphics.utils import create_mpl_fig
from sandbox.utils.tools import Bunch


class TimeSeriesGrapher:
    """The time series graphing class.

    Parameters
    ----------
    model : model
        The time series modeling instance.
    """

    def __init__(self, model):
        self.model = model

    def __repr__(self):
        return "{} ({})".format(self.__class__.__name__, self.model)

    def plot_prediction(
        self,
        X=None,
        y=None,
        colors=("lightblue", "navy", "red"),
        linestyles=(None, "dotted", "solid"),
        labels=("Observed", "Fitted", "Predicted"),
        title="Prediction Plot",
        fig=None,
        ax=None,
        figsize=(16, 4),
    ):
        """Plot in-sample and out-of-sample prediction.

        Parameters
        ----------
        X : {array-like, int, None}, optional
            Design matrix expressing the regression dummies or variables in
            the period to be predicted. If no regression is defined in the model,
            the index expressing the period or the period steps to be predicted
            must be set.
        y : {array-like, None}, optional
            True values for `X`.
        colors : tuple[str, str, str], optional
            Graph colors. The first element is the color of bar plot of observed data.
            The second is the one of line plot of fitted values. The last is line plot of
            predicted values.
        linestyles : tuple[None, str, str], optional
            Graph line style. The observed data is expressed in bar plot, and the line style
            information is unnecessary, then the first element is None as default. The second
            is the line style of line plot of fitted values. The last is line plot of
            predicted values.
        labels : {tuple[str, str, str], None}, optional
            Label information on observed, fitted and predicted values graph. If label expression
            is omitted, set ``labels = None``.
        title : {str, None}, optional
            Graph title. If title expression is omitted in the graph, set ``title = None``.
        fig : matplot.figure.Figure, optional
            The matplotlib top container for graphs.
        ax : matplot.axes.Axes, optional
            The axes object.
        figsize : tuple[float, float], optional
            Figure dimension (width, height) in inches.

        Returns
        -------
        fig : matplot.figure.Figure
            The matplotlib container including prediction plot graph.
        ax : matplot.axes.Axes
            The axes object of prediction plot graph.
        """
        # Whether to draw forecast and observation in prediction period.
        draw_forecast = X is not None
        if draw_forecast:
            draw_y_test = y is not None

        # When fig and ax are not inputted, create new ones.
        if fig is None:
            fig = create_mpl_fig(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot(1, 1, 1)

        # (1) Training period
        # Prepare for data in training period.
        x_train = self.model.data_.common_index
        y_train = self.model.data_.y
        y_fitted = self.model.fittedvalues_

        # Draw ax in training period.
        ax.bar(x_train, y_train, color=colors[0], label=labels[0])
        ax.plot(
            x_train, y_fitted, color=colors[1], linestyle=linestyles[1], label=labels[1]
        )

        # (2) Prediction period
        if draw_forecast:
            # Prepare for data in prediction period.
            x_pred = self.model.data_.split_index_and_X_from_X_pred(X)[0]
            y_pred = self.model.predict(X)

            # Draw forecast values.
            ax.plot(
                x_pred,
                y_pred,
                color=colors[2],
                linestyle=linestyles[2],
                label=labels[2],
            )

            # If applicable, draw test data
            if draw_y_test:
                y_test = get_1d_arr(y)[0]
                ax.bar(x_pred, y_test, color=colors[0])

            # Draw the line to distinguish training period from prediction one.
            ax.axvline(
                self.model.data_.common_index[-1], color="black", linestyle="dashed"
            )

        if labels is not None:
            ax.legend()
        if title is not None:
            ax.set_title(title)

        return fig, ax

    def plot_components(self, X=None, y=None, figsize=None):
        """Plot in-sample and out-of-sample component predictions.

        Parameters
        ----------
        X : {array-like, int, None}, optional
            Design matrix expressing the regression dummies or variables in
            the period to be predicted. If no regression is defined in the model,
            the index expressing the period or the period steps to be predicted
            must be set.
        y : {array-like, None}, optional
            True values for `X`.
        figsize : {tuple[float, float], None}, optional
            Figure dimension (width, height) in inches. As default, ``figsize``
            is calculated as follows: (16, 3 * (number of components + 1)).

        Returns
        -------
        fig : matplot.figure.Figure
            The matplotlib container including components plot graph.
        ax : matplot.axes.Axes
            The axes object of components plot graph.
        """
        # Observed, fitted, predicted values display settings
        labels = ("Observed", "Fitted", "Predicted")
        colors = ("lightblue", "navy", "red")
        linestyles = (None, "dotted", "solid")

        draw_forecast = X is not None

        # 学習期間・予測期間それぞれで推定値・予測値を取得する.
        comp = Bunch(train={}, pred={})
        for component_name in self.model.components_name_:
            # 予測値を算出するときにはNoneに値を代入して管理
            component_pred_arr = None
            if draw_forecast:
                component_pred_arr = getattr(
                    self.model, "{}_predicted_".format(component_name[:-1])
                )(X)

            # LinearGaussianStateSpaceModelのfreq_seasonalのように複数系列を有する場合もあることに注意する.
            component_arr = getattr(self.model, component_name)
            if component_arr.ndim == 1:
                comp.train[component_name[:-1]] = component_arr
                comp.pred[component_name[:-1]] = component_pred_arr

            else:
                n = component_arr.shape[0]
                for i in range(n):
                    comp.train[
                        "{0}_{1}".format(component_name[:-1], i)
                    ] = component_arr[i, :]
                    comp.pred["{0}_{1}".format(component_name[:-1], i)] = (
                        component_pred_arr[i, :]
                        if component_pred_arr is not None
                        else None
                    )

        # x 軸をそれぞれ取得する.
        x_train = self.model.data_.common_index
        x_pred = (
            self.model.data_.split_index_and_X_from_X_pred(X)[0]
            if draw_forecast
            else None
        )

        # figure, axes の定義
        n_plot = len(comp.train.keys()) + 1
        plot_idx = 1

        if figsize is None:
            figsize = (16, 3 * n_plot)

        fig = create_mpl_fig(figsize=figsize)
        ax = fig.add_subplot(n_plot, 1, plot_idx)

        # prediction
        fig, ax = self.plot_prediction(
            X=X,
            y=y,
            colors=colors,
            linestyles=linestyles,
            labels=labels,
            title="Predictions",
            fig=fig,
            ax=ax,
            figsize=None,
        )

        # components
        for name in comp.train.keys():
            plot_idx += 1
            ax = fig.add_subplot(n_plot, 1, plot_idx)
            ax.plot(
                x_train,
                comp.train[name],
                color=colors[1],
                linestyle=linestyles[1],
                label=labels[1],
            )

            if draw_forecast:
                ax.plot(
                    x_pred,
                    comp.pred[name],
                    color=colors[2],
                    linestyle=linestyles[2],
                    label=labels[2],
                )
                ax.axvline(
                    self.model.data_.common_index[-1], color="black", linestyle="dashed"
                )

            ax.legend()
            ax.set_title(name)
        return fig, ax


class TimeSeriesGrapherMixin:
    """Mixin class for graphing of time series modeling.

    See Also
    --------
    sandbox.graphics.ts_grapher.TimeSeriesGrapher

    Examples
    --------
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
    """

    @property
    def graph(self):
        return TimeSeriesGrapher(self)
