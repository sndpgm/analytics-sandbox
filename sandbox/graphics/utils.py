"""The utility functions on graphics."""
from matplotlib.colors import is_color_like as is_color

LINE_STYLES = {
    "solid": "solid",
    "dotted": "dotted",
    "dashed": "dashed",
    "dashdot": "dashdot",
    "loosely dotted": (0, (1, 10)),
    "densely dotted": (0, (1, 1)),
    "loosely dashed": (0, (5, 10)),
    "densely dashed": (0, (5, 1)),
    "loosely dashdotted": (0, (3, 10, 1, 10)),
    "dashdotted": (0, (3, 5, 1, 5)),
    "densely dashdotted": (0, (3, 1, 1, 1)),
    "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
    "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
}


def _import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # ToDo: util関数でimportしていなければエラー (または自動でインストール?) を返すメソッドを用意.
        msg = "Matplotlib is not found."
        raise ImportError(msg)
    return plt


def create_mpl_fig(fig=None, figsize=None):
    if fig is None:
        plt = _import_matplotlib_pyplot()
        fig = plt.figure(figsize=figsize, constrained_layout=True)
    return fig


def is_color_like(c):
    """Return whether c can be interpreted as an RGB(A) color."""
    return is_color(c)


def is_linestyle(linestyle):
    """Return whether linestyle can be defined as the one in this module."""
    if linestyle in LINE_STYLES.keys():
        return True
    else:
        return False


def convert_mpl_linestyle(linestyle):
    """Convert linestyle into the one which can be interpreted in matplotlib."""
    if is_linestyle(linestyle):
        return LINE_STYLES[linestyle]
    else:
        msg = "Specified linestyle must be as follows, not `{}`: {}.".format(
            linestyle, ", ".join(list(LINE_STYLES.keys()))
        )
        raise ValueError(msg)
