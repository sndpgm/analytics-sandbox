:py:mod:`sandbox.graphics.utils`
================================

.. py:module:: sandbox.graphics.utils

.. autoapi-nested-parse::

   The utility functions on graphics.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   sandbox.graphics.utils.create_mpl_fig
   sandbox.graphics.utils.is_color_like
   sandbox.graphics.utils.is_linestyle
   sandbox.graphics.utils.convert_mpl_linestyle



Attributes
~~~~~~~~~~

.. autoapisummary::

   sandbox.graphics.utils.LINE_STYLES


.. py:data:: LINE_STYLES
   

   

.. py:function:: create_mpl_fig(fig=None, figsize=None)


.. py:function:: is_color_like(c)

   Return whether c can be interpreted as an RGB(A) color.


.. py:function:: is_linestyle(linestyle)

   Return whether linestyle can be defined as the one in this module.


.. py:function:: convert_mpl_linestyle(linestyle)

   Convert linestyle into the one which can be interpreted in matplotlib.


