:py:mod:`sandbox.datasets.utils`
================================

.. py:module:: sandbox.datasets.utils

.. autoapi-nested-parse::

   Datasets utils modules.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.datasets.utils.Dataset



Functions
~~~~~~~~~

.. autoapisummary::

   sandbox.datasets.utils.load_csv
   sandbox.datasets.utils.load_dataset



.. py:class:: Dataset(**kw)

   Bases: :py:obj:`dict`

   Dataset class.


.. py:function:: load_csv(base_file, csv_name, sep=',')

   Load standard csv file.


.. py:function:: load_dataset(data: DataFrame | Series, endog_name: str, exog_name: List[str] | None = None)

   Load dataset.


