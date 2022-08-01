:py:mod:`sandbox.datamodel.ts_simulator`
========================================

.. py:module:: sandbox.datamodel.ts_simulator


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   sandbox.datamodel.ts_simulator.UnobservedComponentsSimulator
   sandbox.datamodel.ts_simulator.UnobservedComponentsSimulatorResult




.. py:class:: UnobservedComponentsSimulator(steps, level=True, trend=False, freq_seasonal=None, exog_params=None, start_param_level=0.0, stddev_level=1.0, stddev_trend=1.0, stddev_freq_seasonal=None, seed=123456789, **kwargs)

   Bases: :py:obj:`sandbox.datamodel.base.BaseDataSimulator`

   Base class for data simulator.

   .. py:method:: simulate()



.. py:class:: UnobservedComponentsSimulatorResult(**kwargs)

   Bases: :py:obj:`sandbox.datamodel.base.BaseData`

   .. py:method:: trend()
      :property:


   .. py:method:: freq_seasonal()
      :property:


   .. py:method:: exog()
      :property:


   .. py:method:: reg()
      :property:


   .. py:method:: endog()
      :property:


   .. py:method:: common_index()
      :property:


   .. py:method:: nobs()
      :property:


   .. py:method:: convert_pandas()


   .. py:method:: convert_ndarray()



