:py:mod:`sandbox.datamodel.ts_simulator`
========================================

.. py:module:: sandbox.datamodel.ts_simulator


Module Contents
---------------

.. py:class:: UnobservedComponentsSimulator(steps, level=True, trend=False, freq_seasonal=None, exog_params=None, start_param_level=0.0, stddev_level=1.0, stddev_trend=1.0, stddev_freq_seasonal=None, seed=123456789, **kwargs)

   Bases: :py:obj:`sandbox.datamodel.base.BaseDataSimulator`

   .. py:method:: simulate()


   .. py:method:: _simulate()


   .. py:method:: _simulate_trend_term(steps, start_param, stddev_level, stddev_trend)


   .. py:method:: _simulate_seasonal_term(periodicity, total_cycles, stddev_freq_seasonal, harmonics)

      Simulate seasonality component data.

      :param periodicity: Base cycle.
      :type periodicity: int
      :param total_cycles: Number of cycles.
      :type total_cycles: int
      :param stddev_freq_seasonal: Standard deviation on seasonality component noise.
      :type stddev_freq_seasonal: float
      :param harmonics: The numbers of harmonics
      :type harmonics: {int, None}

      :returns: **series** -- Simulated data array.
      :rtype: numpy.ndarray


   .. py:method:: _simulate_exog(steps, exog_params)



.. py:class:: UnobservedComponentsSimulatorResult(**kwargs)

   Bases: :py:obj:`sandbox.datamodel.base.BaseData`

   .. py:attribute:: _trend
      

      

   .. py:attribute:: _freq_seasonal
      

      

   .. py:attribute:: _exog
      

      

   .. py:attribute:: _reg
      

      

   .. py:attribute:: _endog
      

      

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



