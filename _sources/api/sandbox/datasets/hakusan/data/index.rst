:py:mod:`sandbox.datasets.hakusan.data`
=======================================

.. py:module:: sandbox.datasets.hakusan.data


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   sandbox.datasets.hakusan.data.load



Attributes
~~~~~~~~~~

.. autoapisummary::

   sandbox.datasets.hakusan.data.COPYRIGHT
   sandbox.datasets.hakusan.data.TITLE
   sandbox.datasets.hakusan.data.DESCRIPTION
   sandbox.datasets.hakusan.data.SOURCE
   sandbox.datasets.hakusan.data.NOTE


.. py:data:: COPYRIGHT
   :annotation: = This data is public domain.

   

.. py:data:: TITLE
   :annotation: = HAKUSAN: Ship's Navigation Data

   

.. py:data:: DESCRIPTION
   :annotation: = Multiline-String

    .. raw:: html

        <details><summary>Show Value</summary>

    .. code-block:: text
        :linenos:

        
        A multivariate time series of a ship's yaw rate, rolling, pitching and rudder angles
        which were recorded every second while navigating across the Pacific Ocean.


    .. raw:: html

        </details>

   

.. py:data:: SOURCE
   :annotation: = Multiline-String

    .. raw:: html

        <details><summary>Show Value</summary>

    .. code-block:: text
        :linenos:

        
        http://www.mi.u-tokyo.ac.jp/mds-oudan/lecture_document_2019_math7/時系列データ/hakusan_new.csv


    .. raw:: html

        </details>

   

.. py:data:: NOTE
   :annotation: = Multiline-String

    .. raw:: html

        <details><summary>Show Value</summary>

    .. code-block:: text
        :linenos:

        
        Number of Observations - 1000
        Number of Variables - 4
            YawRate - yaw rate
            Rolling - rolling
            Pitching - pitching
            Rudder - rudder angle


    .. raw:: html

        </details>

   

.. py:function:: load()


