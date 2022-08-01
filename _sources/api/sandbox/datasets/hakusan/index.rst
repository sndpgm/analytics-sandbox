:py:mod:`sandbox.datasets.hakusan`
==================================

.. py:module:: sandbox.datasets.hakusan

.. autoapi-nested-parse::

   Datasets hakusan.



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   data/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   sandbox.datasets.hakusan.load



Attributes
~~~~~~~~~~

.. autoapisummary::

   sandbox.datasets.hakusan.COPYRIGHT
   sandbox.datasets.hakusan.DESCRIPTION
   sandbox.datasets.hakusan.NOTE
   sandbox.datasets.hakusan.SOURCE
   sandbox.datasets.hakusan.TITLE


.. py:data:: COPYRIGHT
   :annotation: = This data is public domain.

   

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

   

.. py:data:: SOURCE
   :annotation: = Multiline-String

    .. raw:: html

        <details><summary>Show Value</summary>

    .. code-block:: text
        :linenos:

        
        http://www.mi.u-tokyo.ac.jp/mds-oudan/lecture_document_2019_math7/時系列データ/hakusan_new.csv


    .. raw:: html

        </details>

   

.. py:data:: TITLE
   :annotation: = HAKUSAN: Ship's Navigation Data

   

.. py:function:: load()


