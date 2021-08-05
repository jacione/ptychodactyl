.. _stages:

Stage controller interfacing
============================

Examples:

============  ============  ==========  ===========================
Manufacturer  Model         Status      Dependencies (not included)
============  ============  ==========  ===========================
Micronix      MMC-200       stable      `pyserial <https://pypi.org/project/pyserial>`_, `Micronix command syntax <https://micronixusa.com/product/download/evPpvw/universal-document/Avj2vR>`_
Attocube      ANC350        unfinished  `pyanc350 <https://github.com/Laukei/pyanc350>`_ (modified), drivers/DLLs from Attocube
============  ============  ==========  ===========================

.. autofunction:: stages.get_stage

.. autoclass:: stages.Stage
    :members:

