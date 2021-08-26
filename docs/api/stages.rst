.. _stages:

Stage controller interfacing
============================

As discussed :ref:`previously <subclassing>`, you will likely need to define your own ``Stages`` subclass to use this software with your positioning systems. The following subclasses are built-in, and may serve as useful examples.

==============  ============  ============  ==========  ===========================
Subclass        Manufacturer  Model         Status      Dependencies (not included)
==============  ============  ============  ==========  ===========================
``Micronix``    Micronix      MMC-200       stable      `pyserial <https://pypi.org/project/pyserial>`_, `Micronix command syntax <https://micronixusa.com/product/download/evPpvw/universal-document/Avj2vR>`_
``Attocube``    Attocube      ANC350        stable      `pyanc350 <https://github.com/Laukei/pyanc350>`_ (modified), SDK (proprietary)
``YourStages``  n/a           n/a           template    n/a
==============  ============  ============  ==========  ===========================

These can each be found in the source code of *stages.py*.

.. autofunction:: ptycho.stages.get_stages

.. autoclass:: ptycho.stages.Stage
    :members:

