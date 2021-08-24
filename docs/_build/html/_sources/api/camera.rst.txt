.. _camera:

Camera interfacing
==================

As discussed :ref:`previously <subclassing>`, you will likely need to define your own ``Camera`` subclass to use this software with your own devices. The following subclasses are built-in, and may serve as useful examples in this process.

==============  ============  ============  ==========  ===========================
Subclass        Manufacturer  Camera model  Status      Dependencies (not included)
==============  ============  ============  ==========  ===========================
``ThorCam``     Thorlabs      S805MU1       stable      `Python.NET <https://pypi.org/project/pythonnet/>`_, `SDK <https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam>`_ (proprietary)
``Mightex``     Mightex       SME-B050-U    buggy       `SDK <https://www.mightexsystems.com/product/usb3-0-monochrome-or-color-5mp-cmos-camera-8-or-12-bit/>`_ (proprietary)
``Andor``       Andor         iKon-L        stable      `SDK <https://andor.oxinst.com/downloads/view/andor-sdk-2.104.30000.0>`_ (proprietary, paid)
``YourCamera``  n/a           n/a           template    n/a
==============  ============  ============  ==========  ===========================

These can each be found in the source code of *camera.py*.

.. autofunction:: camera.get_camera

.. autoclass:: camera.Camera
    :members:
