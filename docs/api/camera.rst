.. _camera:

Camera interfacing
==================

Examples:

============  ============  ==========  ===========================
Manufacturer  Camera model  Status      Dependencies (not included)
============  ============  ==========  ===========================
Thorlabs      S805MU1       stable      `Python.NET <https://pypi.org/project/pythonnet/>`_, `Thorlabs camera SDK <https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam>`_
Andor         DO936N        unfinished  `Andor SDK <https://andor.oxinst.com/downloads/view/andor-sdk-2.104.30000.0>`_ (requires verification)
Mightex       SME-B050-U    buggy       `Mightex camera software package <https://www.mightexsystems.com/product/usb3-0-monochrome-or-color-5mp-cmos-camera-8-or-12-bit/>`_
============  ============  ==========  ===========================

.. autofunction:: camera.get_camera

.. autoclass:: camera.Camera
    :members:
