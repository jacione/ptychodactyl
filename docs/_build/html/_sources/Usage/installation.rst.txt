.. _install-setup:

Installation & Setup
====================

.. _install:

Installing PtychoDactyl
-----------------------

Download the PtychoDactyl source code from my git repository.

.. note::
    PtychoDactyl has only been tested on Windows 10.

PtychoDactyl requires the following packages, all available via ``conda install`` or ``pip install``:

* click (>=8.0.1)
* h5py (>=3.2.1)
* matplotlib (>=3.4.2)
* numpy (>=1.20.3)
* pillow (>=8.2.0)
* progressbar2 (>=3.53.1)
* scikit-image (>=0.18.1)
* scipy (>=1.6.3)

.. _setup:

Experimental setup
------------------

If you want to run a real ptychography experiment, you'll obviously need more than just a code library. This sadly means that, in addition to software dependencies, there are hardware dependencies. In order to properly perform a ptychography experiment with this code, you will need at least four things:

* A laser
* A sample
* A multi-axis positioning system
* A lensless camera

PtychoDactyl doesn't care about the laser or the sample, but it does care (quite a bit, actually) about the stages and the camera.

.. warning::
    You will probably need to write your own controller classes for your own devices.

For full functionality, your classes should inherit the abstract base classes provided in :ref:`stages` and :ref:`camera`.

Additionally, the device interfacing uses some proprietary software development kits provided by Micronix, Mightex, and Thorlabs. This means some DLL and similar files, not created by me, are included in this repository. These files are used under the terms put forth by their respective owners, and all associated copyrights remain with said owners.