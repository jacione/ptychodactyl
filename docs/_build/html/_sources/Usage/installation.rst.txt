.. _install-setup:

Installation & Setup
====================

.. _install:

Installing PtychoDactyl
-----------------------

Download the PtychoDactyl source code from my git repository.

.. note::
    PtychoDactyl has only been tested on Windows 10.

This project must be run on **Python 3.8** (later versions are not yet supported by pythonnet). It requires the following packages, all available via ``conda`` (recommended) or ``pip``:

* click (>=8.0.1)
* h5py (>=3.2.1)
* matplotlib (>=3.4.2)
* numpy (>=1.20.3)
* pillow (>=8.2.0)
* progressbar2 (>=3.53.1)
* pyserial (>=3.5)
* pythonnet (>=2.5.2)
* scikit-image (>=0.18.1)
* scipy (>=1.6.3)

.. _setup:

Experimental setup
------------------

If you want to run a real ptychography experiment, you'll obviously need more than just a code library. This  means that, in addition to software dependencies, there are hardware dependencies. In order to properly perform a ptychography experiment with this code, you will need at least four things:

* A laser
* A sample
* A motorized stage mount
* A lensless camera

PtychoDactyl doesn't care about the laser or the sample, but it does care (quite a bit, actually) about the stages and the camera.

.. warning::
    Unless you are using the same stages/camera I did, you will need to write your own controller classes for your devices. For full functionality, your classes should inherit the abstract base classes provided in :ref:`camera` and :ref:`micronix`.

Additionally, the device interfacing uses some proprietary software development kits provided by Micronix, Mightex, and Thorlabs. This means some DLL and similar files, not created by me, are included in this repository. These files are used under the terms put forth by their respective owners, and all associated copyrights remain with said owners.