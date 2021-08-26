.. _install-setup:

Installation & Setup
====================

.. _install:

Installing PtychoDactyl
-----------------------

.. PtychoDactyl is **NOT** available through PyPI for a number of reasons, and in order to understand them better, let's compare with a module that *is* on PyPI:
    ===========================================  ==================================================
    **NumPy**                                    **PtychoDactyl**
    ===========================================  ==================================================
    Highly general                               Highly specialized
    Designed to be imported                      Designed to be run
    Works out of the box with nearly any system  Must be refactored for each new experimental setup
    ===========================================  ==================================================
    Primarily for these reasons,

Download the PtychoDactyl source code from my git repository (currently private, maybe a link will go here eventually.

.. note::
    PtychoDactyl has only been tested on Windows 10.

PtychoDactyl requires the following packages, all available via ``pip install``:

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

.. note:: TO-DO:
    Include a picture and brief description of a ptycho experiment.

PtychoDactyl doesn't really care about the laser or the sample, but it does care (quite a bit, actually) about the stages and the camera, which brings us to...

.. _subclassing:

Device Interfacing
------------------

.. warning::
    Unless you're very lucky (or happen to work in the lab where this was developed), you will almost certainly need to write your own controller classes for your own devices.

In order to make device interfacing as painless as possible, PtychoDactyl implements abstract base classes for :ref:`stages <stages>` and :ref:`cameras <camera>`, as well as a stub subclass (a stubclass, if you will) which you will need to flesh out to work with your own devices. While you're free to do this in any way you like, here are a few important considerations:

#. Make a copy of the ``YourStages`` and ``YourCamera`` subclass templates so that you still have them if you need to start over or implement another device.
#. If your subclass requires additional ``import`` statements, put those statements in the ``__init__()`` method.
#. Your subclass must implement all of the abstract methods from the base class, but you may also need to override some other methods (e.g. ``cameras.Andor.get_frames()``), and even add some internal methods of your own (e.g. ``stages.Micronix.command()``).
#. If you use a software development kit (SDK), you'll need to reference those libraries within your subclass. To keep your libraries organized, you may want to make a subdirectory for each device within the ``utils`` directory (e.g. ``utils/Polaroid``).
#. Even if the SDK has a fully functional Python class for your device, it will still be beneficial to make your own subclass that wraps around it because the rest of PtychoDactyl won't recognize the foreign class. One easy way to do this is to give your subclass an attribute like ``self._sdk`` or ``self._handle`` that references to the SDK object, and then just have your methods call its methods.
#. The best way to use your new subclass is to add it to the dictionary in either ``stages.get_stages()`` or ``camera.get_camera()``. Other PtychoDactyl scripts use these functions in conjunction with the :ref:`spec file <specs>`. The key should be the name of the class in all lowercase, and the value should be the name of the subclass (e.g. ``'polaroid': Polaroid``).