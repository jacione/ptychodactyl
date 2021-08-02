# PtychoDactyl
A simple, approachable ptychography experimental pipeline written in Python

*Maintained by Nick Porter, jacioneportier@gmail.com*

## Overview
There are some really robust software packages out there for doing ptychography--PtyPy, PyNX, and PtychoShelves, to name just a few. However, while these all deliver incredible performance, they come with an unfortunately necessary tradeoff in readability and approachability. This software is an attempt to do the opposite. It's written with a people-over-performance philosophy, which I think will be helpful for ptychography newcomers (like myself, not too long ago).

This repository contains code for collecting and analyzing 2D ptychography data, with 3D hopefully supported soon. 

## Dependencies
This project must be run on **Python 3.8** (later versions are not yet supported by pythonNET). It requires the following packages, all available via `conda install`:
*  click (>=8.0.1)
*  h5py (>=3.2.1)
*  matplotlib (>=3.4.2)
*  numpy (>=1.20.3)
*  pillow (>=8.2.0)
*  progressbar2 (>=3.53.1)
*  pyserial (>=3.5)
*  pythonnet (>=2.5.2)
*  scikit-image (>=0.18.1)
*  scipy (>=1.6.3)

Additionally, the device interfacing uses some proprietary (I assume) software development kits provided by Micronix, Mightex, and Thorlabs. This means some DLL and similar files, not created by me, are included in this repository. These files are used under the terms put forth by their respective owners, and all associated copyrights remain with said owners.

## Instructions
There are two main scripts designed to be run from the command line, one for data collection (`collect.py`) and the other for data analysis (`reconstruct.py`). These each use an associated text file (`collection_specs.txt` and `reconstruction_specs.txt`) to define the needed parameters. Before running either, double-check the spec file and make sure that it's set up the way you want. Both scripts can be run with a `--spec_file <filename>` option if you want to define a specific set of parameters in a different place.

### Collecting data
Make sure the camera, stages, and laser are all turned on and working properly. Open a commandline and navigate to the ptychodactyl directory, then run
```
python collect.py
```
to start the data collection. Data will be saved as a `.pty` file, which is based on HDF5 protocol. This has the benefit of being able to record all the data (diffraction patterns, positions, pixel size, etc.) in a single file.

### Analyzing/reconstructing data
Fill out the associated spec file, then run
```
python reconstruct.py
```
to start the reconstruction. The reconstructed image (probe and object, amplitude and phase) will be saved directly into the `.pty` file, overwriting any previous reconstruction.
