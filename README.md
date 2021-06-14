# 3D Ptychography

*Maintained by Nick Porter, jacione@physics.byu.edu*

## Overview
This repository contains code for collecting and analyzing 2- or 3D ptychography data.

## Dependencies
This project must be run on **Python 3.8** (later versions are not yet supported by all dependencies). It requires the following packages, all available via `pip install`:
*  click (>=8.0.1)
*  h5py (>=3.2.1)
*  matplotlib (>=3.4.2)
*  numpy (>=1.20.3)
*  progressbar2 (>=3.53.1)
*  pyserial (>=3.5)
*  pythonnet (>=2.5.2)
*  scikit-image (>=0.18.1)
*  scipy (>=1.6.3)

Additionally, the device interfacing uses some proprietary (I assume) software development kits provided by Micronix, Mightex, and Thorlabs. This means some DLL and similar files, not created by me, are included in this repository. These files are used under the terms put forth by their respective owners, and all associated copyrights remain with said owners.

## Instructions
There are two main scripts designed to be run from the command line, one for data collection (`collect.py`) and the other for data analysis (`reconstruct.py`).

### Collecting data
Make sure the camera, stages, and laser are all turned on and working properly. Open a commandline and navigate to the 3d-ptychography directory, then run
```
python collect.py
```
to start the data collection. The commandline interface accepts options such as `-s <title>` to set the title which the data will be saved as or `--exposure <time_in_ms>` to set the exposure time. For a list of all options, run the script with the `-h` or `--help` flag.

Data will be saved as a `.cxi` file, which is based on HDF5 protocol. This has the benefit of being able to record all the data (diffraction patterns, positions, pixel size, etc.) in a single file.

### Analyzing/reconstructing data
Open a commandline and navigate to the 3d-ptychography directory, then run
```
python reconstruct.py -f <datafile>
```
to start the reconstruction. Eventually, this will be set up so that the user can append flags like `-a 5 epie 5 rpie` to do five iterations of the ePIE reconstruction algorithm followed by 5 iterations of the rPIE algorithm. However, this has not yet been implemented. If no file is provided, it will reconstruct a simulated data set located in the /libs directory.