# PtychoDactyl
A simple, approachable ptychography experimental pipeline written in Python.

*Maintained by Nick Porter, jacioneportier@gmail.com*

## Overview
There are some really robust software packages out there for doing ptychography—PtyPy, PyNX, and PtychoShelves, to name just a few. However, while these all deliver incredible performance, they come with an unfortunately necessary tradeoff in readability and approachability. This software is an attempt to do the opposite. It’s written with a people-over-performance philosophy, which I think will be helpful for ptychography newcomers (like myself, not too long ago).

As to why I chose the name “PtychoDactyl,” it was a combination four factors. First, there aren’t very many words that have the silent “p” and I wanted that. Second, I used my fingers (greek dactylos) to type it. Third, my son likes dinosaurs (although technically pterosaurs weren’t dinosaurs). Finally, GitHub asked for a name and it was the first thing I could come up with.

This repository contains code for collecting and analyzing 2D ptychography data, with 3D hopefully supported soon. 

## Setup
Download the PtychoDactyl source code from the git repository (currently private, maybe a link will go here eventually). You will also need the following packages, all available via `pip install`:
*  h5py (>=3.2.1)
*  matplotlib (>=3.4.2)
*  numpy (>=1.20.3)
*  pillow (>=8.2.0)
*  progressbar2 (>=3.53.1)
*  scikit-image (>=0.18.1)
*  scipy (>=1.6.3)
*  tqdm
*  pyyaml
*  tifffile

So far, this version of this project has only been tested on Windows 10, running Python 3.11.
