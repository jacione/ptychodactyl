.. _specs:

Parameter specification
=======================

Before you run a data collection or reconstruction, you'll need to fill out the associated specification file. Here's a quick overview of the fields in each file.

collection_specs.txt
--------------------
**title**: Give your data run a name! The collected data will automatically be saved with this title as well as the date on which it was completed, e.g. ``pterodactylus-15-12-1784.pty``.

**data_dir**: The directory where your data will be saved. If left blank, it will automatically be saved in ``ptychodactyl/data``.

**stages**: Keyword for the stage controller you're using. This should be all lowercase, and must match a keyword from the dictionary defined in ``stages.get_stages()``.

**scan_center**: The x- and y-position of the center of your ptychography scan, in the center-origin laser reference frame. All distances should be given in millimeters unless otherwise specified

**scan_width**: Full horizontal scanning range.

**scan_height**: Full vertical scanning range.

**step_size**: Spacing between probe positions. This will affect the overlap, and should not be larger than about 30% of the probe size.

**pattern**: Geometric pattern for the ptychography scan. Options are *rect* (rectangular grid), *hex* (hexagonal grid, tighter packed), and *spiral* (spiraling out from the center).

**num_rotations**: Number of rotational steps (only for 3D ptycho-tomography, for a 2D scan leave this at zero).

**camera**: Keyword for the camera you're using. This should be all lowercase and must match a keyword from the dictionary defined in ``camera.get_camera()``.

**background_frames**: if set to true, the script will record a background image at the beginning of the collection run. Otherwise, it'll jump straight into the scan.

**frames_per_take**: For each scan position (and the background), the camera will take this many frames and return their sum.

**resolution**: The side length (in pixels) of the desired image arrays. Larger images will be reduced through a combination of binning and cropping.

**exposure**: Exposure time for each frame, in milliseconds.

**gain**: Analog gain applied to the camera to boost signal. Tends to increase noise, so use with caution.

**distance**: Distance from the sample (or lens) to the image sensor, in meters.

**energy**: Laser photon energy, in eV.

reconstruction_specs.txt
------------------------

**load**: The name of the ``.pty`` data file to reconstruct. This should work with or without the ``.pty`` suffix.

**data_dir**: The directory where your data file is located. If left blank, it will automatically load from ``ptychodactyl/data``.

**flip_images**: Determines whether to reflect the diffraction images along a certain direction. Options are *h* (reflect horizontally), *v* (reflect vertically), *hv* (both) or *n* (neither).

**flip_positions**: Determines whether/how to invert the recorded probe positions. Options are the same as flip_images.

**background_subtract**: If true, the reconstruction will subtract the stored background image from all diffraction patterns.

**vbleed_correct**: Reduces vertical pixel bleeding on diffraction images. Essentially it subtracts a certain quantile from each column. Must be between 0 and 1. Recommended starting value is 0.35.

**threshold**: Pixel values below this fraction of the maximum will be set to zero. Must be between 0 and 1. Recommended starting value is 0.0001.

.. note::
    This next section of the file (algorithm thru pro_up_final) should be treated like a table. Each field should be a comma-separated list. The script will reconstruct using each column's parameters in series, meaning that the result of the first column's reconstruction will be passed as the initial guess for the second column, etc.

**algorithm**: The iterative algorithm to use for reconstruction. Currently the only options are ePIE and rPIE. These should be put in the spec file in all lowercase.

**num_iterations**: The number of iterations to run that particular algorithm.

**obj_up_initial**, **obj_up_final**, **pro_up_initial**, **pro_up_final**: The initial and final update strengths for the object and probe. These will affect how heavily the algorithm affects the object and probe images with each iteration. The first iteration will use the initial strength, with values being interpolated each iteration until the last iteration uses the final strength. It is often useful to start with big updates and make them smaller as it iterates. It is also often useful to update the object more strongly than the probe.

**animate**: Determines whether the reconstruction should output an animation of the updates at each iteration.