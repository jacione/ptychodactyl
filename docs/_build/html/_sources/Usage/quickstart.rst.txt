.. _quickstart:

Using PtychoDactyl
==================

.. _quickstart-auto:

Fully automated ptychography
----------------------------

If you have working classes for the devices you're using, then running PtychoDactyl is as easy as pie! There are two main scripts designed to be run from the command line, one for data collection (``collect.py``) and the other for data analysis (``reconstruct.py``). These each use an associated text file (``collection_specs.txt`` and ``reconstruction_specs.txt``) to define the needed parameters. Before running either, double-check the spec file and make sure that it's set up the way you want. Both scripts can be run with a ``--spec_file <filename>`` option if you want to define a specific set of parameters in a different place.

Make sure the camera, stages, and laser are all turned on and working properly. Open a commandline and navigate to the ptychodactyl directory, then run

.. code-block:: console

    $ python collect.py

to start the data collection. Data will be saved as a ``.pty`` file, which is based on HDF5 protocol. This has the benefit of being able to record all the data (diffraction patterns, positions, pixel size, etc.) in a single file.

To analyze and reconstruct the data, fill out the associated spec file, then run

.. code-block:: console

    $ python reconstruct.py

to start the reconstruction. The reconstructed image (probe and object, amplitude and phase) will be saved directly into the ``.pty`` file, overwriting any previous reconstruction.

.. warning::
    Running reconstruct.py with default options *will overwrite* any previous reconstructions within the ``.pty`` file!

.. _quickstart-custom:

Ptychography toolkit
--------------------

If you want a more customizable experience, you can also use PtychoDactyl to create your own pipeline. The :ref:`ptycho_data` module contains very useful classes for managing ptychographic data, and the :ref:`recon` module contains several options for reconstructing that data. For example, say you want to do manual 2D ptychography---that is, without motorized stages or an automated camera capture. You could do something like the following example:

.. code-block:: python

    import ptycho_data
    from skimage.io import imread

    num_positions = 16  # It could be any number here
    path_to_images = 'your/path/here/'
    image_name = 'my_image_'  # Save all your images as 'my_image_X', replacing X with the image number.
    image_fmt = '.png'

    data = ptycho_data.CollectData(
        title='ptyceratops',
        num_positions=num_positions
        # Other arguments here...
    )

    for i in num_takes:
        print('Move stages and input new positions')
        y = float(input('Y: '))
        x = float(input('X: '))
        theta = 0  # Angle is always zero for 2D
        position = np.array([y, x, theta])

        input('Take picture, then press enter...')
        image = imread(f'{path_to_images}{image_name}{i}{image_fmt}', as_gray=True)

        data.record_data(position, image)

    data.save_to_pty(timestamp=False)

And there you have it! All of the images and their associated positions will be stored in a file called ``ptyceratops.pty`` in the ``data`` directory.
