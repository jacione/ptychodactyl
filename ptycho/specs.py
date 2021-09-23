import numpy as np
from pathlib import Path


class CollectionSpecs:
    def __init__(self, title, stages, camera, data_dir='', scan_center=(0.0, 0.0), scan_width=1.0, scan_height=1.0,
                 scan_step=0.1, z_position=0.0, scan_pattern='spiral', num_rotations=0, background=True,
                 frames_per_take=5, resolution=512, exposure=100, gain=1, distance=0.1, energy=1.97, verbose=False):
        """
        A class to specify parameters for ptychographic data collection. All distances should be given in millimeters
        unless otherwise specified.

        :param title: Give your data run a name! The collected data will automatically be saved with this title as well
            as the date on which it was completed, e.g. 'pterodactylus-15-12-1784.pty'.
        :type title: str
        :param stages: Keyword for the stage controller you're using. This should be all lowercase, and must match a
            keyword from the dictionary defined in ``stages.get_stages()``.
        :type stages: str
        :param camera: Keyword for the camera you're using. This should be all lowercase and must match a keyword from
            the dictionary defined in ``camera.get_camera()``.
        :type camera: str
        :param data_dir: The directory where your data will be saved. If no value is given, it will automatically be
            saved in ``ptychodactyl/data``.
        :type data_dir: str
        :param scan_center: The x- and y-position of the center of your ptychography scan, in the center-origin laser
            reference frame.
        :type scan_center: float, float
        :param scan_width: Full horizontal scanning range.
        :type scan_width: float
        :param scan_height: Full vertical scanning range.
        :type scan_height: float
        :param scan_step: Spacing between probe positions. This will affect the overlap, and should not be larger than
            about 30% of the probe size.
        :type scan_step: float
        :param z_position: Position of the sample along the optical axis.
        :type z_position: float
        :param scan_pattern: Geometric pattern for the ptychography scan. Options are *rect* (rectangular grid), *hex*
            (hexagonal grid, tighter packed), and *spiral* (spiraling out from the center).
        :type scan_pattern: str
        :param num_rotations: Number of rotational steps. If no value is given, the dataset will be 2D.
        :type num_rotations: int
        :param background: if ``True`` (default), the script will record a background image at the beginning of the
            collection run. Otherwise, it'll jump straight into the scan.
        :type background: bool
        :param frames_per_take: For each scan position (and the background), the camera will take this many frames and
            return their sum.
        :type frames_per_take: int
        :param resolution: The side length (in pixels) of the desired image arrays. Larger images will be reduced
            through a combination of binning and cropping.
        :type resolution: int
        :param exposure: Exposure time for each frame, in milliseconds.
        :type exposure: int
        :param gain: Analog gain applied to the camera to boost signal. Tends to increase noise; use with caution.
        :type gain: float
        :param distance: Distance from the sample (or lens) to the image sensor, in METERS.
        :type distance: float
        :param energy: Laser photon energy, in eV.
        :type energy: float
        :param verbose: If ``True``, print verbose information during the data collection. Otherwise, show a simple
            progress bar.
        :type verbose: bool
        """
        # Title
        self.title = title
        self.data_dir = data_dir

        # Probe positioning parameters
        self.stages = stages  # Keyword for the stage controller that you're using
        self.scan_center = scan_center  # X,Y stage position where probe is centered on object (mm)
        self.scan_width = scan_width  # Horizontal scanning range (mm)
        self.scan_height = scan_height  # Vertical scanning range (mm)
        self.scan_step = scan_step  # Spacing between probe positions (mm)
        self.z_position = z_position  # Positioning along the beamline
        self.pattern = scan_pattern  # Geometric pattern for ptychography scan (rect, hex, or spiral)
        self.num_rotations = num_rotations  # Number of rotational steps (zero for no rotation)

        # Image acquisition parameters
        self.camera = camera  # Keyword for the camera that you're using
        self.background = background  # Set to true to record a background image with the laser off
        self.frames_per_take = frames_per_take  # Number of frames to sum for each measurement (including background)
        self.resolution = resolution  # Desired final image size (number of pixels on a side)
        self.exposure = exposure  # Exposure time (ms)
        self.gain = gain  # Analog gain

        # Other parameters
        self.distance = distance  # Distance from lens (or sample) to detector (m)
        self.energy = energy  # Laser photon energy (eV)
        self.verbose = verbose


class ReconstructionSpecs:
    def __init__(self, title, data_dir='', flip_images=False, rotate_positions=0, background_subtract=True,
                 vbleed_correct=0.35, threshold=0.0002):
        """

        :param title: The name of the PTY data file to reconstruct. (Works with or without the ``.pty`` suffix).
        :type title: str
        :param data_dir: The directory where your data file is located. If no value is given, it will automatically
            load from ``ptychodactyl/data``.
        :type data_dir: str
        :param flip_images: If ``True``, all diffraction images will be reflected vertically. Whether this is needed
            depends on the behavior and position of the camera.
        :type flip_images: bool
        :param rotate_positions: Rotates all probe position coordinates by the given angle about the optical axis. Use
            only if the camera's image detector is rotated with respect to the stage coordinate system.
        :type rotate_positions: float
        :param background_subtract: If ``True``, the reconstruction will subtract the stored background image from all
            diffraction patterns.
        :type background_subtract: bool
        :param vbleed_correct: Reduces vertical pixel bleeding on diffraction images. Essentially it subtracts a certain
            quantile from each column. Must be between 0 and 1. Recommended starting value is 0.35.
        :type vbleed_correct: float
        :param threshold: Pixel values below this fraction of the maximum will be set to zero. Must be between 0 and 1.
            Recommended starting value is 0.0001.
        :type threshold: float
        """
        # Title
        self.title = title
        if not title.endswith('.pty'):
            title = title + '.pty'
        if data_dir == '':
            file = Path(__file__).parents[1] / 'data' / title
        else:
            file = Path(data_dir) / title
        self.file = file

        # Pre-processing parameters
        self.flip_images = flip_images  # 'h' to flip horizontal, 'v' for vertical, 'hv' for both, 'n' for neither
        self.rotate_positions = rotate_positions  # Angle in degrees between the stages and the camera coordinates
        self.background_subtract = background_subtract  # Set to true to perform background subtraction
        self.vbleed_correct = vbleed_correct  # Vertical bleeding reduction factor (must be between 0 and 1)
        self.threshold = threshold  # Pixel values below this fraction of the maximum will be set to zero

        # Reconstruction parameters
        self.cycles = []

    def add_cycle(self, algorithm, num_iterations, object_update=(0.9, 0.9), probe_update=(0.5, 0.3), animate=False):
        """

        :param algorithm: The iterative algorithm to use for reconstruction. Currently the only options are 'epie' and
            'rpie'.
        :type algorithm: str
        :param num_iterations: The number of iterations to run that particular algorithm.
        :type num_iterations: int
        :param object_update: The initial and final update strengths for the object. These will affect how
            heavily the algorithm affects the object images with each iteration. The first iteration will use
            the initial strength, with values being interpolated each iteration until the last iteration uses the final
            strength.
        :type object_update: (float, float)
        :param probe_update: The initial and final update strengths for the probe. These should typically be smaller
            than the object update strength because the entire probe gets updated with each step.
        :type probe_update: (float, float)
        :param animate: If ``True``, show an animation of the reconstruction progress at each iteration. This will only
            appear after all reconstruction cycles have completed.
        :type animate: bool
        """
        self.cycles.append(ReconCycle(algorithm, num_iterations, object_update, probe_update, animate))


class ReconCycle:
    def __init__(self, algorithm, num_iterations, object_update, probe_update, animate):
        self.algorithm = algorithm  # Options: rpie, epie
        self.num_iterations = num_iterations  # number of iterations to run_cycle
        o_i, o_f = object_update  # Initial/final object update strength
        p_i, p_f = probe_update  # Initial/final probe update strength
        self.o_param = np.linspace(o_i, o_f, num_iterations)
        self.p_param = np.linspace(p_i, p_f, num_iterations)
        self.animate = animate
