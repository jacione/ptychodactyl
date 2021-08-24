"""
Controller classes for motorized stages.
"""
import numpy as np
import time
from abc import ABC, abstractmethod


def get_stages(stage_type, **kwargs):
    """
    Set up the stages with the correct controller subclass.

    .. note::
        If you write your own Stage subclass (recommended), be sure to add it to the dictionary within this function.
        Otherwise, you will get a ``KeyError``.

    :param stage_type: Type of stages to set up.
    :type stage_type: str
    :param kwargs: keyword arguments to pass to stage constructor
    :return: Stage controller object, subclass of stages.Stage
    :rtype: Stage
    """
    options = {
        'micronix': Micronix,
        'attocube': Attocube
    }
    return options[stage_type.lower()](**kwargs)


class Stage(ABC):
    """
    Abstract base class for motorized stage controllers.

    .. warning::
        This class uses two coordinate systems, those of the stages (*x, y, z*) and those of the experiment (*x0, y0,
        z0*). If a rotation stage is included, the two systems are related by the angle between them. Outside of the
        class definition, only the experimental coordinates should be used.

        The experimental coordinates are defined **in terms of the beamline**, with the positive *z0*-axis aligning
        with the beam's path, the *y0*-axis perpendicular to the table (positive up), and the *x0*-axis perpendicular
        to both (positive to the beam's left), to make a right-handed coordinate system.

        To minimize confusion as much as possible, a similar convention is used for the stage coordinates.
        The *y* axis should correspond to the vertical stage, and the two horizontal stages should be *x* and *z* (you
        decide which is which). Note that this is different from how the axes are conventionally defined on the stages
        themselves, where *x* and *y* are horizontal axes, and *z* is vertical.
    """
    @abstractmethod
    def __init__(self, verbose):
        """
        Because this is an abstract class, this constructor must be explicitly called within a subclass constructor.
        At the very least, a subclass constructor should

        #. call this function as ``super().__init__(verbose)``,
        #. connect to the device,
        #. redefine the attributes ``units``, ``axes``, ``limits``, and ``zeros`` if necessary, and
        #. call ``self.measure()``.

        :param verbose: if True, this instance will print information about most processes as it runs.
        :type verbose: bool
        """
        self.verbose = verbose

        # These are the measured positions in the coordinate system of the STAGES.
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        # This is the measured rotational get_position which relates the two coordinate systems
        self.q = 0.0

        # These are the calculated positions in the coordinate system of the BEAMLINE
        self.x0 = 0.0
        self.y0 = 0.0
        self.z0 = 0.0

        self.units = 1  # size of one millimeter in units the controller expects
        self.axes = ['x', 'y', 'z', 'q']  # Basic 4-axis setup
        self.limits = {ax: (0.0, 0.0) for ax in self.axes}  # EOT limits for each axis in AXIS UNITS
        self.zeros = {ax: 0.0 for ax in self.axes}  # Center position for each axis in AXIS UNITS

    @abstractmethod
    def __del__(self):
        """
        This is where you should dispose of any handles to the device. After this function is called, the garbage
        collector should only have to clean up the class object itself.
        """
        return

    @abstractmethod
    def measure(self):
        """
        Check the encoder on each axis and update the attributes within the class instance. This is run automatically
        immediately after any motion command, so that the attributes should always have the correct values.

        .. warning::
            When implementing this method in a subclass, make sure that it returns a value in **millimeters**. This is
            for consistency across all other modules in this package, particularly those in ``ptycho_data.py``.

        .. warning::
            The calculation of ``x0`` and ``z0`` will be different for each setup, but will likely follow some variation
            of *x0 = x cos(q) + z sin(q)*.
        """
        pass

    @abstractmethod
    def check_errors(self):
        """
        Check each axis for errors. For most devices, this will involve calling ``self.get_status()`` for each axis.
        """
        pass

    @abstractmethod
    def get_status(self, ax):
        """
        Check the status of a given axis. Typically, the controller will return a list of booleans, each corresponding
        to a certain attribute (connected, moving, has an error, etc.).
        """
        pass

    @abstractmethod
    def is_moving(self):
        """
        Check whether any of the axes are currently moving. This should return False if ALL stages are at their
        target positions and True if ANY stage is still in motion.
        """
        pass

    # High-level commands #####################################################################################

    @abstractmethod
    def set_position(self, xyzq, laser_frame=True):
        """
        Move the stages to a desired position. When implementing this into a subclass, make sure that it waits until the
        stages have all reached their destination, then calls ``self.measure()`` before returning.

        .. warning::
            When subclassing, if this does not call ``self.measure()``, or if it calls it before the stages are finished
            moving, the object will not read out the correct positions.

        :param xyzq: Desired position for each axis, in mm.
        :type xyzq: (float, float, float, float)
        :param laser_frame: if True, convert positions to be in the laser's reference frame. Default is True.
        :type laser_frame: bool
        """
        pass

    def get_position(self, remeasure=False):
        """
        Get the current position of the sample.

        :param remeasure: If True, call self.measure() before returning a value. If False, use the currently stored
            values. Default is False.
        :type remeasure: bool
        :return: Stage positions in the beam's reference frame (vertical, horizontal, rotational)
        :rtype: np.ndarray
        """
        if remeasure:
            self.measure()
        self.print(f'\nAXIS   POS(mm)\n'
                   f'  X    {self.x0:0.6f}\n'
                   f'  Y    {self.y0:0.6f}\n'
                   f'  Z    {self.z0:0.6f}\n'
                   f'  Q    {self.q:0.6f}')
        return np.array([self.y0, self.x0, self.q])

    def get_stage_positions(self, remeasure=False):
        """
        Get the current positions of each of the stages.

        :param remeasure: If True, call self.measure() before returning a value. If False, use the currently stored
            values. Default is False.
        :type remeasure: bool
        :return: Stage positions in the beam's reference frame (vertical, horizontal, rotational)
        :rtype: np.ndarray
        """
        if remeasure:
            self.measure()
        self.print(f'\nAXIS   POS(mm)\n'
                   f'  X    {self.x:0.6f}\n'
                   f'  Y    {self.y:0.6f}\n'
                   f'  Z    {self.z:0.6f}\n'
                   f'  Q    {self.q:0.6f}')
        return np.array([self.x, self.y, self.z, self.q])

    def check_scan(self, X, Y):
        """
        Check a set of scan positions to make sure that they are all within the stages' travel limits.

        The function checks whether the proposed scan is within the travel limits of the horizontal stages at
        zero-degree rotation (x_ok2), the horizontal stages at 45-deg rotation (x_ok3), and the vertical stages (y_ok)

        :param X: Horizontal scan positions
        :type X: np.ndarray
        :param Y: Vertical scan positions
        :type Y: np.ndarray
        :return: x_ok2, x_ok3, y_ok
        :rtype: (bool, bool, bool)
        """
        xmin = np.max([self.limits['x'][0]-self.zeros['x'], self.limits['z'][0]-self.zeros['z']])
        xmax = np.min([self.limits['x'][1]-self.zeros['x'], self.limits['z'][1]-self.zeros['z']])
        ymin = self.limits['y'][0] - self.zeros['y']
        ymax = self.limits['y'][1] - self.zeros['y']
        X = X * self.units
        Y = Y * self.units
        x_ok2 = np.min(X) > xmin and np.max(X) < xmax
        x_ok3 = np.min(X)*np.sqrt(2) > xmin and np.max(X)*np.sqrt(2) < xmax
        y_ok = np.min(Y) > ymin and np.max(Y) < ymax
        return x_ok2, x_ok3, y_ok

    def home_all(self):
        """Move all stages to their zero positions."""
        self.set_position((0, 0, 0, 0))
        return

    def show_off(self):
        """Moves all the axes around simultaneously...just for fun :)"""
        self.set_position((5, 2.5, 5, 10))
        self.get_position()
        self.set_position((-5, 5, -5, -10))
        self.get_position()
        self.set_position((0, 0, 0, 0))
        self.get_position()

    # Axis-specific commands ##################################################################################

    def set_x0(self, x0_pos):
        """Move the sample to a desired position along the x0-axis (horizontal, orthogonal to the beamline)"""
        self.set_position((x0_pos, self.y0, self.z0, self.q))
        return

    def set_y0(self, y_pos):
        """Move the sample to a desired position along the y0-axis (vertical, orthogonal to the beamline)"""
        self.set_y(y_pos)
        return

    def set_z0(self, z0_pos):
        """Move the sample to a desired position along the x0-axis (horizontal, parallel to the beamline)"""
        self.set_position((self.x0, self.y0, z0_pos, self.q))
        return

    def set_q(self, q_pos):
        """Rotate the stack to a desired angle"""
        self.set_position((self.x0, self.y0, self.z0, q_pos))
        pass

    def set_x(self, x_pos):
        """Move the x-stage to a desired position"""
        self.set_position((x_pos, self.y, self.z, self.q), laser_frame=False)
        return

    def set_y(self, y_pos):
        """Move the y-stage to a desired position"""
        self.set_position((self.x, y_pos, self.z, self.q), laser_frame=False)
        return

    def set_z(self, z_pos):
        """Move the z-stage to a desired position"""
        self.set_position((self.x, self.y, z_pos, self.q), laser_frame=False)
        return

    def home_horizontal(self):
        """Move horizontal stages to their zero positions."""
        self.set_position((0, self.y0, 0, self.q))
        pass

    def get_translation(self):
        """A quick command to get only the positions orthogonal to the beam"""
        return np.array([self.y0, self.x0])

    def get_rotation(self):
        """A quick command to get only the rotational position"""
        return self.q

    # Other commands #######################################################################################

    def print(self, text):
        """A wrapper for print() that checks whether the instance is set to verbose"""
        if self.verbose:
            print(text)


class Attocube(Stage):
    """
    Object-oriented interface for an Attocube ANC350 stage controller.

    This might be a little more tricky than usual because each controller can only connect to 3 stages, which means
    there are 2 controllers:

    Rack    S/N      Axes
    ------  -------  ----
    Top     L010812  rotation, vertical (y), perpendicular-to-beam (coarse x)
    ------  -------  ----
    Bottom  L010810  perpendicular-to-beam (x), parallel-to-beam (z)
    """

    def __init__(self, verbose=False, ignore_rotation=True):
        """
        Instantiate Attocube object
        """
        import utils.Attocube.pyanc350v4 as pyanc

        super().__init__(verbose)
        self.ignore_rotation = ignore_rotation

        self.units = 1e-3  # expects meters, so one millimeter is 0.001

        pyanc.discover(1)
        self.dev_a = pyanc.Positioner(1)
        self.dev_b = pyanc.Positioner(0)

        # Which ANC350 device is each stage connected to?
        self.devices = {
            'q': self.dev_a,
            'x': self.dev_b,
            'y': self.dev_a,
            'z': self.dev_b,
            'xc': self.dev_a
        }
        # Which "axisNo" does each stage occupy on its respective ANC350?
        self.ax_id = {
            'q': 0,
            'x': 0,
            'y': 1,
            'z': 1,
            'xc': 2
        }
        # Each stages EOT limits (m)
        self.limits = {
            'q': (-np.infty, np.infty),
            'x': (0.000196, 0.005270),
            'y': (0.000016, 0.005050),
            'z': (0.000089, 0.005103),
            'xc': (0.000012, 0.015890)
        }
        # Each stage's zero or center position, as measured by the ANC350 (m)
        self.zeros = {
            'q': 168.38,
            'x': sum(self.limits['x'])/2,  # 0.002733
            'y': sum(self.limits['y'])/2,  # 0.002533
            'z': sum(self.limits['z'])/2,  # 0.002596
            'xc': 0.015860
        }

        self.fine_axes = ['x', 'y', 'z', 'q']
        self.all_axes = ['x', 'y', 'z', 'q', 'xc']
        self.xc = 0.0

        # Set default settings
        # Error range on EITHER side of the target position where the stage is still considered "on target"
        target_range = {
            'q': 0.0075,
            'x': 0.0000001,
            'y': 0.0000001,
            'z': 0.0000001,
            'xc': 0.000005
        }
        for ax in self.all_axes:
            self.devices[ax].setAxisOutput(self.ax_id[ax], 1, 0)
            time.sleep(0.05)
            self.devices[ax].setFrequency(self.ax_id[ax], 1000)
            time.sleep(0.05)
            self.devices[ax].startAutoMove(self.ax_id[ax], 1, 0)
            time.sleep(0.05)
            self.devices[ax].setTargetRange(self.ax_id[ax], target_range[ax])
            time.sleep(0.05)
            self.devices[ax].setTargetPosition(self.ax_id[ax], self.zeros[ax])
            time.sleep(0.05)

        self.measure()

    def __del__(self):
        for ax in self.all_axes:
            self.devices[ax].startAutoMove(self.ax_id[ax], 0, 0)
            self.devices[ax].setAxisOutput(self.ax_id[ax], 0, 0)
        self.dev_a.disconnect()
        self.dev_b.disconnect()

    def set_position(self, xyzq, laser_frame=True):
        x, y, z, q = xyzq
        if self.ignore_rotation:
            q = self.q
        if laser_frame:
            q_rads = q * np.pi / 180
            xp = -x*np.cos(q_rads) + z*np.sin(q_rads)
            zp = -x*np.sin(q_rads) - z*np.cos(q_rads)
            x = xp
            z = zp
        xyzq = {'x': x*self.units, 'y': y*self.units, 'z': z*self.units, 'q': q}
        for ax in self.fine_axes:
            if self.ignore_rotation and ax == 'q':
                continue
            self.devices[ax].setTargetPosition(self.ax_id[ax], xyzq[ax]+self.zeros[ax])
        try_count = 0
        while not self.is_on_target():
            time.sleep(0.1)
            try_count += 1
            if try_count > 200:
                raise IOError('Stage is taking too long to find target!')

        self.measure()
        pass

    def measure(self):
        positions = []
        for ax in self.all_axes:
            pos = self.devices[ax].getPosition(self.ax_id[ax])
            if ax != 'q':
                pos *= 1000  # The Attocube controller returns meters
            positions.append(pos)
        self.x, self.y, self.z, self.q, self.xc = tuple(positions)

        q_rads = self.q * np.pi / 180
        self.x0 = self.x * np.cos(q_rads) + self.z * np.sin(q_rads)
        self.z0 = self.x * np.sin(q_rads) - self.z * np.cos(q_rads)
        self.y0 = self.y
        return

    def show_off(self):
        """Moves all the axes around simultaneously...just for fun :)"""
        self.set_position((2, 1, 2, 0))
        self.get_position()
        self.set_position((-2, 2, -2, 0))
        self.get_position()
        self.set_position((0, 0, 0, 0))
        self.get_position()

    def check_errors(self):
        return {ax: self.get_status(ax)[-1] for ax in self.all_axes}

    def check_all(self):
        return {ax: self.get_status(ax) for ax in self.all_axes}

    def get_status(self, ax):
        connected, enabled, moving, target, eotFwd, eotBwd, error = self.devices[ax].getAxisStatus(self.ax_id[ax])
        return connected, enabled, moving, target, eotFwd, eotBwd, error

    def is_moving(self):
        """
        The "moving" status bit is actually quite difficult to work with when "automove" is enabled, since the stage is
        constantly adjusting to stay as close to the target as possible. I've therefore implemented the is_on_target()
        method, which is more useful for our purposes.
        """
        for ax in self.all_axes:
            if self.get_status(ax)[2]:
                return True
        else:
            return False

    def is_on_target(self):
        """Returns whether all of the stages are on target."""
        for ax in self.all_axes:
            if self.ignore_rotation and ax == 'q':
                continue
            if not self.get_status(ax)[3]:
                return False
        else:
            return True


class Micronix(Stage):
    """
    Object-oriented interface for a Micronix MMC-200 stage controller.

    .. note::
        This class only implements a few basic functions which are relevant to ptychography data collection. For more
        advanced functionality, you can use the ``command`` method to send specific serial commands using the
        supported `command line syntax <https://micronixusa.com/product/download/evPpvw/universal-document/Avj2vR>`_.
    """
    def __init__(self, port='COM3', verbose=False):
        """
        Create a Stage object to interface with a Micronix MMC-200 stage controller.

        :param port: Serial port connected to the stage controller.
        :type port: str
        :param verbose: if True, this instance will print information about most processes as it runs.
        :type verbose: bool
        """
        super().__init__(verbose)
        # Connect to the controller
        import serial
        self.port = serial.Serial(port, baudrate=38400, timeout=2)
        self.port.flush()

        # This just makes it easier to access each axis sequentially
        self.axes = ['x', 'y', 'z', 'q']
        self.ax_id = {
            'x': 3,
            'y': 1,
            'z': 2,
            'q': 4
        }
        self.limits = {
            'x': (-9.0, 9.0),
            'y': (0.1, 9.9),
            'z': (-9.0, 9.0),
            'q': (-np.infty. np.infty)
        }

        self.check_errors()
        self.measure()
        pass

    def __del__(self):
        self.port.flush()
        self.port.close()

    def set_position(self, xyzq, laser_frame=True):
        x, y, z, q = xyzq
        if laser_frame:
            q_rads = q * np.pi / 180
            xp = x*np.cos(q_rads) + z*np.sin(q_rads)
            zp = z*np.cos(q_rads) - x*np.sin(q_rads)
            x = xp
            z = zp
        xyzq = {'x': x, 'y': y, 'z': z, 'q': q}
        for ax in self.axes:
            self.command(f'{self.ax_id[ax]}MSA{xyzq[ax]:0.6f}')
        self.command('0RUN')
        while self.is_moving():
            time.sleep(0.1)
        self.measure()
        return

    def command(self, cmd_str):
        """
        Sends a command string to the mightex stage without checking for a response. This can be used for more advanced
        features using the command line syntax provided by Micronix:
        https://micronixusa.com/product/download/evPpvw/universal-document/Avj2vR

        :param cmd_str: ASCII command
        :type cmd_str: str
        """
        self.port.write(bytes(f'{cmd_str}\n\r'.encode('utf-8')))

    def query(self, qry_str):
        """
        Sends a query string to the mightex stage and returns its response. This can be used for more advanced
        features using the command line syntax provided by Micronix:
        https://micronixusa.com/product/download/evPpvw/universal-document/Avj2vR

        :param qry_str: ASCII command
        :type qry_str: str
        :return: Response string
        :rtype: str
        """
        self.port.flush()
        self.port.write(bytes(f'{qry_str}\n\r'.encode('utf-8')))
        time.sleep(0.1)
        s = str(self.port.readline())
        s = s.replace(r'\n', '')
        s = s.replace(r'\r', '')
        if s.startswith("b'"):
            s = s[2:]
        if s.endswith("'"):
            s = s[:-1]
        s = s[s.find('#') + 1:]
        return s

    def measure(self):
        """
        Check the encoder on each axis and update the attributes within the class instance. This is run automatically
        immediately after any motion command, so that the attributes should always have the correct values.
        """
        positions = []
        for ax in self.axes:
            pos = self.query(f'{self.ax_id[ax]}POS?')
            pos = pos.split(',')  # Parse the returned string
            pos = float(pos[1])
            positions.append(pos)

        self.x, self.y, self.z, self.q = tuple(positions)
        q_rads = self.q * np.pi / 180
        self.x0 = self.x * np.cos(q_rads) - self.z * np.sin(q_rads)
        self.z0 = self.x * np.sin(q_rads) + self.z * np.cos(q_rads)
        self.y0 = self.y
        return

    def check_errors(self):
        """Check each axis for errors"""
        for ax in self.axes:
            status = self.get_status(ax)
            self.print(f'{ax}-axis  (SB: {status})')
            err = self.query(f'{ax}ERR?')
            if len(err):
                for s in err.split('#'):
                    if len(s) and not s.isspace():
                        self.print(f'\t{s}')
            else:
                self.print('\tNo errors')

    def get_status(self, ax):
        """Check the status byte for a given axis"""
        s = int(self.query(f'{self.ax_id[ax]}STA?'))
        status = f'{s:b}'.rjust(8, '0')
        return status

    def is_moving(self):
        """Check each axis' status byte to see if any of them are moving"""
        for ax in self.axes:
            if bool(int(self.get_status(ax)[1:4])):
                return True
        else:
            return False


class YourStage(Stage):

    def __init__(self, verbose):
        super().__init__(verbose)

        # Establish connection with the device

        self.home_all()
        self.measure()
        pass

    def __del__(self):
        # Your code goes here
        pass

    def measure(self):
        # Measure each axis, then set self.x, self.y, self.z, self.q, self.x0, self.y0, and self.z0.
        pass

    def check_errors(self):
        # Check each axis for errors
        pass

    def get_status(self, ax):
        # General status check for a particular axis.
        pass

    def is_moving(self):
        pass

    def set_position(self, xyzq, laser_frame=True):

        # Your code goes here!

        self.measure()
        pass


if __name__ == '__main__':
    ctrl = Attocube(verbose=True)
    ctrl.show_off()
