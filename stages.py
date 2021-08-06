"""
Controller classes for motorized stages.
"""
import numpy as np
import time
from abc import ABC, abstractmethod


def get_stage(stage_type, **kwargs):
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

    .. note::
        Axes are defined **in terms of the beamline** which means that the y-axis in this class corresponds to the
        vertical stage, and the z-axis is horizontal and parallel to the beam. This is different from how the axes are
        conventionally defined in the stages themselves, where z is assumed to be vertical.
    """
    @abstractmethod
    def __init__(self, verbose):
        """
        Create a generic Stage object

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

    def __del__(self):
        return

    @abstractmethod
    def measure(self):
        """
        Check the encoder on each axis and update the attributes within the class instance. This is run automatically
        immediately after any motion command, so that the attributes should always have the correct values.

        .. note::
            When implementing this method in a subclass, make sure that it returns a value in **millimeters**. This is
            for consistency across all other modules in this package, particularly those in ``ptycho_data.py``.
        """
        pass

    @abstractmethod
    def check_errors(self):
        """Check each axis for errors"""
        pass

    @abstractmethod
    def get_status(self, ax):
        """Check the status of a given axis"""
        pass

    @abstractmethod
    def is_moving(self):
        """Check whether any of the axes are currently moving"""
        pass

    # High-level commands #####################################################################################

    @abstractmethod
    def set_position(self, xyzq, laser_frame=True):
        """
        Move the stages to a desired position.

        :param xyzq: Desired position for each axis.
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

    Position    S/N         Axes
    Top         L010812     rotation, vertical (y), perpendicular-to-beam (coarse x)
    Bottom      L010810     perpendicular-to-beam (x), parallel-to-beam (z)
    """

    def __init__(self, verbose=False):
        """
        Instantiate Attocube object
        """
        import utils.Attocube.pyanc350v4 as pyanc

        super().__init__(verbose)

        pyanc.discover(1)
        self.dev_a = pyanc.Positioner(0)
        self.dev_b = pyanc.Positioner(1)

        self.devices = {
            'q': self.dev_a,
            'x': self.dev_b,
            'y': self.dev_a,
            'z': self.dev_b,
            'xc': self.dev_a
        }
        self.ax_indeces = {
            'q': 0,
            'x': 0,
            'y': 1,
            'z': 1,
            'xc': 2
        }
        self.fine_axes = ['x', 'y', 'z', 'q']
        self.all_axes = ['x', 'y', 'z', 'q', 'xc']
        self.coarse = 'xc'
        self.xc = 0.0

    def __del__(self):
        self.dev_a.disconnect()
        self.dev_b.disconnect()

    def set_position(self, xyzq, laser_frame=True):
        if laser_frame:
            x, y, z, q = xyzq
            q_rads = q * np.pi / 180
            xp = x*np.cos(q_rads) + z*np.sin(q_rads)
            zp = z*np.cos(q_rads) - x*np.sin(q_rads)
            x = xp
            z = zp
            xyzq = {'x': x, 'y': y, 'z': z, 'q': q}
        for ax in self.fine_axes:
            self.devices[ax].setTargetPosition(self.ax_indeces[ax], xyzq[ax])
            self.devices[ax].startAutoMove(self.ax_indeces[ax], 1, 0)
        while self.is_moving():
            time.sleep(0.1)
        for ax in self.fine_axes:
            self.devices[ax].startAutoMove(self.ax_indeces[ax], 0, 0)

        self.measure()
        pass

    def measure(self):
        positions = []
        for ax in self.fine_axes:
            pos = self.devices[ax].getPosition(self.ax_indeces[ax])
            if ax != 'q':
                pos *= 1000  # The Attocube controller returns meters
            positions.append(pos)
        self.x, self.y, self.z, self.q = tuple(positions)

        self.x0 = self.x * np.cos(self.q) - self.z * np.sin(self.q)
        self.z0 = self.x * np.sin(self.q) + self.z * np.cos(self.q)
        self.y0 = self.y
        return

    def check_errors(self):
        return {ax: self.get_status(ax)[-1] for ax in self.all_axes}

    def get_status(self, ax):
        connected, enabled, moving, target, eotFwd, eotBwd, error = self.devices[ax].getAxisStatus(self.ax_indeces[ax])
        return connected, enabled, moving, target, eotFwd, eotBwd, error

    def is_moving(self):
        for ax in self.all_axes:
            if self.get_status(ax)[2]:
                return True
        else:
            return False


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

        # The numerical index of each stage ON THE CONTROLLER
        self.x_ax = 3
        self.z_ax = 2
        self.y_ax = 1
        self.q_ax = 4

        # This just makes it easier to access each axis sequentially
        self.axes = [self.x_ax, self.y_ax, self.z_ax, self.q_ax]
        self.ax_names = {self.x_ax: 'X', self.y_ax: 'Y', self.z_ax: 'Z', self.q_ax: 'Q'}

        self.check_errors()
        self.measure()
        pass

    def __del__(self):
        self.port.flush()
        self.port.close()

    def set_position(self, xyzq, laser_frame=True):
        if laser_frame:
            x, y, z, q = xyzq
            q_rads = q * np.pi / 180
            xp = x*np.cos(q_rads) + z*np.sin(q_rads)
            zp = z*np.cos(q_rads) - x*np.sin(q_rads)
            x = xp
            z = zp
            xyzq = x, y, z, q
        for i, ax in enumerate(self.axes):
            self.command(f'{ax}MSA{xyzq[i]:0.6f}')
        self.command('0RUN')
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
        while self.is_moving():
            time.sleep(0.1)
        for ax in self.axes:
            pos = self.query(f'{ax}POS?')
            pos = pos.split(',')  # Parse the returned string
            pos = float(pos[1])
            positions.append(pos)

        self.x, self.y, self.z, self.q = tuple(positions)

        self.x0 = self.x * np.cos(self.q) - self.z * np.sin(self.q)
        self.z0 = self.x * np.sin(self.q) + self.z * np.cos(self.q)
        self.y0 = self.y
        return

    def check_errors(self):
        """Check each axis for errors"""
        for ax in self.axes:
            status = self.get_status(ax)
            self.print(f'{self.ax_names[ax]}-axis  (SB: {status})')
            err = self.query(f'{ax}ERR?')
            if len(err):
                for s in err.split('#'):
                    if len(s) and not s.isspace():
                        self.print(f'\t{s}')
            else:
                self.print('\tNo errors')

    def get_status(self, ax):
        """Check the status byte for a given axis"""
        s = int(self.query(f'{ax}STA?'))
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
        pass

    def measure(self):
        pass

    def check_errors(self):
        pass

    def get_status(self, ax):
        pass

    def is_moving(self):
        pass

    def set_position(self, xyzq, laser_frame=True):
        pass


if __name__ == '__main__':
    ctrl = Attocube(verbose=True)
    ctrl.get_stage_positions(True)
