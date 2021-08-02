"""
Controller classes for motorized stages.

*TODO: missing an abstract base class*

Currently implemented:
    - Micronix MMC-200 (stable)
"""
import numpy as np
import serial
import time


class MMC200:
    """
    Object-oriented interface for a Micronix MMC-200 stage controller. Axes are defined **in terms of the beamline**
    which means that the y-axis in this class corresponds to the vertical stage, and the z-axis is horizontal and
    parallel to the beam. This is different from how the axes are defined in the stages themselves, where z is assumed
    to be vertical.

    .. note::
        This class only implements a few basic functions
        which are relevant to ptychography data collection. For more advanced functionality, you can use the ``command``
        method to send specific serial commands using the supported command line syntax
        (https://micronixusa.com/product/download/evPpvw/universal-document/Avj2vR)
    """
    def __init__(self, port='COM3', verbose=False):
        """
        Create an MMC200 object to interface with a Micronix MMC-200 stage controller.

        :param verbose: if True, this instance will print information about most processes as it runs. Default is False.
        :type verbose: bool
        """
        self.verbose = verbose
        # Connect to the controller
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

        # These are the measured positions in the coordinate system of the STAGES
        self.x = 0
        self.y = 0
        self.z = 0

        # This is the measured rotational get_position which relates the two coordinate systems
        self.q = 0

        # These are the calculated positions in the coordinate system of the BEAMLINE
        self.x0 = 0
        self.y0 = 0
        self.z0 = 0

        self.check_errors()
        self.measure()
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.port.flush()
        self.port.close()

    def __del__(self):
        self.port.flush()
        self.port.close()

    # High-level commands #####################################################################################

    def set_position(self, xyzq, laser_frame=True):
        """
        Move the stages to a desired position.

        :param xyzq: Desired position for each axis.
        :type xyzq: (float, float, float, float)
        :param laser_frame: if True, convert positions to be in the laser's reference frame. Default is True.
        :type laser_frame: bool
        """
        x, y, z, q = xyzq
        if laser_frame:
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

    def get_position(self):
        """
        Get the current position of the stages.

        :return: Stage positions in the beam's reference frame (vertical, horizontal, rotational)
        :rtype: np.ndarray
        """
        self.print(f'\nAXIS   POS(mm)\n'
                   f'  X    {self.x0:0.6f}\n'
                   f'  Y    {self.y0:0.6f}\n'
                   f'  Z    {self.z0:0.6f}\n'
                   f'  Q    {self.q:0.6f}')
        return np.array([self.y0, self.x0, self.q])

    def home_all(self):
        """A quick command to reset all stage positions"""
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

    # Low-level interface ##################################################################################
    # These can be used for more advanced features using the command line syntax provided by Micronix:
    # https://micronixusa.com/product/download/evPpvw/universal-document/Avj2vR

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

    # Axis-specific commands ##################################################################################

    def set_x0(self, x0_pos):
        """Move the sample to a desired position along the x0-axis (horizontal, orthogonal to the beamline)"""
        q = self.q * np.pi / 180
        x = x0_pos * np.cos(q)
        z = -x0_pos * np.sin(q)
        self.command(f'{self.x_ax}MVA{x:0.6f}')
        while self.is_moving():
            time.sleep(0.1)
        self.command(f'{self.z_ax}MVA{z:0.6f}')
        self.measure()
        return

    def set_y0(self, y_pos):
        """Move the sample to a desired position along the y0-axis (vertical, orthogonal to the beamline)"""
        self.set_y(y_pos)

    def set_z0(self, z0_pos):
        """Move the sample to a desired position along the x0-axis (horizontal, parallel to the beamline)"""
        q = self.q * np.pi / 180
        x = z0_pos * np.sin(q)
        z = z0_pos * np.cos(q)
        self.command(f'{self.x_ax}MVA{x:0.6f}')
        while self.is_moving():
            time.sleep(0.1)
        self.command(f'{self.z_ax}MVA{z:0.6f}')
        self.measure()
        return

    def set_q(self, q_pos):
        """Rotate the stack to a desired angle"""
        cmd = f'{self.q_ax}MVA{q_pos:0.6f}'
        self.command(cmd)
        self.measure()
        pass

    def set_x(self, x_pos):
        """Move the x-stage to a desired position"""
        cmd = f'{self.x_ax}MVA{x_pos:0.6f}'
        self.command(cmd)
        self.measure()
        return

    def set_y(self, y_pos):
        """Move the y-stage to a desired position"""
        cmd = f'{self.y_ax}MVA{y_pos:0.6f}'
        self.command(cmd)
        self.measure()
        return

    def set_z(self, z_pos):
        """Move the z-stage to a desired position"""
        cmd = f'{self.z_ax}MVA{z_pos:0.6f}'
        self.command(cmd)
        self.measure()
        return

    def home_horizontal(self):
        """A quick command to reset horizontal stage positions"""
        cmd = f'{self.x_ax}MSA0;{self.z_ax}MSA0 \n0RUN'
        self.command(cmd)
        self.measure()
        pass

    def get_translation(self):
        """A quick command to get only the positions orthogonal to the beam"""
        return np.array([self.y0, self.x0])

    def get_rotation(self):
        """A quick command to get only the rotational position"""
        return self.q

    # Status queries ########################################################################

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

    # Other commands #######################################################################################
    # Some of these features may not be complete.

    def print(self, text):
        """A wrapper for print() that checks whether the instance is set to verbose"""
        if self.verbose:
            print(text)

    # def zero(self):
    #     # TODO: Move to zero the sample on the beamline
    #     # TODO: Rotate to align the y-axis parallel to the beamline
    #     # self.command('0ZRO')
    #     # self.measure()
    #     pass


if __name__ == '__main__':
    ctrl = MMC200(verbose=True)
    ctrl.home_all()
