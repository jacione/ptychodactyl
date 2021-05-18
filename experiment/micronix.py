"""
This code contains classes to control the sample positioning stages

Nick Porter
jacioneportier@gmail.com
"""
import numpy as np
import serial
import time


class MMC200:
    """
    Class to interface with a Micronix MMC-200 stage controller.
    """
    def __init__(self):
        # Connect to the controller
        self.port = serial.Serial('COM3', baudrate=38400, timeout=2)
        self.port.flush()

        # The numerical index of each stage ON THE CONTROLLER
        self.x_ax = 3
        self.z_ax = 2
        self.y_ax = 1
        self.q_ax = 4

        # This just makes it easier to access each axis sequentially
        self.axes = [self.x_ax, self.z_ax, self.y_ax, self.q_ax]
        self.ax_names = {self.x_ax: 'X', self.z_ax: 'Y', self.y_ax: 'Z', self.q_ax: 'Q'}

        # These are the measured positions in the coordinate system of the STAGES
        self.x = 0
        self.y = 0
        self.z = 0

        # This is the measured rotational position which relates the two coordinate systems
        self.q = 0

        # These are the calculated positions in the coordinate system of the BEAMLINE
        self.x0 = 0
        self.y0 = 0
        self.z0 = 0

        self.check()
        self.measure()
        pass

    def zero(self):
        # TODO: Move to zero the sample on the beamline
        # TODO: Rotate to align the y-axis parallel to the beamline
        self.command('0ZRO')
        self.measure(False)
        pass

    def home_all(self):
        self.move_to((0, 0, 0, 0))
        return

    def home_horizontal(self):
        cmd = f'{self.x_ax}MSA0;{self.z_ax}MSA0 \n0RUN'
        self.command(cmd)
        self.measure(False)
        pass

    def move_to(self, xyzq):
        for i, ax in enumerate(self.axes):
            self.command(f'{ax}MSA{xyzq[i]}')
        self.command('0RUN')
        self.measure(False)
        return

    def set_x(self, x_pos):
        cmd = f'{self.x_ax}MVA{x_pos:0.6f}'
        self.command(cmd)
        self.measure(False)
        return

    def set_x0(self, x0_pos):
        # Transform the desired h-position into the xyz coordinates
        q = self.q * np.pi / 180
        x = x0_pos*np.cos(q)
        y = -x0_pos*np.sin(q)
        self.command(f'{self.x_ax}MVA{x:0.6f}')
        while self.is_moving():
            time.sleep(0.1)
        self.command(f'{self.z_ax}MVA{y:0.6f}')
        self.measure(False)
        return

    def set_y(self, y_pos):
        cmd = f'{self.z_ax}MVA{y_pos:0.6f}'
        self.command(cmd)
        self.measure(False)
        return

    def set_y0(self, y_pos):
        self.set_y(y_pos)

    def set_z(self, z_pos):
        cmd = f'{self.y_ax}MVA{z_pos:0.6f}'
        self.command(cmd)
        self.measure(False)
        return

    def set_z0(self, z0_pos):
        # Transform the desired h-position into the xyz coordinates
        q = self.q * np.pi / 180
        x = z0_pos * np.sin(q)
        y = z0_pos * np.cos(q)
        self.command(f'{self.x_ax}MVA{x:0.6f}')
        while self.is_moving():
            time.sleep(0.1)
        self.command(f'{self.z_ax}MVA{y:0.6f}')
        self.measure(False)
        return

    def set_q(self, q_pos):
        cmd = f'{self.q_ax}MVA{q_pos:0.6f}'
        self.command(cmd)
        self.measure(False)
        pass

    def show_off(self):
        # Moves all the axes around simultaneously
        # ...just for fun
        self.move_to((5, 5, 5, 10))
        self.move_to((0, 0, 0, 0))

    def measure(self, print_lines=True):
        # Checks the encoder on each axis and updates the attributes within the class instance
        positions = []
        lines = 'AXIS'.ljust(12) + 'CALC'.ljust(12) + 'MEAS\n'
        while self.is_moving():
            time.sleep(0.1)
        for ax in self.axes:
            pos = self.query(f'{ax}POS?')
            pos = pos.split(',')  # Parse the returned string
            pos = (float(pos[0]), float(pos[1]))
            lines = lines + self.ax_names[ax].ljust(12) + str(pos[0]).ljust(12) + str(pos[1]) + '\n'
            positions.append(pos)

        self.x, self.y, self.z, self.q = tuple(positions)

        self.x0 = self.x * np.cos(self.q) - self.z * np.sin(self.q)
        self.z0 = self.x * np.sin(self.q) + self.z * np.cos(self.q)
        self.y0 = self.y
        if print_lines:
            print(lines)
        return

    def position(self):
        return self.x0, self.y0, self.q

    def check(self):
        # Checks each axis for errors
        for ax in self.axes:
            status = self.get_status(ax)
            print(f'{self.ax_names[ax]}-axis  (SB: {status})')
            err = self.query(f'{ax}ERR?')
            if len(err):
                for s in err.split('#'):
                    if len(s) and not s.isspace():
                        print(f'\t{s}')
            else:
                print('\tNo errors')

    def get_status(self, ax):
        # Checks the status byte for a given axis
        s = int(self.query(f'{ax}STA?'))
        status = f'{s:b}'.rjust(8, '0')
        return status

    def is_moving(self):
        # Checks each axis' status byte to see if any of them are moving.
        for ax in self.axes:
            if bool(int(self.get_status(ax)[1:4])):
                return True
        else:
            return False

    def command(self, cmd_str):
        # Sends a command string to the mightex stage
        self.port.write(bytes(f'{cmd_str}\n\r'.encode('utf-8')))
        pass

    def query(self, qry_str):
        # Sends a query string to the mightex stage and reads a response
        self.port.write(bytes(f'{qry_str}\n\r'.encode('utf-8')))
        time.sleep(0.1)
        s = str(self.port.readline())
        s = s.replace(r'\n', '')
        s = s.replace(r'\r', '')
        s = s.removeprefix("b'")
        s = s.removesuffix("'")
        s = s[s.find('#') + 1:]
        return s


if __name__ == '__main__':
    ctrl = MMC200()
    ctrl.show_off()
