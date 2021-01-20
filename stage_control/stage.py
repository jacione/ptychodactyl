"""
This code contains classes to control the sample positioning stages

Nick Porter
jacioneportier@gmail.com
"""
import numpy as np
import serial
import time


class StageController:
    """

    """
    def __init__(self):
        # Connect to the controller
        self.port = serial.Serial()
        self.port.open()

        # The numerical index of each stage ON THE CONTROLLER
        self.x_ax = 1
        self.y_ax = 2
        self.z_ax = 3
        self.q_ax = 4

        # These are the positions in the coordinate system of the STAGES
        self.x = 0
        self.y = 0
        self.z = 0

        # This is the rotational position which relates the two coordinate systems
        self.q = 0

        # These are the positions in the coordinate system of the BEAMLINE
        self.x0 = 0
        self.y0 = 0
        self.z0 = 0

        self.zero()
        pass

    def zero(self):
        # Move to zero the sample on the beamline
        # Rotate to align the y-axis parallel to the beamline
        self.command('0ZRO')
        self.measure()
        pass

    def home_xy(self):
        cmd = f'{self.x_ax}MSA0;{self.y_ax}MSA0 \n0RUN'
        self.command(cmd)
        self.measure()
        pass

    def move_ax(self, ax, pos):
        cmd = f'{ax}MVA{pos}'
        self.command(cmd)
        self.measure()
        return

    def move_x0(self, x0_pos):
        # Transform the desired h-position into the xyz coordinates
        x = x0_pos*np.cos(self.q)
        y = -x0_pos*np.sin(self.q)
        cmd = f'{self.x_ax}MSA{x};{self.y_ax}MSA{y} \n0RUN'
        self.command(cmd)
        self.measure()
        return

    def move_y0(self, y0_pos):
        # Transform the desired h-position into the xyz coordinates
        x = y0_pos*np.sin(self.q)
        y = y0_pos*np.cos(self.q)
        cmd = f'{self.x_ax}MSA{x};{self.y_ax}MSA{y} \n0RUN'
        self.command(cmd)
        self.measure()
        return

    def move_z0(self, z0_pos):
        cmd = f'{self.z_ax}MVA{z0_pos}'
        self.command(cmd)
        self.measure()
        return

    def measure(self):
        axes = [self.x_ax, self.y_ax, self.z_ax, self.q_ax]
        positions = []
        for ax in axes:
            pos_str = self.query(f'{ax}POS?')
            pos = pos_str  # Parse the returned string
            positions.append(pos)

        self.x, self.y, self.z, self.q = tuple(positions)

        self.x0 = self.x*np.cos(self.q) - self.y*np.sin(self.q)
        self.y0 = self.x*np.sin(self.q) + self.y*np.cos(self.q)
        self.z0 = self.z
        pass

    def command(self, cmd_str):
        self.port.write(f'{cmd_str}\n\r')
        pass

    def query(self, qry_str):
        self.port.write(f'{qry_str}\n\r')
        time.sleep(0.25)
        s = self.port.readline()
        return s
