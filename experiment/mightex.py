"""
Script to control the mightex camera
"""

from ctypes import *
import numpy as np
from matplotlib import pyplot as plt
import time
import os


class Camera:
    def __init__(self, resolution=(2560, 1920)):
        # The directory should be wherever the SDK dll file(s) are stored
        # filedir = 'C:/Users/jacione/Documents/Mightex_SM_software/SDK/Lib/x64'
        filedir = 'C:/Users/jacio/OneDrive/Documents/Research/mightex_sdk/SDK/Lib/x64'
        self.dll = CDLL(f'{filedir}/SSClassic_USBCamera_SDK.dll')

        self.is_on = False
        self.dtype = 'i2'

        # Basic IO functions for connecting/disconnecting with the camera
        self.__sdk_InitDevice = self.dll.SSClassicUSB_InitDevice
        self.__sdk_InitDevice.argtypes = []
        self.__sdk_InitDevice.restype = c_int

        self.__sdk_AddCamera = self.dll.SSClassicUSB_AddDeviceToWorkingSet
        self.__sdk_AddCamera.argtypes = [c_int]  # [ID]
        self.__sdk_AddCamera.restype = c_int

        self.__sdk_StartCameraEngine = self.dll.SSClassicUSB_StartCameraEngine
        self.__sdk_StartCameraEngine.argtypes = [c_void_p, c_int, c_int, c_int]  # [GUI, bitdepth, threads, callback]
        self.__sdk_StartCameraEngine.restype = c_int

        self.__sdk_StopCameraEngine = self.dll.SSClassicUSB_StopCameraEngine
        self.__sdk_StopCameraEngine.argtypes = []
        self.__sdk_StopCameraEngine.restype = c_int

        self.__sdk_UnInitDevice = self.dll.SSClassicUSB_UnInitDevice
        self.__sdk_UnInitDevice.argtypes = []
        self.__sdk_UnInitDevice.restype = c_int

        # Methods for collecting frame data
        self.__sdk_SetWorkMode = self.dll.SSClassicUSB_SetCameraWorkMode
        self.__sdk_SetWorkMode.argtypes = [c_int, c_int]  # [ID, mode=0] (0 for normal)
        self.__sdk_SetWorkMode.restype = c_int

        self.__sdk_StartFrameGrab = self.dll.SSClassicUSB_StartFrameGrab
        self.__sdk_StartFrameGrab.argtypes = [c_int, c_int]  # [ID, num_frames=0x8888] (0x8888 for continuous)
        self.__sdk_StartFrameGrab.restype = c_int

        self.__sdk_StopFrameGrab = self.dll.SSClassicUSB_StopFrameGrab
        self.__sdk_StopFrameGrab.argtypes = [c_int]  # [ID]
        self.__sdk_StopFrameGrab.restype = c_int

        # Methods to define camera parameters
        self.__sdk_SetResolution = self.dll.SSClassicUSB_SetCustomizedResolution
        self.__sdk_SetResolution.argtypes = [c_int, c_int, c_int, c_int, c_int]  # [ID, width, height, bin=0, binmode=0]
        self.__sdk_SetResolution.restype = c_int
        self.width = resolution[0]
        self.height = resolution[1]
        self.im_shape = (self.height, self.width)
        self.im_size = self.width * self.height

        self.__sdk_SetExposure = self.dll.SSClassicUSB_SetExposureTime
        self.__sdk_SetExposure.argtypes = [c_int, c_int]  # [ID, exposure_time (x0.05 ms)]
        self.__sdk_SetExposure.restype = c_int
        self.exposure = 0.0  # milliseconds

        self.__sdk_SetGain = self.dll.SSClassicUSB_SetGains
        self.__sdk_SetGain.argtypes = [c_int, c_int]  # [ID, gain]
        self.__sdk_SetGain.restype = c_int
        self.gain = 0  # gain level (gain factor * 8)

        # As I understand it (which is barely) the frame hooker is what grabs the frame data?
        self.__sdk_InstallFrameHooker = self.dll.SSClassicUSB_InstallFrameHooker
        self.__sdk_InstallFrameHooker.argtypes = [c_int, c_void_p]  # [type=0, callback=None] (type=0 for raw data)
        self.__sdk_InstallFrameHooker.restype = c_int

        self.__sdk_GetCurrentFrame = self.dll.SSClassicUSB_GetCurrentFrame16bit
        self.__sdk_GetCurrentFrame.argtypes = [c_int, c_int, np.ctypeslib.ndpointer(self.dtype, 1, (self.im_size+127,))]
                                            # [type,    ID,     pointer to data]
        self.__sdk_GetCurrentFrame.restype = np.ctypeslib.ndpointer(self.dtype, 1, (self.im_size+127,))

        # Initialize the camera
        num_cameras = self.__sdk_InitDevice()
        if num_cameras != 1:
            raise IOError(f'{num_cameras} cameras detected!')

        add = self.__sdk_AddCamera(1)
        if add == -1:
            raise IOError('Could not add camera device to working set!')

        self.camera_on()
        self.set_defaults()

        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_on:
            self.camera_off()
        self.__sdk_UnInitDevice()
        print('SUCCESS: Camera disconnected!')

    def __del__(self):
        if self.is_on:
            self.camera_off()
            self.__sdk_UnInitDevice()
            print('SUCCESS: Camera disconnected!')

    def camera_on(self):
        if self.is_on:
            print('Camera engine already started!')
            return

        # The arguments are [GUI=None, bitdepth=16, threads=4, callback=1)
        start = self.__sdk_StartCameraEngine(None, 16, 4, 1)
        if start == -1:
            raise IOError('Could not start camera engine!')

        mode = self.__sdk_SetWorkMode(1, 0)
        if mode == -1:
            raise IOError('Could not set camera mode!')

        grab = self.__sdk_StartFrameGrab(1, 0x8888)
        if grab == -1:
            raise IOError('Could not begin frame grab!')

        self.is_on = True
        print('SUCCESS: Camera engine started!')
        return

    def camera_off(self):
        if not self.is_on:
            print('Camera engine not started!')
            return

        grab = self.__sdk_StopFrameGrab(1)
        if grab == -1:
            raise IOError('Could not terminate frame grab!')

        stop = self.__sdk_StopCameraEngine()
        if stop == -1:
            raise IOError('Could not terminate camera engine!')

        self.is_on = False
        print('SUCCESS: Camera engine stopped!')
        return

    def set_resolution(self, width: int, height: int):
        success = self.__sdk_SetResolution(1, width, height, 0, 0)
        if success == 1:
            self.width = width
            self.height = height
            self.im_size = self.width * self.height
            self.im_shape = (self.height, self.width)
        return

    def set_exposure(self, ms):
        # Give a value in milliseconds;
        us50 = int(ms * 200)
        if us50 < 1:
            us50 = 1
        elif us50 > 15000:
            us50 = 15000
        success = self.__sdk_SetExposure(1, us50)
        if success == 1:
            self.exposure = ms
        return

    def set_gain(self, gain: int):
        # Gain is given in integer steps from 1-64, corresponding to a gain factor of 0.125x - 8x.
        # A gain of 8 corresponds to 1x gain.
        if gain < 1:
            gain = 1
        if gain > 64:
            gain = 64
        success = self.__sdk_SetGain(1, gain)
        if success == 1:
            self.gain = gain
        return

    def set_defaults(self):
        self.set_resolution(2560, 1920)
        self.set_exposure(5.1)
        self.set_gain(8)
        return

    def get_frame(self):
        if not self.is_on:
            print('Camera is not initialized!')
        sptr = np.empty(self.im_size+127, dtype=self.dtype)
        self.__sdk_InstallFrameHooker(1, None)
        data = self.__sdk_GetCurrentFrame(0, 1, sptr)
        # FIXME: For some reason this doesn't always read the whole image data.

        data = data[127:].reshape(self.im_shape)
        plt.imshow(data, clim=[80, 125])
        plt.tight_layout()
        plt.show()
        return data


if __name__ == '__main__':
    with Camera() as cam:
        cam.set_exposure(300)
        cam.get_frame()
