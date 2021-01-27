"""
Script to control the mightex camera
"""


from ctypes import *
import numpy
from matplotlib import pyplot as plt
import time
import os


class Camera:
    def __init__(self):
        # The directory should be wherever the SDK dll file(s) are stored
        os.chdir("")
        self.dll = CDLL('SSClassic_USBCamera_SDK.dll')

        # Basic IO functions for connecting/disconnecting with the camera
        self.sdk_InitDevice = self.dll.SSClassicUSB_InitDevice
        self.sdk_InitDevice.argtypes = [c_void_p]
        self.sdk_InitDevice.restype = c_int

        self.sdk_AddCamera = self.dll.SSClassicUSB_AddCameraToWorkingSet
        self.sdk_AddCamera.argtypes = [c_int]
        self.sdk_AddCamera.restype = c_int

        self.sdk_StartCameraEngine = self.dll.SSClassicUSB_StartCameraEngine
        self.sdk_StartCameraEngine.argtypes = [c_void_p, c_int, c_int, c_int]
        self.sdk_StartCameraEngine.restype = c_int

        self.sdk_StopCameraEngine = self.dll.SSClassicUSB_StopCameraEngine
        self.sdk_StopCameraEngine.argtypes = [c_void_p]
        self.sdk_StopCameraEngine.restype = c_int

        self.sdk_UnInitDevice = self.dll.SSClassicUSB_UnInitDevice
        self.sdk_UnInitDevice.argtypes = [c_void_p]
        self.sdk_UnInitDevice.restype = c_int

        num_cameras = self.sdk_InitDevice()
        if num_cameras != 1:
            raise IOError(f'{num_cameras} cameras detected!')

        add = self.sdk_AddCamera(1)
        if add == -1:
            raise IOError('Could not add camera device to working set!')

        self.is_on = False

        return

    def __del__(self):
        if self.is_on:
            self.camera_off()
        self.sdk_UnInitDevice()

    def camera_on(self):
        if self.is_on:
            print('Camera engine already started!')
            return

        # The arguments are [GUIwindow=None, bitdepth=16, threads=4, callback=1)
        start = self.sdk_StartCameraEngine(None, 16, 4, 1)
        if start == -1:
            raise IOError('Could not start camera engine!')

        self.is_on = True
        print('SUCCESS: Camera engine started!')
        return

    def camera_off(self):
        if not self.is_on:
            print('Camera engine not started!')
            return
        stop = self.sdk_StopCameraEngine()
        if stop != -1:
            raise IOError('Could not stop camera engine!')

        self.is_on = False
        print('SUCCESS: Camera engine stopped!')
        return


if __name__ == '__main__':
    cam = Camera()
    cam.camera_on()
    cam.camera_off()
