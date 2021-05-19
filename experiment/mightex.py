"""
Script to control the mightex camera
"""

from ctypes import *
import numpy as np
from matplotlib import pyplot as plt


class Camera:
    def __init__(self, resolution=(2560, 1920), verbose=False):
        # The directory should be wherever the SDK dll file(s) are stored
        dir_lab = 'C:/Users/jacione/Documents/Mightex_SM_software/SDK/Lib/x64'
        dir_jnp = 'C:/Users/jacio/OneDrive/Documents/Research/mightex_sdk/SDK/Lib/x64'
        try:
            self.dll = CDLL(f'{dir_lab}/SSClassic_USBCamera_SDK.dll')
        except FileNotFoundError:
            self.dll = CDLL(f'{dir_jnp}/SSClassic_USBCamera_SDK.dll')

        self.is_on = False
        self.width = resolution[0]
        self.height = resolution[1]
        self.image_metadata_size = 128
        self.im_shape = (self.height, self.width)
        self.im_size = self.width * self.height
        self.data_size = self.im_size + self.image_metadata_size
        self.dtype = 'i2'
        self.verbose = verbose

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

        self.__sdk_SetExposure = self.dll.SSClassicUSB_SetExposureTime
        self.__sdk_SetExposure.argtypes = [c_int, c_int]  # [ID, exposure_time (x0.05 ms)]
        self.__sdk_SetExposure.restype = c_int
        self.exposure = 0.0  # milliseconds

        self.__sdk_SetGain = self.dll.SSClassicUSB_SetGains
        self.__sdk_SetGain.argtypes = [c_int, c_int, c_int, c_int]  # [ID, gain]
        self.__sdk_SetGain.restype = c_int
        self.gain = 0  # gain level (gain factor * 8)

        # As I understand it (which is barely) the frame hooker is what grabs the frame data?
        self.__sdk_InstallFrameHooker = self.dll.SSClassicUSB_InstallFrameHooker
        self.__sdk_InstallFrameHooker.argtypes = [c_int, c_void_p]  # [type=0, callback=None] (type=0 for raw data)
        self.__sdk_InstallFrameHooker.restype = c_int

        self.__sdk_GetCurrentFrame = self.dll.SSClassicUSB_GetCurrentFrame16bit
        # [type,    ID,     pointer to data]
        self.__sdk_GetCurrentFrame.argtypes = [c_int, c_int, np.ctypeslib.ndpointer(self.dtype, 1, (self.data_size,))]
        self.__sdk_GetCurrentFrame.restype = np.ctypeslib.ndpointer(self.dtype, 1, (self.data_size,))

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
        self.print('SUCCESS: Camera disconnected!')

    def __del__(self):
        if self.is_on:
            self.camera_off()
            self.__sdk_UnInitDevice()
            self.print('SUCCESS: Camera disconnected!')

    def print(self, text):
        if self.verbose:
            print(text)

    def camera_on(self):
        if self.is_on:
            self.print('Camera engine already started!')
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
        self.print('SUCCESS: Camera engine started!')
        return

    def camera_off(self):
        if not self.is_on:
            self.print('Camera engine not started!')
            return

        grab = self.__sdk_StopFrameGrab(1)
        if grab == -1:
            raise IOError('Could not terminate frame grab!')

        stop = self.__sdk_StopCameraEngine()
        if stop == -1:
            raise IOError('Could not terminate camera engine!')

        self.is_on = False
        self.print('SUCCESS: Camera engine stopped!')
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
        success = self.__sdk_SetGain(1, 0, gain, 0)
        if success == 1:
            self.gain = gain
        return

    def set_defaults(self):
        self.set_resolution(2560, 1920)
        self.set_exposure(2.0)
        self.set_gain(8)
        return

    def get_frame(self, show=False):
        if not self.is_on:
            self.print('Camera is not initialized!')
            self.camera_on()
        data = np.empty(self.data_size, dtype=self.dtype)
        trycount = 0
        while True:
            # For some reason this doesn't always read the whole image data on the lab computer.
            # It would be better to figure out why this happens and fix it at the root, but in the meantime I've added
            # some error handling to detect and discard incomplete measurements.
            # TODO: Figure out how to prevent failure in the first place.
            trycount += 1
            if trycount > 10:
                raise IOError('Camera API refuses to cooperate and refuses to explain why...')
            try:
                sptr = np.empty(self.data_size, dtype=self.dtype)
                self.__sdk_InstallFrameHooker(0, None)
                data = self.__sdk_GetCurrentFrame(0, 1, sptr)
                data = data[self.image_metadata_size:].reshape(self.im_shape)
                if np.min(np.sum(data, axis=1)) == 0:
                    self.print('Camera API: Partial failure, trying again...')
                    continue
            except ValueError:
                self.print('Camera API: Total failure, trying again...')
                continue
            self.print('Camera API: Image captured!')
            break
        if show:
            plt.subplot(111, xticks=[], yticks=[])
            plt.imshow(data, cmap='plasma')
            plt.tight_layout()
            plt.show()
        return data

    def analyze_frame(self):
        img = self.get_frame()
        cx = self.width // 2
        cy = self.height // 2
        x = img[cy]
        y = img[:, cx]
        plt.figure(tight_layout=True)
        plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=3, xticks=[], yticks=[])
        plt.imshow(img, cmap='gray')
        plt.axvline(cx, c='r')
        plt.axhline(cy, c='g')
        plt.subplot2grid((4, 4), (3, 0), colspan=3)
        plt.plot(x, c='g')
        plt.subplot2grid((4, 4), (0, 3), rowspan=3)
        plt.plot(y, np.arange(self.height), c='r')
        plt.show()


if __name__ == '__main__':
    with Camera() as cam:
        cam.set_defaults()
        cam.analyze_frame()
