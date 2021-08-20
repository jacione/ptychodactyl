"""
Controller classes for cameras.
"""
import time
from ctypes import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass
from abc import ABC, abstractmethod
import os
from skimage.transform import downscale_local_mean
from progressbar import ProgressBar, widgets


libs = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/') + '/utils'


def get_camera(camera_type, **kwargs):
    """
    Set up the camera with the correct controller subclass.

    .. note::
        If you write your own Camera subclass (recommended), be sure to add it to the dictionary within this function.
        Otherwise, you will get a ``KeyError``.

    :param camera_type: Type of camera to set up.
    :type camera_type: str
    :param kwargs: keyword arguments associated with type of camera
    :return: Camera controller object, subclass of camera.Camera
    :rtype: Camera
    """
    options = {
        'thorcam': ThorCam,
        'mightex': Mightex,
        'andor': Andor
    }
    return options[camera_type.lower()](**kwargs)


class Camera(ABC):
    """
    Abstract parent class for camera devices
    """
    @abstractmethod
    def __init__(self, defaults, verbose):
        self.defaults = defaults
        self.is_on = False
        self.is_armed = False
        self.verbose = verbose
        self.full_width = self.defaults['width']
        self.full_height = self.defaults['height']
        self.full_shape = (self.full_height, self.full_width)
        self.full_size = (self.full_height, self.full_width)
        self.binning = 1
        self.width = self.defaults['width']
        self.height = self.defaults['height']
        self.im_shape = (self.height, self.width)
        self.im_size = self.width * self.height
        self.pixel_size = self.defaults['pixel_size']
        self.frames_per_take = 1
        self.exposure = self.defaults['exposure']
        self.gain = self.defaults['gain']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.camera_off()

    def __del__(self):
        self.camera_off()

    @abstractmethod
    def camera_on(self):
        """Turn on the camera interface."""
        pass

    @abstractmethod
    def camera_off(self):
        """Turn off the camera interface."""
        pass

    @abstractmethod
    def set_exposure(self, ms):
        """
        Set the camera exposure time in milliseconds.

        .. note::
            Not all camera drivers use milliseconds to define exposure time. When implementing this in a subclass,
            remember to check the units that the device is expecting and convert your value if necessary.

        :param ms: Exposure time in milliseconds
        :type ms: float
        """
        pass

    @abstractmethod
    def set_gain(self, gain):
        """
        Set the camera's analog gain.

        .. note::
            Not all camera drivers use raw coefficients to define analog gain. When implementing this in a subclass,
            remember to check the format that the device is expecting and convert your value if necessary.

        :param gain: Analog gain
        :type gain: float
        """
        pass

    @abstractmethod
    def arm(self):
        """
        Arm the camera for image capture.

        Most cameras have some sort of arming function that must be called before any frames can be captured.
        """
        pass

    @abstractmethod
    def get_frame(self):
        """
        Capture and return a single frame from the image detector.

        :return: 2D image array
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def disarm(self):
        """
        Disarm the camera after image capture.
        """
        pass

    def get_frames(self, show=False):
        """
        Collect an image from the camera.

        :param show: If True, show the image (log scale) and a histogram of the image data.
        :type show: bool
        :return: 2D image array
        :rtype: np.ndarray
        """
        data = np.zeros(self.im_shape)
        self.arm()
        for i in range(self.frames_per_take):
            data = data + downscale_local_mean(self.get_frame(), (self.binning, self.binning))
        self.disarm()
        if show:
            self.imshow(np.log(data + 1))
            plt.hist(np.ravel(data), bins=100)
            plt.yscale('log')
            plt.show()
        return data

    @staticmethod
    def imshow(image):
        """A compact wrapper for pyplot's imshow with some commonly desired parameters preloaded"""
        plt.subplot(111, xticks=[], yticks=[])
        plt.imshow(image, cmap='plasma')
        plt.tight_layout()
        plt.show()

    def print(self, text):
        """A wrapper for print() that checks whether the instance is set to verbose"""
        if self.verbose:
            print(text)

    def set_defaults(self):
        """Set the basic camera features to their default values."""
        self.set_frames_per_take(1)
        self.set_resolution(self.full_height)
        self.set_exposure(self.defaults['exposure'])
        self.set_gain(self.defaults['gain'])
        return

    def set_frames_per_take(self, frames_per_take):
        """
        Set the number of frames to sum for each image measurement
        """
        self.frames_per_take = frames_per_take

    def set_resolution(self, resolution):
        """
        Set camera resolution. The functionality of this is a little counter-intuitive. The **resolution** parameter is
        the side-length of the desired diffraction data. This method will set the camera to bin each image as much as
        possible without reducing the height or width below **resolution**. The data will be cropped to the right
        size in CollectData.finalize() before saving. The binning happens here to reduce runtime memory usage,
        but the cropping has to happen there because it needs to be centered on the diffraction patterns.

        :param resolution: Desired side length of diffraction data.
        :type resolution: int
        """
        if resolution is not None:
            self.binning = self.full_height // resolution
        self.width = self.full_width // self.binning
        self.height = self.full_height // self.binning
        self.im_shape = (self.height, self.width)
        self.im_size = self.width * self.height
        self.pixel_size = self.defaults['pixel_size'] * self.binning
        return

    def find_center(self, img=None):
        """
        Find and return the "center of brightness" on the camera's image sensor. Useful for finding the center of a
        beam or diffraction pattern.
        """
        if img is None:
            img = self.get_frames()
        img[img < np.quantile(img, 0.75)] = 0
        return center_of_mass(img)

    def analyze_frame(self):
        """
        Take an image, then show the image, the center of brightness, and the centered horizontal and vertical line
        readouts.
        """
        img = self.get_frames()
        self.imshow(img)
        cy, cx = center_of_mass(img)
        x = img[int(np.round(cy, 0))]
        y = img[:, int(np.round(cx, 0))]
        plt.figure(tight_layout=True)
        plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=3, xticks=[], yticks=[])
        plt.imshow(img, cmap='bone')
        plt.axhline(cy, c='g')
        plt.axvline(cx, c='r')
        plt.subplot2grid((4, 4), (3, 0), colspan=3)
        plt.plot(x, c='g')
        plt.subplot2grid((4, 4), (0, 3), rowspan=3)
        plt.plot(np.flip(y), np.arange(self.height), c='r')
        plt.show()


class ThorCam(Camera):
    """
    Camera subclass that can interface with a Thorlabs camera.

    This subclass only implements a few basic functions which are relevant to ptychography data collection. A more
    complete control can be achieved by using the `officially provided SDK
    <https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam>`_.

    .. note::
        The Thorcam does support automatic image summation (sending several capture triggers and summing the results),
        but the summing appears to be performed in the camera's on-board memory, which means it has the same dynamic
        range as a single capture. I don't know if this can be changed/fixed, but for now at least the easy solution is
        to externally sum a series of single images.
    """

    def __init__(self, verbose=True):
        """
        Create a ThorCam object

        :param verbose: if True (default), this instance will print information about most processes as it runs.
        :type verbose: bool
        """

        from utils.ThorCam.tl_dotnet_wrapper import TL_SDK

        defaults = {
            'width': 3296,
            'height': 2472,
            'exposure': 0.25,
            'gain': 1,
            'pixel_size': 5.5
        }

        super().__init__(defaults, verbose)

        self._sdk = TL_SDK()
        self._handle = None

        self.camera_on()
        self.set_defaults()
        return

    def camera_on(self):
        """Turns on the camera"""
        if self.is_on:
            self.print('Camera engine already started!')
            return

        self._handle = self._sdk.open_camera(self._sdk.get_camera_list()[0])

        self.is_on = True
        self.print('SUCCESS: Camera engine started!')
        return

    def camera_off(self):
        """Turns off the camera"""
        if not self.is_on:
            self.print('Camera engine not started!')
            return

        self._handle.close()

        self.is_on = False
        self.print('SUCCESS: Camera engine stopped!')
        return

    def set_exposure(self, ms):
        """
        Set the camera exposure time in milliseconds.

        :param ms: exposure time in milliseconds
        :type ms: float
        """
        if ms is not None:
            us = int(ms * 1000)
            self._handle.set_exposure_time_us(us)
        return

    def set_gain(self, gain):
        """
        Set the camera analog gain

        :param gain: analog gain
        :type gain: int
        """
        if gain is not None:
            self._handle.set_gain(gain)
        return

    def arm(self):
        if not self.is_on:
            self.camera_on()
        self._handle.arm()
        self.is_armed = True
        return

    def get_frame(self):
        self._handle.issue_software_trigger()
        frame = None
        while frame is None:
            frame = self._handle.get_pending_frame_or_null()
        return self._handle.frame_to_array(frame)

    def disarm(self):
        self._handle.disarm()
        self.is_armed = False
        return


class Mightex(Camera):
    """
    Camera subclass that can interface with a Mightex camera.

    This subclass only implements a few basic functions which are relevant to ptychography data collection. A more
    complete control can be achieved by using the officially provided SDK.
    """

    def __init__(self, verbose=True):
        """
        Create a Mightex object.

        :param verbose: if True, this instance will print information about most processes as it runs. Default is True
        :type verbose: bool
        """
        defaults = {
            'width': 2560,
            'height': 1920,
            'exposure': 0.25,
            'gain': 8,
            'metadata': 128,
            'dtype': 'i4',
            'pixel_size': 2.2
        }
        super().__init__(defaults, verbose)
        self.dtype = self.defaults['dtype']
        self.image_metadata_size = self.defaults['metadata']
        self.data_size = self.im_size + self.image_metadata_size

        _dll = CDLL(f'{libs}/Mightex/SSClassic_USBCamera_SDK.dll')

        # """ Config constants """
        # For functions that return strings built on C char arrays, this is the max number of characters
        # _STRING_MAX = 4096
        #
        # """ Callback ctypes types """
        # _camera_connect_callback_type = CFUNCTYPE(None, c_char_p, c_int, c_void_p)
        # _camera_disconnect_callback_type = CFUNCTYPE(None, c_char_p, c_void_p)
        # _frame_available_callback_type = CFUNCTYPE(None, c_void_p, POINTER(c_ushort), c_int, POINTER(c_char), c_int,
        #                                            c_void_p)
        # # metadata is ASCII, so use c_char
        # _3x3Matrix_float = (c_float * 9)

        # Basic IO functions for connecting/disconnecting with the camera
        self._sdk_InitDevice = _dll.SSClassicUSB_InitDevice
        self._sdk_InitDevice.argtypes = []
        self._sdk_InitDevice.restype = c_int

        self._sdk_AddCamera = _dll.SSClassicUSB_AddDeviceToWorkingSet
        self._sdk_AddCamera.argtypes = [c_int]  # [ID]
        self._sdk_AddCamera.restype = c_int

        self._sdk_StartCameraEngine = _dll.SSClassicUSB_StartCameraEngine
        self._sdk_StartCameraEngine.argtypes = [c_void_p, c_int, c_int, c_int]  # [GUI, bitdepth, threads, callback]
        self._sdk_StartCameraEngine.restype = c_int

        self._sdk_StopCameraEngine = _dll.SSClassicUSB_StopCameraEngine
        self._sdk_StopCameraEngine.argtypes = []
        self._sdk_StopCameraEngine.restype = c_int

        self._sdk_UnInitDevice = _dll.SSClassicUSB_UnInitDevice
        self._sdk_UnInitDevice.argtypes = []
        self._sdk_UnInitDevice.restype = c_int

        # Methods for collecting frame data
        self._sdk_SetWorkMode = _dll.SSClassicUSB_SetCameraWorkMode
        self._sdk_SetWorkMode.argtypes = [c_int, c_int]  # [ID, mode=0] (0 for normal)
        self._sdk_SetWorkMode.restype = c_int

        self._sdk_StartFrameGrab = _dll.SSClassicUSB_StartFrameGrab
        self._sdk_StartFrameGrab.argtypes = [c_int, c_int]  # [ID, num_frames=0x8888] (0x8888 for continuous)
        self._sdk_StartFrameGrab.restype = c_int

        self._sdk_StopFrameGrab = _dll.SSClassicUSB_StopFrameGrab
        self._sdk_StopFrameGrab.argtypes = [c_int]  # [ID]
        self._sdk_StopFrameGrab.restype = c_int

        # Methods to define camera parameters
        self._sdk_SetResolution = _dll.SSClassicUSB_SetCustomizedResolution
        self._sdk_SetResolution.argtypes = [c_int, c_int, c_int, c_int, c_int]  # [ID, width, height, bin=0, binmode=0]
        self._sdk_SetResolution.restype = c_int

        self._sdk_SetXYStart = _dll.SSClassicUSB_SetXYStart
        self._sdk_SetXYStart.argtypes = [c_int, c_int, c_int]  # [ID, X-start, Y-start]
        self._sdk_SetXYStart.restype = c_int

        self._sdk_SetExposure = _dll.SSClassicUSB_SetExposureTime
        self._sdk_SetExposure.argtypes = [c_int, c_int]  # [ID, exposure_time (x0.05 ms)]
        self._sdk_SetExposure.restype = c_int

        self._sdk_SetGain = _dll.SSClassicUSB_SetGains
        self._sdk_SetGain.argtypes = [c_int, c_int, c_int, c_int]  # [ID, gain]
        self._sdk_SetGain.restype = c_int
        self.gain = 0  # gain level (gain factor * 8)

        # As I understand it (which is barely) the frame hooker is what grabs the frame data?
        self._sdk_InstallFrameHooker = _dll.SSClassicUSB_InstallFrameHooker
        self._sdk_InstallFrameHooker.argtypes = [c_int, c_void_p]  # [type=0, callback=None] (type=0 for raw data)
        self._sdk_InstallFrameHooker.restype = c_int

        self._sdk_GetCurrentFrame = _dll.SSClassicUSB_GetCurrentFrame16bit
        # [type,    ID,     pointer to data]
        self._sdk_GetCurrentFrame.argtypes = [c_int, c_int, np.ctypeslib.ndpointer(self.dtype, 1, (self.data_size,))]
        self._sdk_GetCurrentFrame.restype = np.ctypeslib.ndpointer(self.dtype, 1, (self.data_size,))

        self.camera_on()
        self.set_defaults()

        return

    def camera_on(self):
        """Start the camera engine"""
        if self.is_on:
            self.print('Camera engine already started!')
            return

        # Initialize the camera
        num_cameras = self._sdk_InitDevice()
        if num_cameras != 1:
            raise IOError(f'{num_cameras} cameras detected!')

        add = self._sdk_AddCamera(1)
        if add == -1:
            raise IOError('Could not add camera device to working set!')

        # The arguments are [GUI=None, bitdepth=16, threads=4, callback=1)
        start = self._sdk_StartCameraEngine(None, 16, 4, 1)
        if start == -1:
            raise IOError('Could not start camera engine!')

        mode = self._sdk_SetWorkMode(1, 0)
        if mode == -1:
            raise IOError('Could not set camera mode!')

        grab = self._sdk_StartFrameGrab(1, 0x8888)
        if grab == -1:
            raise IOError('Could not begin frame grab!')

        self.is_on = True
        self.print('SUCCESS: Camera engine started!')
        return

    def camera_off(self):
        """Stop the camera engine"""
        if not self.is_on:
            self.print('Camera engine not started!')
            return

        grab = self._sdk_StopFrameGrab(1)
        if grab == -1:
            raise IOError('Could not terminate frame grab!')

        stop = self._sdk_StopCameraEngine()
        if stop == -1:
            raise IOError('Could not terminate camera engine!')

        self._sdk_UnInitDevice()

        self.is_on = False
        self.print('SUCCESS: Camera disconnected!!')
        return

    def set_resolution(self, resolution):
        """
        Set camera resolution.

        :param resolution: Desired camera resolution (width, height)
        :type resolution: (int, int)
        """
        if resolution is not None:
            width = resolution[0]
            height = resolution[1]
            s1 = self._sdk_SetResolution(1, width, height, 0, 0)
            x_start = (self.full_width - width) // 2
            y_start = (self.full_height - height) // 2
            s2 = self._sdk_SetXYStart(1, x_start, y_start)
            if -1 not in [s1, s2]:
                self.width = width
                self.height = height
                self.im_size = self.width * self.height
                self.im_shape = (self.height, self.width)
        return

    def set_exposure(self, ms):
        """
        Set camera exposure time.

        :param ms: exposure time in milliseconds
        :type ms: float
        """
        if ms is not None:
            # Give a value in milliseconds;
            us50 = int(ms * 200)
            if us50 < 1:
                us50 = 1
            elif us50 > 15000:
                us50 = 15000
            success = self._sdk_SetExposure(1, us50)
            if success == 1:
                self.exposure = ms
        return

    def set_gain(self, gain):
        """
        Set camera gain.

        :param gain: gain factor in eighths (e.g. gain of 8 corresponds to 1x gain)
        :type gain: int
        """
        if gain is not None:
            # Gain is given in integer steps from 1-64, corresponding to a gain factor of 0.125x - 8x.
            # A gain of 8 corresponds to 1x gain.
            if gain < 1:
                gain = 1
            if gain > 64:
                gain = 64
            success = self._sdk_SetGain(1, 0, gain, 0)
            if success == 1:
                self.gain = gain
        return

    def arm(self):
        pass

    def get_frame(self):
        pass

    def disarm(self):
        pass

    def get_frames(self, num_frames=1, show=False):
        """
        Collect images from the camera.

        :param num_frames: number of frames to collect. *Currently has no effect*
        :type num_frames: int
        :param show: if True, show the image before returning it
        :type show: bool
        :return: 2D image array
        :rtype: np.ndarray
        """
        if not self.is_on:
            self.print('Camera is not initialized!')
            self.camera_on()
        data = np.empty(self.data_size, dtype=self.dtype)
        trycount = 0
        while True:
            # FIXME: For some reason this doesn't always read the whole image data on the lab computer.
            # It would be better to figure out why this happens and fix it at the root, but in the meantime I've added
            # some error handling to detect and discard incomplete measurements.
            # TODO: Figure out how to prevent failure in the first place.
            trycount += 1
            if trycount > 10:
                raise IOError('Camera API refuses to cooperate and refuses to explain why...')
            try:
                sptr = np.empty(self.data_size, dtype=self.dtype)
                self._sdk_InstallFrameHooker(0, None)
                data = self._sdk_GetCurrentFrame(0, 1, sptr)
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
            self.imshow(data)
        return data


class Andor(Camera):

    def __init__(self, verbose=False):
        from pyAndorSDK2.atmcd import atmcd
        defaults = {
            'width': 2048,
            'height': 2048,
            'exposure': 80,
            'gain': 0,
            'pixel_size': 13.5,
            'temperature': -60
        }
        super().__init__(defaults, verbose)
        self._sdk = atmcd()
        self.temp = self.defaults['temperature']

        self.camera_on()

    def camera_on(self):
        if self.is_on:
            self.print('Camera engine already started!')
            return

        self._sdk.Initialize("")
        self._sdk.SetAcquisitionMode(2)  # Accumulation/summing mode
        self._sdk.SetReadMode(4)  # Full image mode
        self._sdk.SetTriggerMode(0)  # Internally regulated triggering
        self._sdk.SetImage(1, 1, 1, self.full_width, 1, self.full_height)
        self._sdk.SetShutter(1, 0, 15, 15)
        self._sdk.CoolerON()
        self.set_defaults()

        self.is_on = True
        self.print('SUCCESS: Camera engine started!')
        pass

    def camera_off(self):
        if not self.is_on:
            self.print('Camera engine not started!')
            return

        self._sdk.ShutDown()

        self.is_on = False
        self.print('SUCCESS: Camera engine stopped!')
        return

    def set_temperature(self, temp):
        self.temp = temp
        self._sdk.SetTemperature(temp)
        ret, curr_temp = self._sdk.GetTemperature()
        print(f'Setting camera temperature to {self.temp} °C...')
        pbar = ProgressBar(
            max_value=curr_temp-self.temp+10,
            initial_value=curr_temp,
            widgets=[widgets.Variable('T', '{value} °C'), ' ', widgets.Bar(), ' ', widgets.Timer()],
            variables={'T': curr_temp}
        )
        time.sleep(0.1)
        while ret != self._sdk.DRV_TEMP_STABILIZED:
            time.sleep(1)
            ret, curr_temp = self._sdk.GetTemperature()
            pbar.update(curr_temp-self.temp+5, T=curr_temp)
        pbar.finish()

    def set_defaults(self):
        super().set_defaults()
        self.set_temperature(self.defaults['temperature'])

    def set_exposure(self, ms):
        if ms < 50:
            ms = 50
        seconds = ms / 1000
        self._sdk.SetExposureTime(seconds)
        self._sdk.SetAccumulationCycleTime(seconds + 0.1)
        pass

    def set_gain(self, gain):
        # Feature not supported by camera
        pass

    def set_frames_per_take(self, frames_per_take):
        self.frames_per_take = frames_per_take
        self._sdk.SetNumberAccumulations(frames_per_take)

    def arm(self):
        # The Andor SDK doesn't really have an arming function. The PrepareAcquisition function just allocates the
        # memory for the image(s). It's not necessary, but may speed up collection time in accumulation mode.
        self._sdk.PrepareAcquisition()
        self.is_armed = True
        pass

    def get_frame(self):
        # The AndorSDK supports automatic image summation with (unlike the ThorCam, I might add) corresponding dynamic
        # range increase, so there's no need for this function.
        pass

    def get_frames(self, show=False):
        self.arm()
        self._sdk.StartAcquisition()
        self._sdk.WaitForAcquisition()
        _, buffer = self._sdk.GetMostRecentImage(self.im_size)
        image = np.ctypeslib.as_array(buffer, self.im_size)
        image = np.reshape(image, self.im_shape)
        image = np.asarray(image, dtype='Q')
        self.disarm()
        if show:
            self.imshow(image)
        return image

    def disarm(self):
        self.is_armed = False
        pass


class YourCamera(Camera):

    def __init__(self, verbose):
        defaults = {
            'width': 0,         # Horizontal size of the detector (in pixels)
            'height': 0,        # Vertical size of the detector (in pixels)
            'exposure': 0,      # Exposure time (typically in milliseconds)
            'gain': 1,          # Analog gain (typically a unitless multiplier)
            'pixel_size': 1     # Pixel side length (typically in microns)
        }
        super().__init__(defaults, verbose)
        pass

    def camera_on(self):
        pass

    def camera_off(self):
        pass

    def set_exposure(self, ms):
        pass

    def set_gain(self, gain):
        pass

    def arm(self):
        pass

    def get_frame(self):
        pass

    def disarm(self):
        pass


if __name__ == '__main__':
    cam = Andor(False)
    cam.analyze_frame()
