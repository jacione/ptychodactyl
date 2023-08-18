"""
Controller classes for cameras.
"""
import time
from ctypes import *
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from threading import Thread
from queue import Queue

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from progressbar import ProgressBar, widgets

from ptycho.utils import crop_to_square


libs = Path(__file__).parents[1] / "libs"
for p in libs.iterdir():
    if p.is_dir():
        sys.path.append(p.as_posix())

THORCAM = "thorcam"
MIGHTEX = "mightex"
ANDOR = "andor"


class CameraManager:
    def __init__(self):
        self.valid_keys = (THORCAM, MIGHTEX, ANDOR)
        self._sdk = {name: None for name in self.valid_keys}
        self._registry = {name: 0 for name in self.valid_keys}
        self.cameras = []

    def __getitem__(self, item):
        return self.cameras[item]

    def __del__(self):
        self.close_all()

    def add_camera(self, key, **kwargs):
        if self._sdk[key] is None:
            self._open_sdk(key)

        if key == THORCAM:
            self.cameras.append(ThorCam(self._sdk[key], **kwargs))
        elif key == MIGHTEX:
            pass
        elif key == ANDOR:
            self.cameras.append(Andor(self._sdk[key], **kwargs))

        self._registry[key] += 1
        return self.cameras[-1]

    def remove_camera(self, index):
        key = self.cameras[index].key
        self.cameras[index].camera_off()
        self._registry[key] -= 1
        if not self._registry[key]:
            self._close_sdk(key)

    def close_all(self):
        for i, cam in enumerate(self.cameras):
            if cam.is_on:
                self.remove_camera(i)

    def get_frames(self, cam_index=None):
        if len(self.cameras) == 1:
            cam_index = 0
        if cam_index is not None:
            return [self.cameras[cam_index].get_frames()]
        threads = [Thread(target=cam.add_frame_to_queue) for cam in self.cameras]
        [t.start() for t in threads]
        [t.join() for t in threads]
        return [cam.get_frame_from_queue() for cam in self.cameras]

    def get_frames_from_queue(self):
        return [cam.get_frame_from_queue() for cam in self.cameras]

    def _open_sdk(self, key):
        if key not in self.valid_keys:
            raise KeyError(f"\"{key}\" is not a valid camera key")
        if key == THORCAM:
            from libs.ThorCam.tl_dotnet_wrapper import TL_SDK
            self._sdk[key] = TL_SDK()
            print("Opened TLCameraSDK")
            print(f"Available devices: {self._sdk[key].get_camera_list()}")
        elif key == MIGHTEX:
            pass
        elif key == ANDOR:
            from pyAndorSDK2 import atmcd
            self._sdk[key] = atmcd

    def _close_sdk(self, key):
        if key not in self.valid_keys:
            raise KeyError(f"\"{key}\" is not a valid camera key")

        if key == THORCAM:
            self._sdk[key].close()
        elif key == MIGHTEX:
            pass
        elif key == ANDOR:
            pass

        self._sdk[key] = None


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
        'andor': Andor,
        'yourcamera': YourCamera
    }
    return options[camera_type.lower()](**kwargs)


class Camera(ABC):
    """
    Abstract parent class for camera devices
    """
    @abstractmethod
    def __init__(self, camera_specs, verbose):
        """
        Because this is an abstract class, this constructor must be explicitly called within a subclass constructor.
        At the very least, a subclass constructor should

        #. define a ``defaults`` dict,
        #. call this function as ``super().__init__(defaults, verbose)``, and
        #. call ``self.camera_on()``.

        These three steps are already implemented into the ``YourCamera`` stubclass.

        When subclassing, you may want to include other parameters (such as temperature control target) in your
        constructor. These are OK as long as they are keyword arguments (i.e. they have a default value). They can then
        be passed to ``get_camera`` without any issue.

        :param camera_specs: Default camera parameters. Must include the following keys: ["width", "height", "exposure",
            "gain", "pixel_size", or will raise a ``KeyError``.
        :type camera_specs: dict
        :param verbose: Whether to print lots of information about camera processes.
        :type verbose: bool
        """
        self.key = None
        self.is_on = False
        self.is_armed = False
        self.verbose = verbose

        try:
            # Specs inherent to the camera
            self._width = camera_specs['width']
            self._height = camera_specs['height']
            self._shape = np.array([self._height, self._width])
            self._nbits = camera_specs['nbits']
            self._pixel_size = camera_specs['pixel_size']
        except KeyError:
            raise KeyError(f'The dict containing the default parameters did not contain all the expected keys.\n'
                           f'\tExpected: ["width", "height", "pixel_size", "nbits"]\n'
                           f'\tFound:    {list(camera_specs.keys())}')

        # Camera settings
        self._settings = {
            "binning": 1,
            "im_size": min(self._shape),
            "accums": 1,
            "exposure": 0.25,
            "gain": 1
        }

        self._queue = Queue()

    @abstractmethod
    def camera_off(self):
        """
        Turns off the camera interface.

        This is where you should dispose of any handles to the device. After this function is called, all camera
        functions should be disabled (except ``self.camera_on()``), and the garbage collector should only have to clean
        up the class object itself.
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
        Collect a composite image from the camera. The returned image will be a sum of *N* frames, where *N* is set
        using ``self.set_frames_per_take()``.

        :param show: If True, show the image (log scale) and a histogram of the image data.
        :type show: bool
        :return: 2D image array
        :rtype: np.ndarray
        """
        data = np.zeros((self._settings["im_size"], self._settings["im_size"]))
        self.arm()
        for i in range(self._settings["accums"]):
            frame = self.get_frame()
            data = data + crop_to_square(frame, self._settings["im_size"])
        self.disarm()
        if show:
            plt.subplot(111, xticks=[], yticks=[])
            plt.imshow(np.log(data+1), cmap='plasma')
            plt.tight_layout()
            plt.figure()
            plt.hist(np.ravel(data), bins=100)
            plt.yscale('log')
            plt.show()
        return data

    def add_frame_to_queue(self):
        self._queue.put(self.get_frames())

    def get_frame_from_queue(self):
        if not self._queue.empty():
            return self._queue.get()
        else:
            raise IOError("No frame in camera queue.")

    def clear_queue(self):
        self._queue = Queue()

    @staticmethod
    def imshow(image):
        """A compact wrapper for pyplot's imshow with some commonly desired parameters preloaded"""
        plt.subplot(111, xticks=[], yticks=[])
        plt.imshow(image, cmap='plasma')
        plt.tight_layout()
        plt.show()

    def get_specs(self):
        return {
            'width': self._width,
            'height': self._height,
            'x_pixel_size': self._pixel_size,
            'y_pixel_size': self._pixel_size,
            'nbits': self._nbits
        }

    def set(self, **kwargs):
        for setting, value in kwargs.items():
            try:
                self._settings[setting] = value
            except KeyError:
                raise RuntimeWarning(f"Setting {setting} does not exist on {self.key} camera.")
        self._set_all()

    @property
    def settings(self):
        return self._settings

    @abstractmethod
    def _set_all(self):
        pass

    def print(self, text):
        """A wrapper for print() that checks whether the instance is set to verbose"""
        if self.verbose:
            print(text)

    def set_defaults(self):
        """Set the basic camera features to their default values."""
        self.set(binning=1, im_shape=self._shape, accums=1, exposure=0.25, gain=1)
        return

    def find_center(self, img=None):
        """
        Find and return the "center of brightness" on the camera's image sensor. Useful for finding the center of a
        beam or diffraction pattern.
        """
        if img is None:
            img = self.get_frames()
        img = np.array(img, dtype='Q')
        img[img < np.quantile(img, 0.99)] = 0
        return ndi.center_of_mass(img)

    def analyze_frame(self):
        """
        Take an image, then show the image, the center of brightness, and the centered horizontal and vertical line
        readouts.
        """
        img = self.get_frames()
        cy, cx = self.find_center(img)
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
        plt.plot(np.flip(y), np.arange(self._settings["im_size"]), c='r')
        plt.show()

    def running_analysis(self):
        plt.figure(tight_layout=True)
        while True:
            img = self.get_frames()
            cy, cx = self.find_center(img)
            x = img[int(np.round(cy, 0))]
            y = img[:, int(np.round(cx, 0))]
            plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=3, xticks=[], yticks=[])
            plt.imshow(np.log(img+1), cmap='bone')
            plt.axhline(cy, c='g')
            plt.axvline(cx, c='r')
            plt.subplot2grid((4, 4), (3, 0), colspan=3)
            plt.plot(x, c='g')
            plt.subplot2grid((4, 4), (0, 3), rowspan=3)
            plt.plot(np.flip(y), np.arange(self._height), c='r')
            plt.draw()
            plt.pause(0.1)


class ThorCam(Camera):
    """
    Camera subclass that can interface with a Thorlabs camera.

    This subclass only implements a few basic functions which are relevant to ptycho data collection. A more
    complete control can be achieved by using the `officially provided SDK
    <https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam>`_.

    .. note::
        The Thorcam does support automatic image summation (sending several capture triggers and summing the results),
        but the summing appears to be performed in the camera's on-board memory, which means it has the same dynamic
        range as a single capture. I don't know if this can be changed/fixed, but for now at least the easy solution is
        to externally sum a series of single images.
    """

    def __init__(self, sdk, serial_num, verbose=False):
        """
        Create a ThorCam object

        :param verbose: if True (default), this instance will print information about most processes as it runs.
        :type verbose: bool
        """

        camera_specs = {
            "08949": {'width': 3296, 'height': 2472, 'pixel_size': 5.5, 'nbits': 14},
            "21723": {'width': 1440, 'height': 1080, 'pixel_size': 3.45, 'nbits': 10},
        }

        super().__init__(camera_specs[serial_num], verbose)

        self.key = THORCAM
        self._handle = sdk.open_camera(serial_num)

        self.set_defaults()
        print("Camera created")
        self.is_on = True
        return

    def camera_off(self):
        """Turns off the camera"""
        self.print("Attempting to close camera...")
        if not self.is_on:
            self.print('Camera engine not started!')
            return

        self._handle.close()

        self.is_on = False
        self.print('SUCCESS: Camera engine stopped!')
        return

    def _set_all(self):
        self._handle.set_exposure_time_us(int(self._settings["exposure"] * 10 ** 6))
        self._handle.set_gain(self._settings["gain"])

    def arm(self):
        self._handle.arm()
        self.is_armed = True
        return

    def get_frame(self):
        self._handle.issue_software_trigger()
        frame = None
        while frame is None:
            frame = self._handle.get_pending_frame_or_null()
        print(frame)
        return self._handle.frame_to_array(frame)

    def disarm(self):
        self._handle.disarm()
        self.is_armed = False
        return


class Mightex(Camera):
    """
    Camera subclass that can interface with a Mightex camera.

    This subclass only implements a few basic functions which are relevant to ptycho data collection. A more
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
        self.dtype = self.specs['dtype']
        self.image_metadata_size = self.specs['metadata']
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
            x_start = (self._width - width) // 2
            y_start = (self._height - height) // 2
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

    def __init__(self, sdk, verbose=False, temperature=-59, hold_temp=True):
        defaults = {
            'width': 2048,
            'height': 2048,
            'exposure': 80,
            'gain': 0,
            'pixel_size': 13.5,
            'temperature': temperature,
            'hold_temp': hold_temp,
        }
        super().__init__(defaults, verbose)
        self.key = ANDOR
        self._handle = sdk.atmcd()
        self._settings["temperature"] = temperature
        self.hold_temp = hold_temp

        self._handle.Initialize("")
        self._handle.SetAcquisitionMode(1)  # Single image capture
        self._handle.SetReadMode(4)  # Full image mode
        self._handle.SetTriggerMode(0)  # Internally regulated triggering
        self._handle.SetImage(1, 1, 1, self._width, 1, self._height)
        self._handle.SetShutter(1, 0, 15, 15)
        self._handle.CoolerON()
        self._handle.SetCoolerMode(int(self.hold_temp))
        self.set_defaults()

        self.print('SUCCESS: Camera engine started!')

    def camera_off(self):
        if not self.is_on:
            self.print('Camera engine not started!')
            return

        self._handle.ShutDown()

        self.is_on = False
        self.print('SUCCESS: Camera engine stopped!')
        return

    def set_temperature(self, temp):
        self.temp = temp
        self._handle.SetTemperature(temp)
        ret, curr_temp = self._handle.GetTemperature()
        if self.temp == curr_temp:
            return
        print(f'Setting camera temperature to {self.temp} °C...')
        pbar = ProgressBar(
            max_value=curr_temp-self.temp+10,
            initial_value=curr_temp,
            widgets=[widgets.Variable('T', '{value} °C'), ' ', widgets.Bar(), ' ', widgets.Timer()],
            variables={'T': curr_temp}
        )
        time.sleep(0.1)
        while ret != self._handle.DRV_TEMP_STABILIZED or abs(curr_temp - self.temp) > 1:
            time.sleep(1)
            ret, curr_temp = self._handle.GetTemperature()
            t_val = curr_temp-self.temp+5
            if t_val > pbar.max_value:
                t_val = pbar.max_value
            if t_val < 0:
                t_val = 0
            pbar.update(t_val, T=curr_temp)
        pbar.finish()

    def set_defaults(self):
        super().set_defaults()
        self.set_temperature(self.specs['temperature'])

    def set_exposure(self, ms):
        if ms < 50:
            ms = 50
        seconds = ms / 1000
        self._handle.SetExposureTime(seconds)
        return

    def set_gain(self, gain):
        # Feature not supported by camera
        pass

    def set_accums(self, accums):
        self.frames_per_take = accums
        return

    def arm(self):
        # The Andor SDK doesn't really have an arming function. The PrepareAcquisition function just allocates the
        # memory for the image(s). It's not necessary, but may speed up collection time in accumulation mode.
        self.is_armed = True
        return

    def get_frame(self):
        # The AndorSDK supports automatic image summation with (unlike the ThorCam, I might add) corresponding dynamic
        # range increase, so there's no need for this function.
        self._handle.StartAcquisition()
        self._handle.WaitForAcquisition()
        _, buffer = self._handle.GetMostRecentImage(self.size)
        image = np.ctypeslib.as_array(buffer, self.size)
        image = np.reshape(image, self._shape)
        return image

    def disarm(self):
        # The AndorSDK doesn't have an explicit arming/disarming sequence
        self.is_armed = False
        return


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
        self.camera_on()
        return

    def camera_on(self):
        if self.is_on:
            self.print('Camera engine already started!')
            return

        # Your code goes here!

        self.is_on = True
        self.print('SUCCESS: Camera engine started!')
        return

    def camera_off(self):
        if not self.is_on:
            self.print('Camera engine not started!')
            return

        # Your code goes here!

        self.is_on = False
        self.print('SUCCESS: Camera engine stopped!')
        return

    def set_exposure(self, ms):
        self.exposure = ms
        # Your code goes here!
        pass

    def set_gain(self, gain):
        self.gain = gain
        # Your code goes here!
        pass

    def arm(self):
        # Your code goes here!
        pass

    def get_frame(self):
        # Your code goes here!
        pass

    def disarm(self):
        # Your code goes here!
        pass


if __name__ == '__main__':
    mgr = CameraManager()
    mgr.add_camera("thorcam", serial_num="21723")
    mgr.add_camera("thorcam", serial_num="08949")
    for cam in mgr:
        cam.set(im_size=1024, exposure=0.001)
        cam.analyze_frame()
    mgr.close_all()

