"""
Script to control the mightex camera
"""

from ctypes import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass
from abc import ABC, abstractmethod
import os

""" Config constants """
_STRING_MAX = 4096  # for functions that return strings built on C char arrays, this is the max number of characters

""" Callback ctypes types """
_camera_connect_callback_type = CFUNCTYPE(None, c_char_p, c_int, c_void_p)
_camera_disconnect_callback_type = CFUNCTYPE(None, c_char_p, c_void_p)
_frame_available_callback_type = CFUNCTYPE(None, c_void_p, POINTER(c_ushort), c_int, POINTER(c_char), c_int, c_void_p)
# metadata is ASCII, so use c_char
_3x3Matrix_float = (c_float * 9)


# def c_cmd(cmd, handle, ctype):
#     ret_val = ctype()
#
#
#
#
MIGHTEX_DEFAULTS = {
    'width': 2560,
    'height': 1920,
    'exposure': 0.05,
    'gain': 8,
    'metadata': 128,
    'dtype': 'i2',
    'pixel_size': 2.2
}
THORCAM_DEFAULTS = {
    'width': 3296,
    'height': 2472,
    'exposure': 0.05,
    'gain': 1,
    'metadata': 128,
    'dtype': 'i2',
    'pixel_size': 5.5,
    'serial_number': '09489'
}

libs = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/') + '/libs'


class Camera(ABC):
    @abstractmethod
    def __init__(self, defaults, verbose):
        self.defaults = defaults
        self.is_on = False
        self.verbose = verbose
        self.full_width = self.defaults['width']
        self.full_height = self.defaults['height']
        self.width = self.defaults['width']
        self.height = self.defaults['height']
        self.im_shape = (self.height, self.width)
        self.im_size = self.width * self.height
        self.dtype = self.defaults['dtype']
        self.image_metadata_size = self.defaults['metadata']
        self.data_size = self.im_size + self.image_metadata_size
        self.pixel_size = self.defaults['pixel_size']
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
        pass

    @abstractmethod
    def camera_off(self):
        pass

    @abstractmethod
    def set_resolution(self, resolution):
        return

    @abstractmethod
    def set_exposure(self, ms):
        pass

    @abstractmethod
    def set_gain(self, gain: int):
        pass

    @abstractmethod
    def get_frame(self):
        pass

    @staticmethod
    def imshow(image):
        plt.subplot(111, xticks=[], yticks=[])
        plt.imshow(image, cmap='plasma')
        plt.tight_layout()
        plt.show()

    def print(self, text):
        if self.verbose:
            print(text)

    def set_defaults(self):
        self.set_resolution((self.defaults['width'], self.defaults['height']))
        self.set_exposure(self.defaults['exposure'])
        self.set_gain(self.defaults['gain'])
        return

    def find_center(self):
        img = self.get_frame()
        # img = np.roll(img, (300, 300), (0, 1))
        img = img - 2 * np.median(img)
        img[img < 0] = 0
        return center_of_mass(img)

    def analyze_frame(self):
        img = self.get_frame()
        # img = np.roll(img, (300, 300), (0, 1))
        img = img - 2 * np.median(img)
        img[img < 0] = 0
        cy, cx = center_of_mass(img)
        if self.verbose:
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
            plt.plot(y, np.arange(self.height), c='r')
            plt.show()


class Mightex(Camera):
    _dll = CDLL(f'{libs}/Mightex/SSClassic_USBCamera_SDK.dll')

    def __init__(self, verbose=False):
        super().__init__(MIGHTEX_DEFAULTS, verbose)

        # Basic IO functions for connecting/disconnecting with the camera
        self._sdk_InitDevice = Mightex._dll.SSClassicUSB_InitDevice
        self._sdk_InitDevice.argtypes = []
        self._sdk_InitDevice.restype = c_int

        self._sdk_AddCamera = Mightex._dll.SSClassicUSB_AddDeviceToWorkingSet
        self._sdk_AddCamera.argtypes = [c_int]  # [ID]
        self._sdk_AddCamera.restype = c_int

        self._sdk_StartCameraEngine = Mightex._dll.SSClassicUSB_StartCameraEngine
        self._sdk_StartCameraEngine.argtypes = [c_void_p, c_int, c_int, c_int]  # [GUI, bitdepth, threads, callback]
        self._sdk_StartCameraEngine.restype = c_int

        self._sdk_StopCameraEngine = Mightex._dll.SSClassicUSB_StopCameraEngine
        self._sdk_StopCameraEngine.argtypes = []
        self._sdk_StopCameraEngine.restype = c_int

        self._sdk_UnInitDevice = Mightex._dll.SSClassicUSB_UnInitDevice
        self._sdk_UnInitDevice.argtypes = []
        self._sdk_UnInitDevice.restype = c_int

        # Methods for collecting frame data
        self._sdk_SetWorkMode = Mightex._dll.SSClassicUSB_SetCameraWorkMode
        self._sdk_SetWorkMode.argtypes = [c_int, c_int]  # [ID, mode=0] (0 for normal)
        self._sdk_SetWorkMode.restype = c_int

        self._sdk_StartFrameGrab = Mightex._dll.SSClassicUSB_StartFrameGrab
        self._sdk_StartFrameGrab.argtypes = [c_int, c_int]  # [ID, num_frames=0x8888] (0x8888 for continuous)
        self._sdk_StartFrameGrab.restype = c_int

        self._sdk_StopFrameGrab = Mightex._dll.SSClassicUSB_StopFrameGrab
        self._sdk_StopFrameGrab.argtypes = [c_int]  # [ID]
        self._sdk_StopFrameGrab.restype = c_int

        # Methods to define camera parameters
        self._sdk_SetResolution = Mightex._dll.SSClassicUSB_SetCustomizedResolution
        self._sdk_SetResolution.argtypes = [c_int, c_int, c_int, c_int, c_int]  # [ID, width, height, bin=0, binmode=0]
        self._sdk_SetResolution.restype = c_int

        self._sdk_SetXYStart = Mightex._dll.SSClassicUSB_SetXYStart
        self._sdk_SetXYStart.argtypes = [c_int, c_int, c_int]  # [ID, X-start, Y-start]
        self._sdk_SetXYStart.restype = c_int

        self._sdk_SetExposure = Mightex._dll.SSClassicUSB_SetExposureTime
        self._sdk_SetExposure.argtypes = [c_int, c_int]  # [ID, exposure_time (x0.05 ms)]
        self._sdk_SetExposure.restype = c_int

        self._sdk_SetGain = Mightex._dll.SSClassicUSB_SetGains
        self._sdk_SetGain.argtypes = [c_int, c_int, c_int, c_int]  # [ID, gain]
        self._sdk_SetGain.restype = c_int
        self.gain = 0  # gain level (gain factor * 8)

        # As I understand it (which is barely) the frame hooker is what grabs the frame data?
        self._sdk_InstallFrameHooker = Mightex._dll.SSClassicUSB_InstallFrameHooker
        self._sdk_InstallFrameHooker.argtypes = [c_int, c_void_p]  # [type=0, callback=None] (type=0 for raw data)
        self._sdk_InstallFrameHooker.restype = c_int

        self._sdk_GetCurrentFrame = Mightex._dll.SSClassicUSB_GetCurrentFrame16bit
        # [type,    ID,     pointer to data]
        self._sdk_GetCurrentFrame.argtypes = [c_int, c_int, np.ctypeslib.ndpointer(self.dtype, 1, (self.data_size,))]
        self._sdk_GetCurrentFrame.restype = np.ctypeslib.ndpointer(self.dtype, 1, (self.data_size,))

        self.camera_on()
        self.set_defaults()

        return

    def camera_on(self):
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

    def set_gain(self, gain: int):
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
            # Figure out how to prevent failure in the first place.
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


class ThorCam(Camera):
    _dll = CDLL(f'{libs}/ThorCam/thorlabs_tsi_camera_sdk.dll')
    _dll.tl_camera_discover_available_cameras.argtypes = [c_char_p, c_int]
    _dll.tl_camera_open_camera.argtypes = [c_char_p, POINTER(c_void_p)]
    _dll.tl_camera_set_camera_connect_callback.argtypes = [_camera_connect_callback_type, c_void_p]
    _dll.tl_camera_set_camera_disconnect_callback.argtypes = [_camera_disconnect_callback_type, c_void_p]
    _dll.tl_camera_close_camera.argtypes = [c_void_p]
    _dll.tl_camera_set_frame_available_callback.argtypes = [c_void_p, _frame_available_callback_type, c_void_p]
    _dll.tl_camera_get_pending_frame_or_null.argtypes = [c_void_p, POINTER(POINTER(c_ushort)), POINTER(c_int),
                                                         POINTER(POINTER(c_char)), POINTER(c_int)]
    _dll.tl_camera_get_measured_frame_rate.argtypes = [c_void_p, POINTER(c_double)]
    _dll.tl_camera_get_is_data_rate_supported.argtypes = [c_void_p, c_int, POINTER(c_bool)]
    _dll.tl_camera_get_is_taps_supported.argtypes = [c_void_p, POINTER(c_bool), c_int]
    _dll.tl_camera_get_color_correction_matrix.argtypes = [c_void_p, POINTER(_3x3Matrix_float)]
    _dll.tl_camera_get_default_white_balance_matrix.argtypes = [c_void_p, POINTER(_3x3Matrix_float)]
    _dll.tl_camera_arm.argtypes = [c_void_p, c_int]
    _dll.tl_camera_issue_software_trigger.argtypes = [c_void_p]
    _dll.tl_camera_disarm.argtypes = [c_void_p]
    _dll.tl_camera_get_exposure_time.argtypes = [c_void_p, POINTER(c_longlong)]
    _dll.tl_camera_set_exposure_time.argtypes = [c_void_p, c_longlong]
    _dll.tl_camera_get_image_poll_timeout.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_set_image_poll_timeout.argtypes = [c_void_p, c_int]
    _dll.tl_camera_get_exposure_time_range.argtypes = [c_void_p, POINTER(c_longlong), POINTER(c_longlong)]
    _dll.tl_camera_get_firmware_version.argtypes = [c_void_p, c_char_p, c_int]
    _dll.tl_camera_get_frame_time.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_get_trigger_polarity.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_set_trigger_polarity.argtypes = [c_void_p, c_int]
    _dll.tl_camera_get_binx.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_set_binx.argtypes = [c_void_p, c_int]
    _dll.tl_camera_get_sensor_readout_time.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_get_binx_range.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
    _dll.tl_camera_get_is_hot_pixel_correction_enabled.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_set_is_hot_pixel_correction_enabled.argtypes = [c_void_p, c_int]
    _dll.tl_camera_get_hot_pixel_correction_threshold.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_set_hot_pixel_correction_threshold.argtypes = [c_void_p, c_int]
    _dll.tl_camera_get_hot_pixel_correction_threshold_range.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
    _dll.tl_camera_get_sensor_width.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_get_gain_range.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
    _dll.tl_camera_get_image_width_range.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
    _dll.tl_camera_get_sensor_height.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_get_image_height_range.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
    _dll.tl_camera_get_model.argtypes = [c_void_p, c_char_p, c_int]
    _dll.tl_camera_get_name.argtypes = [c_void_p, c_char_p, c_int]
    _dll.tl_camera_set_name.argtypes = [c_void_p, c_char_p]
    _dll.tl_camera_get_name_string_length_range.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
    _dll.tl_camera_get_frames_per_trigger_zero_for_unlimited.argtypes = [c_void_p, POINTER(c_uint)]
    _dll.tl_camera_set_frames_per_trigger_zero_for_unlimited.argtypes = [c_void_p, c_uint]
    _dll.tl_camera_get_frames_per_trigger_range.argtypes = [c_void_p, POINTER(c_uint), POINTER(c_uint)]
    _dll.tl_camera_get_usb_port_type.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_get_communication_interface.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_get_operation_mode.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_set_operation_mode.argtypes = [c_void_p, c_int]
    _dll.tl_camera_get_is_armed.argtypes = [c_void_p, POINTER(c_bool)]
    _dll.tl_camera_get_is_eep_supported.argtypes = [c_void_p, POINTER(c_bool)]
    _dll.tl_camera_get_is_led_supported.argtypes = [c_void_p, POINTER(c_bool)]
    _dll.tl_camera_get_is_cooling_supported.argtypes = [c_void_p, POINTER(c_bool)]
    _dll.tl_camera_get_cooling_enable.argtypes = [c_void_p, POINTER(c_bool)]
    _dll.tl_camera_get_is_nir_boost_supported.argtypes = [c_void_p, POINTER(c_bool)]
    _dll.tl_camera_get_camera_sensor_type.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_get_color_filter_array_phase.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_get_camera_color_correction_matrix_output_color_space.argtypes = [c_void_p, c_char_p]
    _dll.tl_camera_get_data_rate.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_set_data_rate.argtypes = [c_void_p, c_int]
    _dll.tl_camera_get_sensor_pixel_size_bytes.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_get_sensor_pixel_width.argtypes = [c_void_p, POINTER(c_double)]
    _dll.tl_camera_get_sensor_pixel_height.argtypes = [c_void_p, POINTER(c_double)]
    _dll.tl_camera_get_bit_depth.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_get_roi.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int), POINTER(c_int),
                                       POINTER(c_int)]
    _dll.tl_camera_set_roi.argtypes = [c_void_p, c_int, c_int, c_int, c_int]
    _dll.tl_camera_get_roi_range.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int),
                                             POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)]
    _dll.tl_camera_get_serial_number.argtypes = [c_void_p, c_char_p, c_int]
    _dll.tl_camera_get_serial_number_string_length_range.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
    _dll.tl_camera_get_is_led_on.argtypes = [c_void_p, POINTER(c_bool)]
    _dll.tl_camera_set_is_led_on.argtypes = [c_void_p, c_bool]
    _dll.tl_camera_get_eep_status.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_set_is_eep_enabled.argtypes = [c_void_p, c_bool]
    _dll.tl_camera_get_biny.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_set_biny.argtypes = [c_void_p, c_int]
    _dll.tl_camera_get_biny_range.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
    _dll.tl_camera_get_gain.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_set_gain.argtypes = [c_void_p, c_int]
    _dll.tl_camera_get_black_level.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_set_black_level.argtypes = [c_void_p, c_int]
    _dll.tl_camera_get_black_level_range.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
    _dll.tl_camera_get_frames_per_trigger_zero_for_unlimited.argtypes = [c_void_p, POINTER(c_uint)]
    _dll.tl_camera_set_frames_per_trigger_zero_for_unlimited.argtypes = [c_void_p, c_uint]
    _dll.tl_camera_get_frames_per_trigger_range.argtypes = [c_void_p, POINTER(c_uint), POINTER(c_uint)]
    _dll.tl_camera_get_image_width.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_get_image_height.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_get_polar_phase.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_get_frame_rate_control_value_range.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double)]
    _dll.tl_camera_get_is_frame_rate_control_enabled.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_set_is_frame_rate_control_enabled.argtypes = [c_void_p, c_int]
    _dll.tl_camera_get_frame_rate_control_value.argtypes = [c_void_p, POINTER(c_double)]
    _dll.tl_camera_set_frame_rate_control_value.argtypes = [c_void_p, c_double]
    _dll.tl_camera_get_timestamp_clock_frequency.argtypes = [c_void_p, POINTER(c_int)]
    _dll.tl_camera_convert_gain_to_decibels.argtypes = [c_void_p, c_int, POINTER(c_double)]
    _dll.tl_camera_convert_decibels_to_gain.argtypes = [c_void_p, c_double, POINTER(c_int)]
    _dll.tl_camera_get_is_operation_mode_supported.argtypes = [c_void_p, c_int, POINTER(c_bool)]

    _dll.tl_camera_get_last_error.restype = c_char_p
    # noinspection PyProtectedMember
    _dll._internal_command.argtypes = [c_void_p, c_char_p, c_uint, c_char_p, c_uint]

    def __init__(self, verbose):
        super().__init__(THORCAM_DEFAULTS, verbose)

        ThorCam._dll.tl_camera_open_sdk()
        serial_number_bytes = self.defaults['serial_number'].encode("utf-8") + b'\0'
        c_camera_handle = c_void_p()  # void *
        ThorCam._dll.tl_camera_open_camera(serial_number_bytes, c_camera_handle)
        self._handle = c_camera_handle
        self.frames_per_trigger = 1

        self._grab_data = ThorCam._dll.tl_camera_set_frames_per_trigger_zero_for_unlimited

        self.camera_on()
        self.set_defaults()
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.camera_off()
        ThorCam._dll.tl_camera_close_sdk()

    def __del__(self):
        self.camera_off()
        ThorCam._dll.tl_camera_close_sdk()

    def camera_on(self):
        if self.is_on:
            self.print('Camera engine already started!')
            return

        ThorCam._dll.tl_camera_open_camera(self._handle)
        ThorCam._dll.tl_camera_set_frames_per_trigger_zero_for_unlimited(self.frames_per_trigger)

        self.is_on = True
        self.print('SUCCESS: Camera engine started!')
        return

    def camera_off(self):
        if not self.is_on:
            self.print('Camera engine not started!')
            return

        ThorCam._dll.tl_camera_close_camera(self._handle)

        self.is_on = False
        self.print('SUCCESS: Camera engine stopped!')
        return

    def set_resolution(self, resolution):
        pass

    def set_exposure(self, ms):
        us = ms * 1000
        ThorCam._dll.tl_camera_set_exposure_time(self._handle, us)
        print(ThorCam._dll.tl_camera_get_exposure_time(self._handle))
        return

    def set_gain(self, gain: int):
        ThorCam._dll.tl_camera_set_gain(self._handle, gain)
        return

    def set_frames_per_trigger(self, fpt):
        self.frames_per_trigger = fpt
        ThorCam._dll.tl_camera_set_frames_per_trigger_zero_for_unlimited(self.frames_per_trigger)
        return

    def get_frame(self, show=False):
        if not self.is_on:
            self.camera_on()
        data = np.zeros(self.im_shape, dtype=self.dtype)
        ThorCam._dll.tl_camera_arm(self._handle, self.frames_per_trigger)
        for i in range(self.frames_per_trigger):
            image_buffer = POINTER(c_ushort)()
            frame_count = c_int()
            meta_data = POINTER(c_char)()
            meta_size = c_int()
            error_code = self._grab_data(self._handle, image_buffer, frame_count, meta_data, meta_size)
            image_buffer._wrapper = self
            data = data + np.ctypeslib.as_array(image_buffer, self.im_shape)
        ThorCam._dll.tl_camera_disarm(self._handle)
        if show:
            self.imshow(data)
        return data


if __name__ == '__main__':
    cam = ThorCam(False)
