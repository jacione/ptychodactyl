import time
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
from threading import Thread

import h5py
import numpy as np
import tifffile
import tqdm
import yaml

from ptycho.camera import CameraManager
from ptycho.scan import xy_scan
from ptycho.stages import get_stages
from ptycho.utils import remove_recursive


class Experiment:
    """
    PtychoData subclass for collecting, organizing, and saving ptycho data during an experiment
    """

    def __init__(self, config):
        # Generate the scanning positions
        self._config = config
        self._X, self._Y, self._num_translations = xy_scan(config)
        self._Z = config["z_position"]
        self._Q, self._num_rotations = [0], 1
        self._num_entries = self._num_translations * self._num_rotations  # Total number of translations and rotations
        self._is3d = self._num_rotations > 1
        self.verbose = config["verbose"]
        self.show_images = config["show_images"]
        self.rm_temp = config["rm_temp"]

        # Initialize the devices and data structures
        self.stages = get_stages(config)
        self.camera_mgr = CameraManager()
        self._num_detectors = 0
        while True:
            try:
                settings = config[f"camera_{self._num_detectors}"]
            except KeyError:
                break
            camera = self.camera_mgr.add_camera(key=settings["key"], serial_num=settings["serial_num"])
            camera.set(accums=settings["accums"], exposure=settings["exposure"], im_size=settings["im_size"])
            self._num_detectors += 1

        # Indexing
        self._n = 0  # Current measurement index
        self._i = -1  # Current rotation index
        self._j = 0  # Current translation index

        # These parameters shouldn't change
        self._energy = config["energy"]  # photon energy in eV
        self._wavelength = 1.240 / config["energy"]  # wavelength in microns
        self._distance = config["distance"] * 1e6  # sample-to-detector distance in microns

        # Parameters for saving the data afterward
        self._start_time = None
        self._end_time = None
        self.title = config["title"] + '-' + date.today().isoformat()
        if config["data_dir"] == '':
            self._dir = Path(__file__).parents[1] / 'data' / config["title"]
        else:
            self._dir = Path(config["data_dir"]) / config["title"]
        self._dir.mkdir(exist_ok=True)
        (self._dir / "positions").mkdir(exist_ok=True)
        self.pos_file = None
        for d in range(self._num_detectors):
            (self._dir / f"det_{d}").mkdir(exist_ok=True)
        return

    def run(self):
        print(f'Run type: {2 + self._is3d}D ptycho')
        print(f'Scan area:')
        print(f'\tPattern:   {self._config["scan_pattern"].upper()}')
        print(f'\tWidth:     {self._config["scan_width"]:0.4} mm')
        print(f'\tHeight:    {self._config["scan_height"]:0.4} mm')
        print(f'\tStep:      {self._config["scan_step"]:0.4} mm')
        if self._is3d:
            print(f'\tRotations: {self._num_rotations}')
        print(f'Total: {self._num_entries} measurements\n')

        if self._config["background"]:
            input('Preparing to take background images. Turn laser OFF, then press ENTER to continue...')
            self.record_background()

        input('Preparing to take ptycho data. Turn laser ON, then press ENTER to continue...')
        self._start_time = time.perf_counter()
        z = self._Z
        for i, q in enumerate(self._Q):
            print(f'Rotation {i}: {q:.3f} deg')
            self.new_rotation()
            for j, (x, y) in enumerate(zip(self._X, self._Y)):
                # self.print(f"    Position {j}: {x:.3f}, {y:.3f}")
                print()
                self.stages.set_position((x, y, z, q))
                self.record_data()
                # The CollectData.record_data() method should return True only after taking num_translations images.
                # This ensures that the data from each rotation angle stays together.

        self._end_time = time.perf_counter()
        print(f"Total time: {timedelta(seconds=self._end_time-self._start_time)}")

        self.stages.home_all()
        self.camera_mgr.close_all()

        self.save_to_pty()
        if self.rm_temp:
            self.remove_temp_files()

    def print(self, text=""):
        """A wrapper for print() that checks whether the instance is set to verbose"""
        if self.verbose:
            print(text)

    def new_rotation(self):
        self._i += 1
        self._j = 0
        self.pos_file = self._dir / f"positions/rot_{self._i:03}.npy"
        self.pos_file.unlink(missing_ok=True)
        for d in range(self._num_detectors):
            try:
                (self._dir / f"det_{d}/rot_{self._i:03}").mkdir()
            except FileExistsError:
                for f in (self._dir / f"det_{d}/rot_{self._i:03}").iterdir():
                    f.unlink()

    def record_data(self):
        threads = [Thread(target=cam.add_frame_to_queue) for cam in self.camera_mgr]
        threads.append(Thread(target=self.stages.add_position_to_queue, args=(True,)))
        [t.start() for t in threads]
        [t.join() for t in threads]
        stage_data = self.stages.get_position_from_queue()
        im_data = [cam.get_frame_from_queue() for cam in self.camera_mgr]

        # Record the image data
        if len(im_data) != self._num_detectors:
            raise IndexError(f"Tried to save data from {len(im_data)} detectors (expected {self._num_detectors})")
        for d, img in enumerate(im_data):
            tifffile.imwrite(self._dir / f"det_{d}/rot_{self._i:03}/img_{self._j:04}.tiff", img)

        # Record the position data
        try:
            pos_array = np.load(self.pos_file)  # Record in millimeters / degrees
            np.save(self.pos_file, np.vstack((pos_array, stage_data)))
        except FileNotFoundError:
            np.save(self.pos_file, stage_data)

        self.print(f'Take {self._n} of {self._num_entries} recorded!')
        self._n += 1
        self._j += 1
        return

    def record_background(self):
        """
        Record a background image without any associated positional data. This should be called when the beam is turned
        off or otherwise blocked.
        """
        for d, img in enumerate(self.camera_mgr.get_frames()):
            tifffile.imwrite(self._dir / f"det_{d}/background.tiff", img)

    def save_to_pty(self):
        """
        Save collected data as a PTY (HDF5) file. Data is saved using gzip lossless compression, which reduces file
        size by about a factor of 4.
        """

        print(f'Compressing and saving data...')

        # Create file using HDF5 protocol.
        filepath = self._dir / f'{self.title}.pty'
        f = h5py.File(filepath, 'w')

        # Save collected data
        t0 = perf_counter()
        data = f.create_group('data')

        all_positions = np.array([
            np.load(f"{self._dir / f'positions/rot_{i:03}.npy'}") for i in range(self._num_rotations)
        ])
        # Translate all the positions so that the minimum is zero.
        offset = np.min(all_positions, axis=1)

        for i in range(self._num_rotations):
            rot = data.create_group(f"rot_{i}")
            rot.attrs.create("nominal_angle", self._Q[i])
            rot.create_dataset('position', data=np.load(f"{self._dir / f'positions/rot_{i:03}.npy'}") - offset,
                               compression='gzip')
            for d in range(self._num_detectors):
                im_files = [str(self._dir/f"det_{d}/rot_{i:03}/img_{j:04}.tiff") for j in range(self._num_translations)]
                rot.create_dataset(f"det_{d}", data=tifffile.imread(im_files, ioworkers=4, maxworkers=4),
                                   compression="gzip")

        # Save experimental parameters
        equip = f.create_group('equipment')
        source = equip.create_group('source')
        source.attrs.create('energy', data=self._energy)
        source.attrs.create('wavelength', data=self._wavelength)
        for d, cam in enumerate(self.camera_mgr):
            det = equip.create_group(f'det_{d}')
            specs = cam.get_specs() | cam.settings | {"distance": self._config[f"camera_{d}"]["distance"]}
            for key, val in specs.items():
                det.attrs.create(key, val)
            try:
                det.create_dataset("background", data=tifffile.imread(self._dir / f"det_{d}/background.tiff"))
            except FileNotFoundError:
                det.create_dataset("background", data=np.zeros((specs["im_size"], specs["im_size"])))

        t1 = perf_counter()
        f.close()

        print(f'Data saved as {self.title}.pty')
        print(f'Save time: {np.round(t1 - t0, 4)} seconds')
        print(f'Data size: {np.round(filepath.stat().st_size / 1024 / 1024, 3)} MB')
        return

    def remove_temp_files(self):
        remove_recursive(self._dir/"positions")
        for d, _ in enumerate(self.camera_mgr):
            remove_recursive(self._dir/f"det_{d}")


class Simulation(Experiment):
    pass


if __name__ == "__main__":
    conf_file = Path(__file__).parents[1] / "config_example/collection_config.yml"
    conf = yaml.safe_load(conf_file.read_text())
    exp = Experiment(conf)
    exp.run()
