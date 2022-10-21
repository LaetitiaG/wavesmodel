from collections import namedtuple  # available in Python 2.6+
from dataclasses import dataclass
from pathlib import Path

simulation_params = namedtuple("simulation_params",
                               ["freq_temp", "freq_spacial", "amplitude", "phase_offset"])

# mri_paths = namedtuple("mri_paths",
#                        ["varea_l", "varea_r", "angle_l", "angle_r", "eccen_l", "eccen_r"])

# Pass tuple to each mri type. Must be in the same hemisphere order: (left, right)
mri_paths = namedtuple("mri_paths", ["varea", "angle", "eccen"])

screen_params = namedtuple("screen_params",
                           ["width",  # pixel
                            "height",  # pixel
                            "distanceFrom",  # cm
                            "heightCM"  # cm
                            ])


@dataclass
class Entry:
    """Data class corresponding to the entry structure"""
    measured: Path = Path('/')
    retino_map: Path = Path('/')
    simulation_params: simulation_params = simulation_params(*[0] * len(simulation_params._fields))
    screen_params: screen_params = screen_params(*[0] * len(screen_params._fields))
    mri_params: mri_paths = None

    def set_simulation_params(self, simulation_params_list):
        self.simulation_params = simulation_params(*simulation_params_list)

    def set_screen_params(self, screen_params_list):
        self.screen_params = screen_params(*screen_params_list)

    def set_mri_params(self, mri_params_list):
        self.mri_params = mri_paths(*mri_params_list)


