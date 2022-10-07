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
    params: simulation_params = None

    def set_params(self, param_list):
        self.params = simulation_params(*param_list)


