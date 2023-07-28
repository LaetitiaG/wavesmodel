from collections import namedtuple  # available in Python 2.6+
from pathlib import Path
import numpy as np

CONFIG_PATH = Path('../config') if 'toolbox' in str(Path('./').absolute()) else Path('config')
SIM_CONF = CONFIG_PATH / 'simulation.ini'
SCREEN_CONF = CONFIG_PATH / 'screen.ini'

simulation_params = namedtuple("simulation_params",
                               ["freq_temp", "freq_spatial", "amplitude", "phase_offset", "decay", "e0"])

# Pass tuple to each mri type. Must be in the same hemisphere order: (left, right)
mri_paths = namedtuple("mri_paths", ["varea", "angle", "eccen"])

screen_params = namedtuple("screen_params",
                           ["width",  # pixel
                            "height",  # pixel
                            "distanceFrom",  # cm
                            "heightCM"  # cm
                            ])


def load_param_from_config(dic, config_obj, section, param_class):
    """
    Simulation parameters used in the traveling/standing waves equations:
        freq_temp: temporal frequency
        freq_spatial: spatial frequency
        amplitude: amplitude 
        phase_offset: initial phase
        decay: spatial decay of the amplitude of the wave
        e0: eccentricity (in dva) from which starts the spatial decay
    """

    if config_obj and config_obj.has_section(section):
        return param_class(*map(eval, config_obj[section].values()))
    else:
        params = []
        for field in param_class._fields:
            params.append(eval(dic[field]))
        return param_class(*params)
