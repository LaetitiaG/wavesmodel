from collections import namedtuple  # available in Python 2.6+
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path('../config')
SIM_CONF = CONFIG_PATH / 'simulation.ini'
SCREEN_CONF = CONFIG_PATH / 'screen.ini'

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
    simulation_config_section: str = 'None'
    screen_config_section: str = 'None'
    simulation_params: simulation_params = simulation_params(*[0] * len(simulation_params._fields))
    screen_params: screen_params = screen_params(*[0] * len(screen_params._fields))
    mri_params: mri_paths = None
    stim: str = 'None'
    c_space: str = 'None'

    def set_simulation_params(self, simulation_params_list):
        self.simulation_params = simulation_params(*simulation_params_list)

    def set_screen_params(self, screen_params_list):
        self.screen_params = screen_params(*screen_params_list)

    def set_mri_params(self, mri_params_list):
        self.mri_params = mri_paths(*mri_params_list)

    def create_dictionary(self):
        entry_dict = {'measured': str(self.measured),
                      'retino_map': str(self.retino_map),
                      'simulation_config_name': self.simulation_config_section,
                      'screen_config_name': self.screen_config_section}
        entry_dict.update(self.simulation_params._asdict())
        entry_dict.update(self.screen_params._asdict())
        return entry_dict

    def __load_param_from_config(self, dic, config_obj, section, param_class):
        if config_obj and config_obj.has_section(section):
            return param_class(*config_obj[section].values())
        else:
            params = []
            for field in param_class._fields:
                params.append(dic[field])
            return param_class(*params)

    def load_entry(self, dic, sim_config_obj=None, screen_config_obj=None):
        self.measured = dic['measured']
        self.retino_map = dic['retino_map']
        self.simulation_config_section = dic['simulation_config_section']
        self.screen_config_section = dic['screen_config_section']
        self.simulation_params = self.__load_param_from_config(dic, sim_config_obj,
                                                               self.simulation_config_section,
                                                               self.simulation_params.__class__)
        self.screen_params = self.__load_param_from_config(dic, screen_config_obj,
                                                           self.screen_config_section,
                                                           self.screen_params.__class__)
