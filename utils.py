from collections import namedtuple  # available in Python 2.6+
from dataclasses import dataclass
from pathlib import Path
import numpy as np

CONFIG_PATH = Path('config')
SIM_CONF = CONFIG_PATH / 'simulation.ini'
SCREEN_CONF = CONFIG_PATH / 'screen.ini'

simulation_params = namedtuple("simulation_params",
                               ["freq_temp", "freq_spacial", "amplitude", "phase_offset"])

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
    To allow the 'eval' to work with more than numbers, you have to define the text here
    for example:
    `pi = np.pi`
    defines the word 'pi' in the eval to be used as np.pi.
    """
    pi = np.pi
    if config_obj and config_obj.has_section(section):
        return param_class(*map(eval, config_obj[section].values()))
    else:
        params = []
        for field in param_class._fields:
            params.append(eval(dic[field]))
        return param_class(*params)


@dataclass
class Entry:
    """This is a data class to hold the entry data structure

    Attributes:
        measured (Path): Path to measured EEG .fif file.
        freesurfer (Path): Path to freesurfer folder.
        fwd_model (Path): Path to forward model file.
        stim (str): Type of stimulation, of {'TRAV_OUT', 'STANDING', 'TRAV_IN'}
        c_space (str): Type of spacial, of {'full', 'quad', 'fov'}
        simulation_config_section (str): Section in simulation configuration file. Default is 'None'.
        screen_config_section (str): Section in screen configuration file. Default is 'None'.
        simulation_params (Namedtuple): Simulation parameters namedtuple.
        screen_params (Namedtuple): Screen parameters namedtuple.
    """
    measured: Path = Path('/')
    freesurfer: Path = Path('/')
    fwd_model: Path = Path('/')
    stim: str = 'None'
    c_space: str = 'None'
    simulation_config_section: str = 'None'
    screen_config_section: str = 'None'
    simulation_params: simulation_params = simulation_params(*[0] * len(simulation_params._fields))
    screen_params: screen_params = screen_params(*[0] * len(screen_params._fields))

    def set_simulation_params(self, simulation_params_list):
        self.simulation_params = simulation_params(*simulation_params_list)

    def set_screen_params(self, screen_params_list):
        self.screen_params = screen_params(*screen_params_list)

    def create_dictionary(self):
        """Return a dictionary representation of the Entry object.

            Returns:
                dict: A dictionary containing the values of the Entry object's attributes.
        """
        entry_dict = {'measured': str(self.measured),
                      'freesurfer': str(self.freesurfer),
                      'fwd_model': str(self.fwd_model),
                      'stim': self.stim,
                      'c_space': self.c_space,
                      'simulation_config_section': self.simulation_config_section,
                      'screen_config_section': self.screen_config_section}
        entry_dict.update(self.simulation_params._asdict())
        entry_dict.update(self.screen_params._asdict())
        return entry_dict

    def load_entry(self, dic, sim_config_obj=None, screen_config_obj=None):
        """
            Load the entry data from a dictionary and sets the Entry class values accordingly.

            Args:
                dic (dict): A dictionary containing the entry values.
                sim_config_obj (simulation_params, optional): The simulation configuration settings. Defaults to None.
                screen_config_obj (screen_params, optional): The screen configuration settings. Defaults to None.

            Returns:
                self (Entry)
        """
        self.measured = Path(dic['measured'])
        self.freesurfer = Path(dic['freesurfer'])
        self.fwd_model = Path(dic['fwd_model'])
        self.stim = dic['stim']
        self.c_space = dic['c_space']
        self.simulation_config_section = dic['simulation_config_section']
        self.screen_config_section = dic['screen_config_section']
        self.simulation_params = load_param_from_config(dic, sim_config_obj,
                                                        self.simulation_config_section,
                                                        self.simulation_params.__class__)
        self.screen_params = load_param_from_config(dic, screen_config_obj,
                                                    self.screen_config_section,
                                                    self.screen_params.__class__)
        return self
