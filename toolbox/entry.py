from dataclasses import dataclass
from pathlib import Path

from toolbox.utils import simulation_params, screen_params, load_param_from_config


@dataclass
class Entry:
    """This is a data class to hold the entry data structure

    Attributes
    ----------
    measured (Path):
        Path to measured EEG .fif file.
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
    forward_model: Path = Path('/')
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
                      'forward_model': str(self.forward_model),
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
        self.forward_model = Path(dic['forward_model'])
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
