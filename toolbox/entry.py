from dataclasses import dataclass
from pathlib import Path
from typing import Union

from toolbox.utils import simulation_params as simp, screen_params as scp, load_param_from_config


@dataclass
class Entry:
    """This is a data class to hold the entry data structure

    Attributes
    ----------
    measured (Path):
        Path to measured EEG .fif file.
    freesurfer (Path): 
        Path to freesurfer folder.
    fwd_model (Path): 
        Path to forward model file.
    stim (str): {'TRAV_OUT', 'STANDING', 'TRAV_IN'}
        Type of traveling wave: traveling out (from fovea to periphery), standing or traveling in (from periphery to fovea)
    c_space (str): {'full', 'quad', 'fov'}
        Spatial extend for the simulation, can be either full-screen, quadrant or foveal simulation of brain activity.
    simulation_config_section (str): 
        Section in simulation configuration file. Default is 'None'.
    screen_config_section (str): 
        Section in screen configuration file. Default is 'None'.
    simulation_params (Namedtuple): [freq_temp, freq_spatial, amplitude, phase_offset, decay, e0]
        Simulation parameters specifying the traveling wave equations: temporal frequency (Hz), spatial frequency (cycles per mm of cortex), amplitude (A.m), phase offset (rad), decay parameter (between 0 and 1), and eccentricity for the foveal condition (DVA).
    screen_params (Namedtuple): [width, height, distancefrom, heightcm]
        Screen parameters specifying the width and height of the screen (in pixels) used for stimulus display, the distance of screen from participants'eyes, and the height of the screen in centimeters.
    """
    def __init__(self,
                 measured: Union[Path, str] = Path(),
                 freesurfer: Union[Path, str] = Path(),
                 forward_model: Union[Path, str] = Path(),
                 stim: str = 'None',
                 c_space: str = 'None',
                 simulation_config_section: str = 'None',
                 screen_config_section: str = 'None',
                 simulation_params: simp = simp(*[0] * len(simp._fields)),
                 screen_params: scp = scp(*[0] * len(scp._fields))
                 ):
        self._measured = measured
        self._freesurfer = freesurfer
        self._forward_model = forward_model
        self._stim = stim
        self._c_space = c_space
        self._simulation_config_section = simulation_config_section
        self._screen_config_section = screen_config_section
        self._simulation_params = simulation_params
        self._screen_params = screen_params

    def __repr__(self):
        attributes = []
        for attr, value in self.__dict__.items():
            attributes.append(f"{attr}={value!r}")
        return "Entry(\n  " + ",\n  ".join(attributes) + "\n)"   
    
    # Getter and setter for 'measured'
    @property
    def measured(self):
        return self._measured

    @measured.setter
    def measured(self, value):
        self._measured = Path(value)

    # Getter and setter for 'freesurfer'
    @property
    def freesurfer(self):
        return self._freesurfer

    @freesurfer.setter
    def freesurfer(self, value):
        self._freesurfer = Path(value)

    # Getter and setter for 'forward_model'
    @property
    def forward_model(self):
        return self._forward_model

    @forward_model.setter
    def forward_model(self, value):
        self._forward_model = Path(value)

    # Getter and setter for 'stim'
    @property
    def stim(self):
        return self._stim

    @stim.setter
    def stim(self, value):
        self._stim = value

    # Getter and setter for 'c_space'
    @property
    def c_space(self):
        return self._c_space

    @c_space.setter
    def c_space(self, value):
        self._c_space = value

    # Getter and setter for 'simulation_config_section'
    @property
    def simulation_config_section(self):
        return self._simulation_config_section

    @simulation_config_section.setter
    def simulation_config_section(self, value):
        self._simulation_config_section = value

    # Getter and setter for 'screen_config_section'
    @property
    def screen_config_section(self):
        return self._screen_config_section

    @screen_config_section.setter
    def screen_config_section(self, value):
        self._screen_config_section = value

    # Getter and setter for 'simulation_params'
    @property
    def simulation_params(self):
        return self._simulation_params

    @simulation_params.setter
    def simulation_params(self, value):
        """Passed value can be a sim_params, or a list to create sim_params object"""
        if type(value) is simp:
            self._simulation_params = value
        else:
            self._simulation_params = simp(*value)

    # Getter and setter for 'screen_params'
    @property
    def screen_params(self):
        return self._screen_params

    @screen_params.setter
    def screen_params(self, value):
        """Passed value can be a screen_params, or a list to create screen_params object"""
        if type(value) is scp:
            self._screen_params = value
        else:
            self._screen_params = scp(*value)

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
        self.measured = dic['measured']
        self.freesurfer = dic['freesurfer']
        self.forward_model = dic['forward_model']
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
