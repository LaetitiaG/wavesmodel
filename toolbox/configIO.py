from configparser import ConfigParser
from tools import simulation_params, screen_params
from dataclasses import dataclass


@dataclass
class Config:
    simulation_params: simulation_params
    screen_params: screen_params


def create_config_file(config):
    sim = config.simulation_params
    screen = config.screen_params
    config_obj = ConfigParser()

    config_obj["SIMULATION"] = sim._asdict()
    config_obj["SCREEN"] = screen._asdict()
    return config_obj


def write_config(config_object):
    with open('config.ini', 'w') as config:
        config_object.write(config)


def read_config(filepath):
    config_object = ConfigParser()
    config_object.read(filepath)
    simulation = config_object["SIMULATION"]
    sim_params = simulation_params(*simulation.values())
    return sim_params

# params = simulation_params(5, 0.05, 10e-9, 2)
# screen_params = screen_params(1920, 1080, 78, 44.2)
# conf = Config(params, screen_params)
# config_obj = create_config_file(conf)
# write_config(config_obj)
