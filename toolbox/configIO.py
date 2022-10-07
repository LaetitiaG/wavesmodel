from configparser import ConfigParser
from utils import simulation_params
from dataclasses import dataclass


@dataclass
class Config:
    simulation_params: simulation_params


def create_config_file(config):
    sim = config.simulation_params
    config_obj = ConfigParser()

    config_obj["SIMULATION"] = sim._asdict()
    return config_obj


def write_config(config_object):
    with open('config.ini', 'w') as conf:
        config_object.write(conf)


params = simulation_params(5, 0.05, 10e-9, 2)
config = Config(params)
config_object = create_config_file(config)
# write_config(config_object)
