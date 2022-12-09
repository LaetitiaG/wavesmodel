import configparser
from configparser import ConfigParser
from utils import simulation_params, screen_params
from dataclasses import dataclass
import os.path as op
from os import makedirs


@dataclass
class Config:
    simulation_params: simulation_params
    screen_params: screen_params


def create_config_file(config_obj, params, name, path):
    config_obj[name] = params._asdict()
    write_config(config_obj, path)


def write_config(config_object, path):
    if not op.exists(path.parent):
        makedirs(path.parent)

    if not path.match("*.ini"):
        path = path.with_suffix(".ini")

    with open(path, 'w') as config:        config_object.write(config)


def read_config(filepath):
    config_object = ConfigParser()
    config_object.read(filepath)
    simulation = config_object["SIMULATION"]
    sim_params = simulation_params(*simulation.values())
    return sim_params


def get_config_object(filepath):
    config_object = ConfigParser()
    try:
        config_object.read(filepath)
    except configparser.DuplicateSectionError:
        print("ERROR: config file is invalid")
    return config_object


def load_config(filepath):
    if op.exists(filepath):
        return get_config_object(filepath)


# params = simulation_params(5, 0.05, 10e-9, 2)
# screen_params = screen_params(1920, 1080, 78, 44.2)
# conf = Config(params, screen_params)
# config_obj = create_config_file(conf)
# write_config(config_obj)
