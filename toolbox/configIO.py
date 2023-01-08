import configparser
from configparser import ConfigParser
import utils
import os.path as op
from os import makedirs


def create_config_section(config_obj, params, name, path):
    config_obj[name] = params._asdict()
    write_config(config_obj, path)


def write_config(config_object, path):
    if not op.exists(path.parent):
        makedirs(path.parent)

    if not path.match("*.ini"):
        path = path.with_suffix(".ini")

    with open(path, 'w') as config:
        config_object.write(config)


def get_config_object(filepath):
    # needs better handling of errors
    config_object = ConfigParser()
    try:
        config_object.read(filepath)
    except:
        print("ERROR: config file is invalid")
    return config_object


def load_config(filepath):
    if op.exists(filepath):
        return get_config_object(filepath)


def read_entry_config(filepath):
    entry_list = []
    config_obj = get_config_object(filepath)
    sim_obj = get_config_object(utils.SIM_CONF)
    screen_obj = get_config_object(utils.SCREEN_CONF)
    for section in config_obj.sections():
        entry = utils.Entry()
        entry.load_entry(config_obj[section], sim_obj, screen_obj)
        entry_list.append(entry)
    return entry_list

# params = simulation_params(5, 0.05, 10e-9, 2)
# screen_params = screen_params(1920, 1080, 78, 44.2)
# conf = Config(params, screen_params)
# config_obj = create_config_file(conf)
# write_config(config_obj)
