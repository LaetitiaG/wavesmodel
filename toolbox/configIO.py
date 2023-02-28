import pathlib
from configparser import ConfigParser
import toolbox.utils as utils
import os.path as op
from os import makedirs


def create_config_section(config_obj, params, name, path):
    config_obj[name] = params._asdict()
    write_config(config_obj, path)


def write_config(config_object, path: pathlib.Path):
    """

    :param config_object:
    :param path:
    """
    if not op.exists(path.parent):
        makedirs(path.parent)

    if not path.match("*.ini"):
        path = path.with_suffix(".ini")

    with open(path, 'w') as config:
        config_object.write(config)


def __get_config_object(filepath):
    config_object = ConfigParser()
    config_object.read(filepath)
    return config_object


def load_config(filepath):
    if not op.exists(filepath):
        return ConfigParser()
    if filepath.is_dir():
        raise ValueError("Invalid config file: file is a directory.")
    return __get_config_object(filepath)


def read_entry_config(filepath):
    entry_list = []
    config_obj = __get_config_object(filepath)
    sim_obj = __get_config_object(utils.SIM_CONF)
    screen_obj = __get_config_object(utils.SCREEN_CONF)
    for section in config_obj.sections():
        entry = utils.Entry()
        entry.load_entry(config_obj[section], sim_obj, screen_obj)
        entry_list.append(entry)
    return entry_list
