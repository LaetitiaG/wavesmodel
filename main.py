import argparse
import sys
from pathlib import Path

import toolbox.configIO
from GUI import input
from toolbox.simulation import generate_simulation
from toolbox.projection import project_wave
from toolbox.comparison import compare_meas_simu


stim_list = ['TRAV_OUT', 'STANDING', 'TRAV_IN']
c_space_list = ['full', 'quad', 'fov']


def __get_path(string):
    try:
        return Path(string)
    except Exception as e:
        raise argparse.ArgumentTypeError(f'Error: {e}')


def __path(string):
    p = __get_path(string)
    if not p.exists():
        raise argparse.ArgumentTypeError('Provided path does not exist, please give a valid path.')
    return p


def __path_or_list(string):
    p = __get_path(string)
    if p.exists():
        return p
    try:
        return list(map(int, string.split(',')))
    except ValueError:
        # If the string cannot be parsed as a list of integers, raise an error
        raise argparse.ArgumentTypeError('Must be a path or a list of integers separated by commas.')


def parse_cli(argv):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--gui', action='store_true', help='run the GUI mode of the toolbox')
    group.add_argument('--entry-config', type=__path,
                        help='the path of an entry config file containing entries with all values')
    parser.add_argument('--sim-config', required=False, type=__path_or_list,
                        help='the path of the simulation config file to be used to load simulation parameters, '
                             'or a list of integers separated by commas')
    parser.add_argument('--screen-config', required=False, type=__path_or_list,
                        help='the path of the screen config file to be used to load screen parameters, or a list of '
                             'integers separated by commas')
    # parser.add_argument('--sensor-file', required=False, type=__path,
    #                     help='the path of the measured data file')
    # parser.add_argument('--mri-path', required=False, type=__path,
    #                     help='the path where to find freesurfer output, and the forward model following the'
    #                          'architecture of the documentation')
    # parser.add_argument('--stim', required=False, default='TRAV_OUT', choices=stim_list,
    #                     help='a string corresponding to the type of simulation')

    args = parser.parse_args(argv)

    return {
        "gui": args.gui,
        "entry_config_path": args.entry_config,
        "sim_config_path": args.sim_config,
        "screen_config_path": args.screen_config,
        # "sensor_file_path": args.sensor_file,
        # "mri_path": args.mri_path,
        # "stim": args.stim
    }


def run_pipeline(entry_list):
    for entry in entry_list:
        stc = generate_simulation(entry)
        proj = project_wave(entry, stc)
        compare = compare_meas_simu(entry, proj)
        phases, ampls, times, info, zscores, R2_all, pval_all, matrices = compare
        print(compare)


def run_main(argv):
    args = parse_cli(argv)
    if args.get('gui'):
        return input.run_gui()
    entry_file = args.get('entry_config_path')
    if entry_file is None:
        raise ValueError('You must provide an entry file')
    entry_list = toolbox.configIO.read_entry_config(entry_file)
    run_pipeline(entry_list)
    print(args)


if __name__ == "__main__":
    run_main(sys.argv[1:])
