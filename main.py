import argparse
import sys
from pathlib import Path
from GUI import input
from toolbox.simulation import generate_simulation
from toolbox.projection import project_wave


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
    parser.add_argument('--gui', action='store_true', help='run the GUI mode of the toolbox')
    parser.add_argument('--entry-config', required=False, type=__path,
                        help='the path of an entry config file containing entries with all values')
    parser.add_argument('--sensor-file', required=False, type=__path,
                        help='the path of the measured data file')
    parser.add_argument('--mri-path', required=False, type=__path,
                        help='the path where to find freesurfer output, and the forward model following the'
                             'architecture of the documentation')
    parser.add_argument('--stim', required=False, default='TRAV_OUT', choices=stim_list,
                        help='a string corresponding to the type of simulation')
    parser.add_argument('--sim-config', required=False, type=__path_or_list,
                        help='the path of the simulation config file to be used to load simulation parameters, '
                             'or a list of integers separated by commas')
    parser.add_argument('--screen-config', required=False, type=__path_or_list,
                        help='the path of the screen config file to be used to load screen parameters, or a list of '
                             'integers separated by commas')

    args = parser.parse_args(argv)

    return {
        "entry_config_path": args.entry_config,
        "sensor_file_path": args.sensor_file,
        "mri_path": args.mri_path,
        "stim": args.stim,
        "sim_config_path": args.sim_config,
        "screen_config_path": args.screen_config,
        "gui": args.gui
    }


def run_pipeline(entry_list):
    for entry in entry_list:
        stc_gen = generate_simulation(entry)
        fwd = None
        project_wave(entry, fwd, stc_gen)


def run(argv):
    print(argv)
    args = parse_cli(argv)
    if args.get('gui'):
        return input.run_gui()
    # if args.get('entry_config_path') is None:
    #     raise argparse.ArgumentTypeError('You must provide an entry file')
    print(args)


if __name__ == "__main__":
    run(sys.argv[1:])
