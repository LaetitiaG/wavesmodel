import argparse
from pathlib import Path


def __check_stim(string):
    if string not in ['TRAV_OUT', 'STANDING', 'TRAV_IN']:
        raise argparse.ArgumentTypeError('Must be one of TRAV_OUT, STANDING, TRAV_IN')
    return string


def __path(string):
    try:
        return Path(string)
    except Exception as e:
        raise argparse.ArgumentTypeError(f'Error: {e}')


def __path_or_list(string):
    p = __path(string)
    if p.exists():
        return p
    try:
        return list(map(int, string.split(',')))
    except ValueError:
        # If the string cannot be parsed as a list of integers, raise an error
        raise argparse.ArgumentTypeError('Must be a path or a list of integers separated by commas')


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry-config', required=True, type=__path,
                        help='the path of an entry config file containing entries with all values')
    parser.add_argument('--sensor-file', required=False, type=__path,
                        help='the path of the measured data file')
    parser.add_argument('--mri-path', required=False, type=__path,
                        help='the path where to find freesurfer output, and the forward model following the'
                             'architecture of the documentation')
    parser.add_argument('--stim', required=False, default='TRAV_OUT', type=__check_stim,
                        help='a string corresponding to the type of simulation')
    parser.add_argument('--sim-config', required=False, type=__path_or_list,
                        help='the path of the simulation config file to be used to load simulation parameters, '
                             'or a list of integers separated by commas')
    parser.add_argument('--screen-config', required=False, type=__path_or_list,
                        help='the path of the screen config file to be used to load screen parameters, or a list of '
                             'integers separated by commas')
    parser.add_argument('--gui', action='store_true', help='run the GUI mode of the toolbox')

    args = parser.parse_args()

    entry_config_path = args.entry_config
    sensor_file_path = args.sensor_file
    mri_path = args.mri_path
    stim = args.stim
    sim_config_path = args.sim_config
    screen_config_path = args.screen_config
    gui = args.gui

    print(entry_config_path)
    print(stim)


if __name__ == "__main__":
    parse_cli()
