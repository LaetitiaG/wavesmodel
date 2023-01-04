import argparse


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry-config', required=True,
                        help='the path of an entry config file containing entries with all values')
    parser.add_argument('--sensor-file', required=False, help='the path of the measured data file')
    parser.add_argument('--mri-path', required=False,
                        help='the path where to find freesurfer output, and the forward model following the'
                             'architecture of the documentation')
    parser.add_argument('--stim', required=False, default='TRAV_OUT',
                        help='a string corresponding to the type of simulation')
    parser.add_argument('--sim-config', required=False,
                        help='the path of the simulation config file to be used to load simulation parameters, '
                             'or a list of values corresponding to the simulation parameters')
    parser.add_argument('--screen-config', required=False,
                        help='the path of the screen config file to be used to load screen parameters, or a list of '
                             'values corresponding to the screen parameters')
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
