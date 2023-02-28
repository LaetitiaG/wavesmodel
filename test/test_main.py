import unittest
from argparse import ArgumentTypeError
from toolbox.main import parse_cli
from pathlib import Path
import subprocess


class TestParseCLI(unittest.TestCase):
    def test_no_arguments(self):
        # Test that an error is raised if no arguments are provided
        with self.assertRaises(ArgumentTypeError):
            parse_cli([])

    def test_invalid_entry_config(self):
        # Test that an error is raised if the --entry-config argument is invalid
        process = subprocess.run(['python', '../main.py', '--entry-config', '/invalid/path'],
                                 stderr=subprocess.PIPE, universal_newlines=True)
        self.assertTrue(process.stderr)

    def test_valid_entry_config(self):
        # Test that the correct dictionary is returned if the --entry-config argument is valid
        expected_result = {
            "entry_config_path": Path('./entry/entry.ini')
            # "sensor_file_path": '/another/valid/path',
            # "mri_path": '/yet/another/valid/path',
            # "stim": 'TRAV_OUT',
            # "sim_config_path": '/one/more/valid/path',
            # "screen_config_path": '/last/valid/path',
            # "gui": False
        }
        self.assertEqual(parse_cli(['--entry-config', './entry/entry.ini']), expected_result)
        # '--sensor-file', '/another/valid/path',
        # '--mri-path', '/yet/another/valid/path', '--sim-config', '/one/more/valid/path',
        # '--screen-config', '/last/valid/path'])


if __name__ == '__main__':
    unittest.main()
