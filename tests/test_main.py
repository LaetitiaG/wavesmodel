import unittest
from argparse import ArgumentTypeError
from main import parse_cli


class TestParseCLI(unittest.TestCase):
    def test_no_arguments(self):
        # Test that an error is raised if no arguments are provided
        with self.assertRaises(ArgumentTypeError):
            parse_cli([])

    def test_invalid_entry_config(self):
        # Test that an error is raised if the --entry-config argument is invalid
        with self.assertRaises(ArgumentTypeError):
            parse_cli(['--entry-config', '/invalid/path'])

    def test_valid_entry_config(self):
        # Test that the correct dictionary is returned if the --entry-config argument is valid
        expected_result = {
            "entry_config_path": '/valid/path',
            "sensor_file_path": '/another/valid/path',
            "mri_path": '/yet/another/valid/path',
            "stim": 'TRAV_OUT',
            "sim_config_path": '/one/more/valid/path',
            "screen_config_path": '/last/valid/path',
            "gui": False
        }
        self.assertEqual(parse_cli(['--entry-config', '/valid/path', '--sensor-file', '/another/valid/path',
                                    '--mri-path', '/yet/another/valid/path', '--sim-config', '/one/more/valid/path',
                                    '--screen-config', '/last/valid/path']), expected_result)
