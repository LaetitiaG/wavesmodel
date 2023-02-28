import configparser
from pathlib import Path
import unittest
import toolbox.configIO as configIO
from toolbox.configIO import read_entry_config
from toolbox import utils
import os


class TestWriteConfig(unittest.TestCase):
    cfg_obj = configparser.ConfigParser()
    path = Path("./config/test_file")

    def test_write_valid_path_without_suffix(self):
        cfg_obj = self.cfg_obj
        path = self.path
        path_suffix = path.with_suffix('.ini')
        configIO.write_config(cfg_obj, path)
        self.assertTrue(os.path.exists(path_suffix))
        os.remove(path_suffix)

        os.rmdir(path.parent)

    def test_write_valid_path_with_suffix(self):
        cfg_obj = self.cfg_obj
        path = self.path.with_suffix('.ini')

        configIO.write_config(cfg_obj, path)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

        os.rmdir(path.parent)


class TestLoadConfig(unittest.TestCase):

    def test_load_valid_file(self):
        path = Path('./entry/entry.ini')
        cfg_obj = configIO.load_config(path)
        self.assertIsNotNone(cfg_obj)
        self.assertTrue(cfg_obj.sections())

    def test_load_invalid_file(self):
        path = Path('notafile.ini')
        with self.assertRaises(ValueError):
            configIO.load_config(path)

    def test_load_invalid_folder(self):
        path = Path('entry/')
        with self.assertRaises(ValueError):
            configIO.load_config(path)


class TestReadEntryConfig(unittest.TestCase):

    def setUp(self):
        self.config_file = 'test_config_file.ini'

        # create a test config file
        config = configparser.ConfigParser()
        config['entry1'] = {
            "measured": "C:\\Users\\laeti/Data/wave_model/data_MEEG/sensors\\2XXX72_fullField_stand - ave.fif",
            "freesurfer": "C:/Users/laeti/Data/wave_model/data_MRI/preproc\\freesurfer\\2XXX72",
            "fwd_model": "C:/Users/laeti/Data/wave_model/data_MEEG/preproc\\2XXX72\\forwardmodel\\2XXX72_session1_ico5 - fwd.fif",
            "stim": "STANDING",
            "c_space": "full",
            "simulation_config_section": "0cfg",
            "screen_config_section": "cfg1",
            "freq_temp": 5,
            "freq_spacial": 0.05,
            "amplitude": 10e-9,
            "phase_offset": "np.pi / 2",
            "width": 1920,
            "height": 1080,
            "distancefrom": 78,
            "heightcm": 44.2
        }
        config['entry2'] = {
            "measured": "false/path/file.fif",
            "freesurfer": "another/wrong/path",
            "fwd_model": "yet/another/wrong/path.fif",
            "stim": "STANDING",
            "c_space": "full",
            "simulation_config_section": "0cfg",
            "screen_config_section": "cfg1",
            "freq_temp": 1,
            "freq_spacial": 0.05,
            "amplitude": 10e-9,
            "phase_offset": "np.pi / 2",
            "width": 1920,
            "height": 1080,
            "distancefrom": 78,
            "heightcm": 44.2
        }
        with open(self.config_file, 'w') as f:
            config.write(f)

    def tearDown(self):
        os.remove(self.config_file)

    def test_return_list(self):
        # test if the function returns a list object
        result = read_entry_config(self.config_file)
        self.assertIsInstance(result, list)

    def test_length_of_list(self):
        # test if the length of the returned list matches the number of sections in the config file
        result = read_entry_config(self.config_file)
        config = configparser.ConfigParser()
        config.read(self.config_file)
        self.assertEqual(len(result), len(config.sections()))

    def test_type_of_list_objects(self):
        # test if each object in the returned list is of type `utils.Entry`
        result = read_entry_config(self.config_file)
        for obj in result:
            self.assertIsInstance(obj, utils.Entry)

    def test_load_entry_function(self):
        result = read_entry_config(self.config_file)
        for obj in result:
            self.assertEqual(obj.stim, 'STANDING')
            self.assertEqual(obj.c_space, 'full')


if __name__ == '__main__':
    unittest.main()
