import configparser
from pathlib import Path
import unittest
import toolbox.configIO as configIO
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


if __name__ == '__main__':
    unittest.main()
