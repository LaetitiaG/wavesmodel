import unittest

from utils import mri_paths, simulation_params, screen_params
import toolbox.simulation as simulation
from toolbox.simulation import *
from pathlib import Path
import numpy as np


class TestLoadRetino(unittest.TestCase):
    """
    Test function for simulation.load_retino
    Needs server to be mounted on Z: to work
    """
    def create_test_mri_paths_valid(self):
        subj_dir = Path('Z:/DugueLab_Research/Current_Projects/LGr_GM_JW_DH_LD_WavesModel'
                        '/Experiments/Data/data_MRI/preproc/freesurfer/2XXX72/prfs')
        paths = mri_paths((Path(subj_dir / 'lh.inferred_varea.mgz'), Path(subj_dir / 'rh.inferred_varea.mgz')),
                          (Path(subj_dir / 'lh.inferred_angle.mgz'), Path(subj_dir / 'rh.inferred_angle.mgz')),
                          (Path(subj_dir / 'lh.inferred_eccen.mgz'), Path(subj_dir / 'rh.inferred_eccen.mgz')))
        return paths

    def create_test_mri_paths_missing_data(self):
        subj_dir = Path('./')
        paths = mri_paths((Path(subj_dir / 'lh.inferred_varea.mgz'), Path(subj_dir / 'rh.inferred_varea.mgz')),
                          (Path(subj_dir / 'lh.inferred_angle.mgz'), Path(subj_dir / 'rh.inferred_angle.mgz')),
                          (Path(subj_dir / 'lh.inferred_eccen.mgz'), Path(subj_dir / 'rh.inferred_eccen.mgz')))
        return paths

    def create_test_mri_paths_invalid_data(self):
        subj_dir = Path('./config')
        paths = mri_paths((Path(subj_dir / 'entry.ini'), Path(subj_dir / 'entry.ini')),
                          (Path(subj_dir / 'entry.ini'), Path(subj_dir / 'entry.ini')),
                          (Path(subj_dir / 'entry.ini'), Path(subj_dir / 'entry.ini')))
        return paths

    def test_return_len(self):
        mri_paths = self.create_test_mri_paths_valid()
        ret = load_retino(mri_paths)
        self.assertEqual(len(ret), 3)
        varea, angle, eccen = ret
        self.assertEqual(len(varea), 2)
        self.assertEqual(len(angle), 2)
        self.assertEqual(len(eccen), 2)

    def test_missing_input(self):
        # Test the case where the input data is missing
        mri_paths = self.create_test_mri_paths_missing_data()

        # Verify that the function raises an error when the input data is missing
        with self.assertRaises(ValueError):
            result = load_retino(mri_paths)

    def test_invalid_input(self):
        # Test the case where the input data is invalid
        mri_paths = self.create_test_mri_paths_invalid_data()

        # Verify that the function raises an error when the input data is invalid
        with self.assertRaises(ValueError):
            result = load_retino(mri_paths)


class TestCreateScreenGrid(unittest.TestCase):
    """
    Test function for simulation.create_screen_grid
    """
    def create_test_screen_config_valid(self):
        screen_config = screen_params(20, 10, 50, 10) # 1 pix = 1cm
        # define the expected output for the upper input
        diag = np.sqrt(5**2 + 10**2) # diag valu in cm
        expected_eccen_screen_max = np.degrees(np.arctan(diag/50))
        expected_e_cort = 0
        expected_output = (expected_eccen_screen_max, expected_e_cort)
        return screen_config, expected_output

    def test_create_screen_grid_output(self):
        screen_config, expected_output = self.create_test_screen_config_valid()
        # Call the create_screen_grid function with the test data
        eccen,e_cort = create_screen_grid(screen_config)

        # Verify that the output is as expected
        self.assertEqual(np.max(eccen), expected_output[0])
        # to do for e_cort too

    def test_missing_input(self):
        # Test the case where the input data is missing
        screen_config = None

        # Verify that the function raises an error when the input data is missing
        with self.assertRaises(ValueError):
            result = create_screen_grid(screen_config)

    def test_invalid_input(self):
        # Test the case where the input data is invalid
        screen_config = screen_params(1920, -1080, 78, 0)

        # Verify that the function raises an error when the input data is invalid
        with self.assertRaises(ValueError):
            result = create_screen_grid(screen_config)


class TestCreateSimInducer(unittest.TestCase):
    """
    Test function for simulation.load_labels
    """
    tstep = 1 / 200
    times = np.arange(2 / tstep + 1) * tstep
    params = simulation_params(5, 0.05, 10e-9, np.pi / 2)
    screen_config = screen_params(1920, 1080, 78, 44.2)

    def test_trav_out_stimulation(self):
        # Create the stimulus for TRAV_OUT stimulation
        _, e_cort = simulation.create_screen_grid(self.screen_config)
        sin_inducer = create_stim_inducer(self.screen_config, self.times, self.params, e_cort, simulation.TRAV_OUT)

        # Verify that the shape of the returned array is correct
        self.assertTrue(sin_inducer.shape == (len(self.times), self.screen_config.height, self.screen_config.width),
                        "Shape error for TRAV_OUT stimulation")

        # Verify that the array contains non-zero values
        self.assertNotEqual(np.count_nonzero(sin_inducer), 0, "Array was not filled for TRAV_OUT stimulation")

        # Verify expected value ??

    def test_standing_stimulation(self):
        # Create the stimulus for STANDING stimulation
        _, e_cort = simulation.create_screen_grid(self.screen_config)
        sin_inducer = create_stim_inducer(self.screen_config, self.times, self.params, e_cort, simulation.STANDING)

        # Verify that the shape of the returned array is correct
        self.assertTrue(sin_inducer.shape == (len(self.times), self.screen_config.height, self.screen_config.width),
                        "Shape error for STANDING stimulation")

        # Verify that the array contains non-zero values
        self.assertNotEqual(np.count_nonzero(sin_inducer), 0, "Array was not filled for STANDING stimulation")

    def test_trav_in_stimulation(self):
        # Create the stimulus for TRAV_IN stimulation
        _, e_cort = simulation.create_screen_grid(self.screen_config)
        sin_inducer = create_stim_inducer(self.screen_config, self.times, self.params, e_cort, simulation.TRAV_IN)

        # Verify that the shape of the returned array is correct
        self.assertTrue(sin_inducer.shape == (len(self.times), self.screen_config.height, self.screen_config.width),
                        "Shape error for TRAV_IN stimulation")

        # Verify that the array contains non-zero values
        self.assertNotEqual(np.count_nonzero(sin_inducer), 0, "Array was not filled for TRAV_IN stimulation")


if __name__ == '__main__':
    unittest.main()
