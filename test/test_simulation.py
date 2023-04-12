import unittest

from toolbox.utils import simulation_params, screen_params
import toolbox.simulation as simulation
from toolbox.simulation import *
from pathlib import Path
import numpy as np


class TestLoadRetino(unittest.TestCase):
    """
    Test function for simulation.load_retino
    Needs server to be mounted on Z: to work
    """

    def test_return_len(self):
        mri_path = Path('Z:/DugueLab_Research/Current_Projects/LGr_GM_JW_DH_LD_WavesModel'
                        '/Experiments/Data/data_MRI/preproc/freesurfer/2XXX72/')
        ret = load_retino(mri_path)
        self.assertEqual(len(ret), 3)
        varea, angle, eccen = ret
        self.assertEqual(len(varea), 2)
        self.assertEqual(len(angle), 2)
        self.assertEqual(len(eccen), 2)

    def test_missing_input(self):
        # Test the case where the input data is missing
        mri_path = Path('./')

        # Verify that the function raises an error when the input data is missing
        with self.assertRaises(ValueError):
            result = load_retino(mri_path)

    def test_invalid_input(self):
        # Test the case where the input data is invalid
        mri_path = Path('./config')

        # Verify that the function raises an error when the input data is invalid
        with self.assertRaises(ValueError):
            result = load_retino(mri_path)


class TestCreateScreenGrid(unittest.TestCase):
    """
    Test function for simulation.create_screen_grid
    """

    def create_test_screen_config_valid(self):
        screen_config = screen_params(20, 10, 50, 10)  # 1 pix = 1cm
        # define the expected output for the upper input
        diag = np.sqrt(5 ** 2 + 10 ** 2)  # diag valu in cm
        expected_eccen_screen_max = np.degrees(np.arctan(diag / 50))
        expected_e_cort = 0
        expected_output = (expected_eccen_screen_max, expected_e_cort)
        return screen_config, expected_output

    def test_create_screen_grid_output(self):
        screen_config, expected_output = self.create_test_screen_config_valid()
        # Call the create_screen_grid function with the test data
        eccen, e_cort = create_screen_grid(screen_config)

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
    Test function for simulation.create_stim_inducer
    """
    def setUp(self) -> None:
        self.tstep = 1 / 200
        self.times = np.arange(2 / self.tstep + 1) * self.tstep
        self.params = simulation_params(5, 0.05, 10e-9, np.pi / 2)
        self.screen_config = screen_params(1920, 1080, 78, 44.2)
        _, self.e_cort = simulation.create_screen_grid(self.screen_config)

    def test_trav_out_stimulation(self):
        # Create the stimulus for TRAV_OUT stimulation
        e_cort = deepcopy(self.e_cort)
        sin_inducer = create_stim_inducer(self.screen_config, self.times, self.params, e_cort, simulation.TRAV_OUT)

        # Verify that the shape of the returned array is correct
        self.assertTrue(sin_inducer.shape == (len(self.times), self.screen_config.height, self.screen_config.width),
                        "Shape error for TRAV_OUT stimulation")

        # Verify that the array contains non-zero values
        self.assertNotEqual(np.count_nonzero(sin_inducer), 0, "Array was not filled for TRAV_OUT stimulation")

        # Verify expected value ??

    def test_standing_stimulation(self):
        # Create the stimulus for STANDING stimulation
        e_cort = deepcopy(self.e_cort)
        sin_inducer = create_stim_inducer(self.screen_config, self.times, self.params, e_cort, simulation.STANDING)

        # Verify that the shape of the returned array is correct
        self.assertTrue(sin_inducer.shape == (len(self.times), self.screen_config.height, self.screen_config.width),
                        "Shape error for STANDING stimulation")

        # Verify that the array contains non-zero values
        self.assertNotEqual(np.count_nonzero(sin_inducer), 0, "Array was not filled for STANDING stimulation")

    def test_trav_in_stimulation(self):
        # Create the stimulus for TRAV_IN stimulation
        e_cort = deepcopy(self.e_cort)
        sin_inducer = create_stim_inducer(self.screen_config, self.times, self.params, e_cort, simulation.TRAV_IN)

        # Verify that the shape of the returned array is correct
        self.assertTrue(sin_inducer.shape == (len(self.times), self.screen_config.height, self.screen_config.width),
                        "Shape error for TRAV_IN stimulation")

        # Verify that the array contains non-zero values
        self.assertNotEqual(np.count_nonzero(sin_inducer), 0, "Array was not filled for TRAV_IN stimulation")


class TestCortEccenMM(unittest.TestCase):
    """
    Test function for simulation.cort_eccen_mm
    """

    def test_valid_eccen(self):
        self.assertEqual(simulation.cort_eccen_mm(0), 0)

        # simulation.cort_eccen_mm(40) =  17.25
        # cort_eccen_mm < 30  # mm - maximal size of V1


if __name__ == '__main__':
    unittest.main()
