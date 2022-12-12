import unittest

from utils import mri_paths, simulation_params, screen_params
import toolbox.simulation as simulation
from toolbox.simulation import load_retino, create_stim_inducer
from pathlib import Path
import numpy as np


class TestLoadRetino(unittest.TestCase):
    """
    Test function for simulation.load_retino
    Needs server to be mounted on Z: to work
    """
    subj_dir = Path('Z:/DugueLab_Research/Current_Projects/LGr_GM_JW_DH_LD_WavesModel'
                    '/Experiments/Data/data_MRI/preproc/freesurfer/2XXX72/prfs')
    paths = mri_paths((Path(subj_dir / 'lh.inferred_varea.mgz'), Path(subj_dir / 'rh.inferred_varea.mgz')),
                      (Path(subj_dir / 'lh.inferred_angle.mgz'), Path(subj_dir / 'rh.inferred_angle.mgz')),
                      (Path(subj_dir / 'lh.inferred_eccen.mgz'), Path(subj_dir / 'rh.inferred_eccen.mgz')))

    def test_return_len(self):
        ret = load_retino(self.paths)
        self.assertEqual(len(ret), 3)
        varea, angle, eccen = ret
        self.assertEqual(len(varea), 2)
        self.assertEqual(len(angle), 2)
        self.assertEqual(len(eccen), 2)


class TestCreateSimInducer(unittest.TestCase):
    """
    Test function for simulation.create_stim_inducer
    """
    tstep = 1 / 200
    times = np.arange(2 / tstep + 1) * tstep
    params = simulation_params(5, 0.05, 10e-9, np.pi / 2)
    screen_config = screen_params(1920, 1080, 78, 44.2)

    def test_filling_array(self):
        _, e_cort = simulation.create_screen_grid(self.screen_config)
        sin_inducer = create_stim_inducer(self.screen_config, self.times, self.params, e_cort, simulation.TRAV_OUT)
        self.assertTrue(sin_inducer.shape == (len(self.times), self.screen_config.height, self.screen_config.width),
                        "Shape error")
        self.assertNotEqual(np.count_nonzero(sin_inducer), 0, "Array was not filled")

class TestCort_eccen_mm(unittest.TestCase):
    """
    Test function for simulation.cort_eccen_mm
    """   

    self.assertEqual(simulation.cort_eccen_mm(0), 0)
    
    # simulation.cort_eccen_mm(40) =  17.25        
    simulation.cort_eccen_mm < 30 # mm - maximal size of V1
    
if __name__ == '__main__':
    unittest.main()
