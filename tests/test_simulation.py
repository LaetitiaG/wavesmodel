import unittest

from utils import mri_paths
from toolbox.simulation import load_labels
from pathlib import Path


class TestLoadLabels(unittest.TestCase):
    """"
    Test function for simulation.load_labels
    Needs server to be mounted on Z: to work
    """
    subj_dir = Path('Z:\DugueLab_Research\Current_Projects\LGr_GM_JW_DH_LD_WavesModel'
                    '\Experiments\Data\data_MRI\preproc/freesurfer/2XXX72\prfs')
    paths = mri_paths((Path(subj_dir / 'lh.inferred_varea.mgz'), Path(subj_dir / 'rh.inferred_varea.mgz')),
                                (Path(subj_dir / 'lh.inferred_angle.mgz'), Path(subj_dir / 'rh.inferred_angle.mgz')),
                                (Path(subj_dir / 'lh.inferred_eccen.mgz'), Path(subj_dir / 'rh.inferred_eccen.mgz')))

    def test_return_len(self):
        ret = load_labels(self.paths)
        self.assertEqual(len(ret), 3)
        varea, angle, eccen = ret
        self.assertEqual(len(varea), 2)
        self.assertEqual(len(angle), 2)
        self.assertEqual(len(eccen), 2)


if __name__ == '__main__':
    unittest.main()
