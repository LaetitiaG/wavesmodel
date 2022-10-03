import sys
import nibabel.freesurfer.mghformat as mgh
import numpy as np


def apply_tuple(t, f):
    x, y = t
    return f(x), f(y)


def apply_mask(msk, tpl):
    x, y = tpl
    m1, m2 = msk
    return x[m1], y[m2]


def load_labels(mri_paths):
    """
    Load retinotopy, visual phase and eccentricity for labels of both hemis
    Return follows the form of utils.mri_paths with 2-tuple for both hemis
    """
    retino_labels = apply_tuple(mri_paths.varea, mgh.load)
    # Select V1 (according to the codes used in varea)
    # 1	V1 / 2	V2 / 3	V3 / 4	hV4 / 5	VO1 / 6	VO2 / 7	LO1 / 8	LO2 / 9	TO1
    # 10	TO2 / 11	V3b / 12	V3a
    lab_ind = 1
    msk_label = apply_tuple(retino_labels, lambda x: x.get_fdata() == lab_ind)
    def mask(tpl): return apply_mask(msk=msk_label, tpl=tpl)
    inds_label = apply_tuple(retino_labels,
                             lambda x: np.where(np.squeeze(x.get_fdata()) == lab_ind)[0])
    angle = apply_tuple(mri_paths.angle, mgh.load)
    angle_label = mask(apply_tuple(angle, lambda x: x.get_fdata()))
    eccen = apply_tuple(mri_paths.eccen, mgh.load)
    eccen_label = mask(apply_tuple(eccen, lambda x: x.get_fdata()))
    return inds_label, angle_label, eccen_label


if __name__ == '__main__':
    file = sys.argv[1]
