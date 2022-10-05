import sys
import mne
import nibabel.freesurfer.mghformat as mgh
import numpy as np
import utils

## Constant to acces lh and rh in tuple
LEFT_HEMI = 0
RIGHT_HEMI = 1


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


def cort_eccen_mm(x):
    ''' calculate distance from center of fovea in the cortex (cort_eccen in mm) given the
    % eccentricity in visual field (x in d.v.a)
    % cort_eccen = M.x
    % magnification inverse: M-1 = M0-1(1 + ax) with M0 = 23.07 and E2 = 0.75
    (from Fig. 9, Strasburger et al., 2011, values from  Horton and Hoyt (1991))
    % hence cort_eccen = x/M-1
    '''
    return np.sign(x) * (17.3 * np.abs(x) / (np.abs(x) + 0.75))


def create_screen_grid(distanceFromScreen, heightScreenCM, widthScreenPix, heightScreenPix):
    """
    Returns 2 screen grids:
    eccen_screen: each screen voxel has its eccentricity value in cm
    e_cort: corresponds to putative cortical distance
    """
    halfXscreenPix, halfYscreenPix = int(widthScreenPix / 2), int(heightScreenPix / 2)
    cmPerPixel = heightScreenCM / heightScreenPix  # cm
    degsPerPixel = np.degrees(
        2 * np.arctan(heightScreenCM / (2 * heightScreenPix * distanceFromScreen)))  # deg of visual angle
    widthArray = np.arange(-halfXscreenPix, halfXscreenPix, step=1, dtype=int)
    heightArray = np.arange(-halfYscreenPix, halfYscreenPix, step=1, dtype=int)
    [x, y] = np.meshgrid(widthArray, heightArray)  # coordinates in pixels
    eccen_screen = np.sqrt((x * cmPerPixel) ** 2 + (y * cmPerPixel) ** 2)

    # Create screen grid corresponding to putative cortical distance
    e_cort = cort_eccen_mm(np.sqrt((x * degsPerPixel) ** 2 + (y * degsPerPixel) ** 2))  # in mm of cortex
    return eccen_screen, e_cort


def create_stim_inducer(heightScreenPix, widthScreenPix, times, params, e_cort):
    sin_inducer = np.zeros((3, len(times), heightScreenPix, widthScreenPix))
    for ind_t, t in enumerate(times):
        # traveling wave (out)
        sin_inducer[0, ind_t] = params.amplitude * \
                                np.sin(2 * np.pi * params.freq_spacial *
                                       e_cort - 2 * np.pi * params.freq_temp * t + params.phase_offset)

        # standing wave
        sin_inducer[1, ind_t] = params.amplitude * \
                                np.sin(2 * np.pi * params.freq_spacial * e_cort + params.phase_offset) * \
                                np.cos(2 * np.pi * params.freq_temp * t)

        # traveling wave (in)
        sin_inducer[2, ind_t] = params.amplitude * \
                                np.sin(2 * np.pi * params.freq_spacial *
                                       e_cort + 2 * np.pi * params.freq_temp * t + params.phase_offset)
    return sin_inducer


def map_stim_value(times, sin_inducer, eccen_screen, eccen_label):
    wave_label = np.zeros((3, len(eccen_label), len(times)))
    for ind_l, l in enumerate(eccen_label):
        if l > np.max(eccen_screen):
            continue
        imin = np.argmin(np.abs(eccen_screen - eccen_label[ind_l]))
        ind_stim = np.unravel_index(imin, np.shape(eccen_screen))
        wave_label[0, ind_l] = sin_inducer[0, :, ind_stim[0], ind_stim[1]]  # trav
        wave_label[1, ind_l] = sin_inducer[1, :, ind_stim[0], ind_stim[1]]  # stand
        wave_label[2, ind_l] = sin_inducer[2, :, ind_stim[0], ind_stim[1]]  # trav in
    return wave_label


def main(sensorsFile, mri_paths):
    ## Magic numbers -- must be parameters later
    distanceFromScreen = 78  # cm
    heightScreenCM = 44.2  # cm
    widthScreenPix = 1920  # pixels
    heightScreenPix = 1080  # pixels
    params = utils.simulation_params(5, 0.05, 10e-9, np.pi / 2)
    snr = 20
    noise_amp = params.amplitude / snr  # noise amplitude corresponding to snr
    rng = np.random.RandomState()

    info = mne.io.read_info(sensorsFile)
    # Time Parameters for the source signal
    tstep = 1 / info['sfreq']
    times = np.arange(2 / tstep + 1) * tstep

    inds_label, angle_label, eccen_label = load_labels(mri_paths)

    eccen_screen, e_cort = create_screen_grid(distanceFromScreen, heightScreenCM, widthScreenPix, heightScreenPix)
    # Recreate stim inducer
    sin_inducer = create_stim_inducer(heightScreenPix, widthScreenPix, times, params, e_cort)

    # Map stim values on voxel label (for lh and rh labels)
    def map_stim(eccen_lbl): map_stim_value(times, sin_inducer, eccen_screen, eccen_lbl)

    wave_label = apply_tuple(eccen_label, map_stim)


if __name__ == '__main__':
    file = sys.argv[1]
