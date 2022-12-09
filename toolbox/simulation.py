import sys
import mne
import nibabel.freesurfer.mghformat as mgh
import numpy as np
from copy import deepcopy
import utils
from collections import namedtuple

## Constant to acces lh and rh in tuple
LEFT_HEMI = 0
RIGHT_HEMI = 1

TRAV_OUT = "trav_out"
STANDING = "standing"
TRAV_IN = "trav_in"

# select in which label the simulation is done 
# 1	V1 / 2	V2 / 3	V3 / 4	hV4 / 5	VO1 / 6	VO2 / 7	LO1 / 8	LO2 / 9	TO1
# 10	TO2 / 11	V3b / 12	V3a
lab_ind = 1 


def apply_tuple(t, f):
    x, y = t
    return f(x), f(y)


def apply_mask(msk, tpl):
    x, y = tpl
    m1, m2 = msk
    return x[m1], y[m2]


def safe_tupple_load(retino_tuple):
    try:
        return apply_tuple(retino_tuple, mgh.load)
    except:
        raise ValueError('Invalid input data')


def load_retino(mri_paths):
    """
    Load retinotopy, visual phase and eccentricity for labels of both hemis
    Args:
        mri_paths: A named tuple with the following fields:
            - varea (tuple): A tuple containing the paths to the varea MRIs for the left and right hemispheres.
            - angle (tuple): A tuple containing the paths to the angle MRIs for the left and right hemispheres.
            - eccen (tuple): A tuple containing the paths to the eccen MRIs for the left and right hemispheres.

    Returns: A tuple containing the following elements:
        - inds_label (tuple): A tuple containing the indices of the labeled voxels in the left and right hemispheres.
        - angle_label (tuple): A tuple containing the angle values for the labeled voxels
            in the left and right hemispheres.
        - eccen_label (tuple): A tuple containing the eccentricity values for the labeled voxels
            in the left and right hemispheres.
    """
    if any(path is None for path in mri_paths):
        raise ValueError('Missing input data')
    retino_labels = safe_tupple_load(mri_paths.varea)
    # Select V1 (according to the codes used in varea)
    msk_label = apply_tuple(retino_labels, lambda x: x.get_fdata() == lab_ind)

    def mask(tpl): return apply_mask(msk=msk_label, tpl=tpl)

    inds_label = apply_tuple(retino_labels,
                             lambda x: np.where(np.squeeze(x.get_fdata()) == lab_ind)[0])
    angle = safe_tupple_load(mri_paths.angle)
    angle_label = mask(apply_tuple(angle, lambda x: x.get_fdata()))
    eccen = safe_tupple_load(mri_paths.eccen)
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


def create_screen_grid(screen_config):
    """
    Returns 2 screen grids:
    eccen_screen: each screen voxel has its eccentricity value in cm
    e_cort: corresponds to putative cortical distance

    Args:
        screen_config: A named tuple with the following fields:
            - width (int): The width of the screen in pixels.
            - height (int): The height of the screen in pixels.
            - distanceFrom (int): The distance from the screen in cm.
            - heightCM (int): The height of the screen in cm.

    Returns:
        A tuple containing the following elements:
            - eccen_screen (np.ndarray): A 2D array of eccentricity values for each screen voxel.
            - e_cort (np.ndarray): A 2D array of putative cortical distances.
    """
    # Check if any of the screen_config values are invalid
    if any(value <= 0 for value in screen_config):
        raise ValueError('Invalid input data')

    width_screen_pix = screen_config.width
    height_screen_pix = screen_config.height
    height_screen_cm = screen_config.heightCM
    distance_from_screen = screen_config.distanceFrom
    half_x_screen_pix = int(width_screen_pix / 2)
    half_y_screen_pix = int(height_screen_pix / 2)
    cm_per_pixel = height_screen_cm / height_screen_pix  # cm
    degs_per_pixel = np.degrees(
        2 * np.arctan(height_screen_cm / (2 * height_screen_pix * distance_from_screen)))  # deg of visual angle
    width_array = np.arange(-half_x_screen_pix, half_x_screen_pix, step=1, dtype=int)
    height_array = np.arange(-half_y_screen_pix, half_y_screen_pix, step=1, dtype=int)
    x, y = np.meshgrid(width_array, height_array)  # coordinates in pixels
    eccen_screen = np.sqrt((x * cm_per_pixel) ** 2 + (y * cm_per_pixel) ** 2)

    # Create screen grid corresponding to putative cortical distance
    e_cort = cort_eccen_mm(np.sqrt((x * degs_per_pixel) ** 2 + (y * degs_per_pixel) ** 2))  # in mm of cortex

    return eccen_screen, e_cort


def create_stim_inducer(screen_config, times, params, e_cort, stim):
    """
    Create the visual stimulus presented on the screen, which should induced cortical waves.
    Return values of the screen luminance for each time point and pixel.
    """
    sin_inducer = np.zeros((len(times), screen_config.height, screen_config.width))

    if stim == TRAV_OUT:
        def func(t): return params.amplitude * \
                       np.sin(2 * np.pi * params.freq_spacial *
                              e_cort - 2 * np.pi * params.freq_temp * t + params.phase_offset)
    elif stim == STANDING:
        def func(t): return params.amplitude * \
                         np.sin(2 * np.pi * params.freq_spacial * e_cort + params.phase_offset) * \
                         np.cos(2 * np.pi * params.freq_temp * t)
    elif stim == TRAV_IN:
        def func(t): return params.amplitude * \
                         np.sin(2 * np.pi * params.freq_spacial *
                                e_cort + 2 * np.pi * params.freq_temp * t + params.phase_offset)
    else:
        raise ValueError('Incorrect stimulation value')  # needs to be InputStimError
    # apply func on times
    for idx, time in enumerate(times):
        sin_inducer[idx] = func(time)
    return sin_inducer


def map_stim_value(times, sin_inducer, eccen_screen, eccen_label):
    """
    Map stim values on voxel label (for lh and rh labels)
    Used with apply_tuple, avoiding to have to handle tuple inside
    """
    wave_label = np.zeros(len(eccen_label), len(times))
    for ind_l, l in enumerate(eccen_label):
        if l > np.max(eccen_screen):
            continue
        imin = np.argmin(np.abs(eccen_screen - eccen_label[ind_l]))
        ind_stim = np.unravel_index(imin, np.shape(eccen_screen))
        wave_label[ind_l] = sin_inducer[:, ind_stim[0], ind_stim[1]]
    return wave_label


def create_wave_stims(wave_label, angle_label, eccen_label):
    # Create wave stim for quadrant condition (session 1)
    wave_quad = deepcopy(wave_label[LEFT_HEMI])
    mask_quad = (angle_label[LEFT_HEMI] > 90 + 5) & (angle_label[LEFT_HEMI] < 180 - 5)
    mask = np.invert(mask_quad)  # right lower quadrant
    wave_quad[mask, :] = 0

    # Create wave stim for foveal condition (session 2)
    wave_fov = apply_tuple(wave_label, deepcopy)
    wave_fov[LEFT_HEMI][eccen_label[0] > 5, :] = 0
    wave_fov[RIGHT_HEMI][eccen_label[1] > 5, :] = 0

    # Create stim for half left stimulated V1 = cond, other half = other cond
    wave_halfHalf = np.zeros(np.shape(wave_label[LEFT_HEMI]))
    # Handle this case of inverting wave types when only 1 is now available
    wave_halfHalf[0] = wave_label[LEFT_HEMI][1]
    wave_halfHalf[1] = wave_label[RIGHT_HEMI][0]  # switch wave types
    wave_halfHalf[mask_quad, :] = wave_quad[mask_quad, :]

    return wave_quad, wave_fov, wave_halfHalf


def create_stc(forward_model, times, tstep, mri_path):
    # Load forward model
    fwd = mne.read_forward_solution(forward_model)
    src = fwd['src']  # source space

    # Create empty stc
    # need to indicate the directory of the freesurfer files for given subject
    labels = mne.read_labels_from_annot(mri_path.name, subjects_dir=mri_path.parent, parc='aparc.a2009s')  # aparc.DKTatlas
    n_labels = len(labels)
    signal = np.zeros((n_labels, len(times)))
    return mne.simulation.simulate_stc(src, labels, signal, times[0], tstep,
                                       value_fun=lambda x: x)  # labels or label_sel


def fill_stc(stc_gen, c_space, inds_label, angle_label, eccen_label, wave_label, wave_quad):
    stc_angle = stc_gen.copy()  # only for left hemisphere
    stc_eccen = stc_gen.copy()

    if c_space == 'full':
        tmp = stc_gen.copy()
        for i in inds_label[0]:  # lh
            if i in stc_gen.lh_vertno:
                i_stc = np.where(i == stc_gen.lh_vertno)[0][0]
                tmp.lh_data[i_stc] = wave_label[0][inds_label[0] == i]
                stc_eccen.lh_data[i_stc] = eccen_label[0][inds_label[0] == i]
                stc_angle.lh_data[i_stc] = angle_label[0][inds_label[0] == i]

        for i in inds_label[1]:  # rh
            if i in stc_gen.rh_vertno:
                i_stc = np.where(i == stc_gen.rh_vertno)[0][0]
                tmp.rh_data[i_stc] = wave_label[1][inds_label[1] == i]
                stc_eccen.rh_data[i_stc] = eccen_label[1][inds_label[1] == i]
                stc_angle.rh_data[i_stc] = angle_label[1][inds_label[1] == i]
        return tmp

    elif c_space == 'quad':
        tmp = stc_gen.copy()
        for i in inds_label[0]:  # lh
            if i in stc_gen.lh_vertno:
                i_stc = np.where(i == stc_gen.lh_vertno)[0][0]
                tmp.lh_data[i_stc] = wave_quad[inds_label[0] == i]

        return tmp


def fill_wave_activity(stc_gen, inds_label, wave_halfHalf):
    ## V1 with half cond, the other half other cond
    tmp = stc_gen.copy()
    for i in inds_label[0]:  # lh
        if i in stc_gen.lh_vertno:
            i_stc = np.where(i == stc_gen.lh_vertno)[0][0]
            tmp.lh_data[i_stc] = wave_halfHalf[inds_label[0] == i]

    return tmp


def generate_simulation(sensorsFile, mri_paths, forward_model, stim, mri_path):
    # Magic numbers -- must be parameters later
    screen_config = utils.screen_params(1920, 1080, 78, 44.2)
    params = utils.simulation_params(5, 0.05, 10e-9, np.pi / 2)
    c_space = 'full'

    info = mne.io.read_info(sensorsFile)
    # Time Parameters for the source signal
    tstep = 1 / info['sfreq']
    times = np.arange(2 / tstep + 1) * tstep

    labels = load_retino(mri_paths)
    inds_label, angle_label, eccen_label = labels

    # Create the visual stimulus presented on the screen, which should induced cortical waves
    eccen_screen, e_cort = create_screen_grid(screen_config)
    sin_inducer = create_stim_inducer(screen_config, times, params, e_cort, stim)

    #define function to handle eccen_label as a tuple of hemis
    def map_stim(eccen_lbl): map_stim_value(times, sin_inducer, eccen_screen, eccen_lbl)

    wave_label = apply_tuple(eccen_label, map_stim)
    wave_quad, wave_fov, wave_halfHalf = create_wave_stims(wave_label, angle_label, eccen_label)

    stc_gen = create_stc(forward_model, times, tstep, mri_path)
    
    stc_wave = fill_wave_activity(stc_gen, inds_label, wave_halfHalf)
    stc = fill_stc(stc_gen, c_space, *labels, wave_label, wave_quad)


if __name__ == '__main__':
    file = sys.argv[1]
