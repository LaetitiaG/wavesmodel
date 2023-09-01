import sys
import mne
import nibabel.freesurfer.mghformat as mgh
import numpy as np
from copy import deepcopy
from toolbox import utils
from numba import jit, float64

## Constant to acces lh and rh in tuple
LEFT_HEMI = 0
RIGHT_HEMI = 1

TRAV_OUT = "TRAV_OUT"
STANDING = "STANDING"
TRAV_IN = "TRAV_IN"
TRAV_OUT_STAND = "TRAV_OUT_STAND"

# select in which label the simulation is done 
# 1	V1 / 2	V2 / 3	V3 / 4	hV4 / 5	VO1 / 6	VO2 / 7	LO1 / 8	LO2 / 9	TO1
# 10	TO2 / 11	V3b / 12	V3a
lab_ind = 1


def apply_tuple(t, f):
    """ Utility function to apply function 'f' to both elements of tuple 't'
    """
    x, y = t
    return f(x), f(y)


def apply_mask(msk, tpl):
    """ Utility function to apply mask as tuple to a tuple
    """
    x, y = tpl
    m1, m2 = msk
    return x[m1], y[m2]


def safe_tupple_load(retino_tuple):
    """ Applies the mgh.load funtion to both elements of the tuple
    """
    try:
        return apply_tuple(retino_tuple, mgh.load)
    except:
        raise ValueError('Invalid input data')


def cort_eccen_mm(x):
    ''' calculate distance from center of fovea in the cortex (cort_eccen in mm) given the
    % eccentricity in visual field (x in d.v.a)
    % cort_eccen = M.x
    % magnification inverse: M-1 = M0-1(1 + ax) with M0 = 23.07 and E2 = 0.75
    (from Fig. 9, Strasburger et al., 2011, values from  Horton and Hoyt (1991))
    % hence cort_eccen = x/M-1
    '''
    return np.sign(x) * (17.3 * np.abs(x) / (np.abs(x) + 0.75))


def create_sim_from_entry(entry):
    return Simulation(
        measured=entry.measured,
        freesurfer=entry.freesurfer,
        forward_model=entry.forward_model,
        simulation_params=entry.simulation_params,
        screen_params=entry.screen_params,
        stim=entry.stim,
        c_space=entry.c_space
    )


class Simulation:
    def __init__(self,
                 measured,
                 freesurfer,
                 forward_model,
                 simulation_params,
                 screen_params,
                 stim,
                 c_space,
                 verbose=False):
        self.measured = measured
        self.freesurfer = freesurfer
        self.forward_model = forward_model
        self.simulation_params = simulation_params
        self.screen_params = screen_params
        self.stim = stim
        self.c_space = c_space
        self.verbose = verbose

        # Time Parameters for the source signal
        info = mne.io.read_info(self.measured, verbose=self.verbose)
        self.tstep = 1 / info['sfreq']
        self.times = np.arange(2 / self.tstep + 1) * self.tstep
        
    def __repr__(self):
        attributes = []
        for attr, value in self.__dict__.items():
            attributes.append(f"{attr}={value!r}")
        return "Sim(\n  " + ",\n  ".join(attributes) + "\n)" 
    
    def __load_retino(self):
        """
        Load retinotopy, visual phase and eccentricity for labels of both hemis

        Returns: A tuple containing the following elements:
            - inds_label (tuple): A tuple containing the indices of the labeled voxels in the left and right hemispheres.
            - angle_label (tuple): A tuple containing the angle values for the labeled voxels
                in the left and right hemispheres.
            - eccen_label (tuple): A tuple containing the eccentricity values for the labeled voxels
                in the left and right hemispheres.
        """
        prfs = self.freesurfer / 'prfs'
        if not prfs.exists():
            raise ValueError('Invalid freesurfer folder architecture')

        retino_paths = utils.mri_paths((prfs / 'lh.inferred_varea.mgz', prfs / 'rh.inferred_varea.mgz'),
                                       (prfs / 'lh.inferred_angle.mgz', prfs / 'rh.inferred_angle.mgz'),
                                       (prfs / 'lh.inferred_eccen.mgz', prfs / 'rh.inferred_eccen.mgz'))
        if any(any(not path.exists() for path in tup) for tup in retino_paths):
            raise ValueError('Input data is missing or has invalid name')

        retino_labels = safe_tupple_load(retino_paths.varea)
        # Select V1 (according to the codes used in varea)
        msk_label = apply_tuple(retino_labels, lambda x: x.get_fdata() == lab_ind)

        def mask(tpl):
            return apply_mask(msk=msk_label, tpl=tpl)

        inds_label = apply_tuple(retino_labels,
                                   lambda x: np.where(np.squeeze(x.get_fdata()) == lab_ind)[0])
        angle = safe_tupple_load(retino_paths.angle)
        angle_label = mask(apply_tuple(angle, lambda x: x.get_fdata()))
        eccen = safe_tupple_load(retino_paths.eccen)
        eccen_label = mask(apply_tuple(eccen, lambda x: x.get_fdata()))
        return inds_label, angle_label, eccen_label

    def __create_screen_grid(self):
        """
        Returns 2 screen grids:
        eccen_screen: grid of eccentricity value for each pixel (in °VA from the fovea)
        e_cort: grid of putative cortical distance for each pixel (in mm of cortex)

        Args:
            screen_config: A named tuple with the following fields, see :class:`toolbox.utils.Entry`:
                - width (int): The width of the screen in pixels.
                - height (int): The height of the screen in pixels.
                - distanceFrom (float): The distance from the screen in cm.
                - heightCM (float): The height of the screen in cm.

        Returns:
            A tuple containing the following elements:
                - eccen_screen (np.ndarray): A 2D array of eccentricity values.
                - e_cort (np.ndarray): A 2D array of cortical distances.
        """
        # Check if any of the screen_config values are invalid
        if any([value <= 0 for value in self.screen_params]):
            raise ValueError('Invalid input data')

        width_screen_pix = self.screen_params.width
        height_screen_pix = self.screen_params.height
        height_screen_cm = self.screen_params.heightCM
        distance_from_screen = self.screen_params.distanceFrom

        # Find pixel coordinates of the center
        half_x_screen_pix = int(width_screen_pix / 2)
        half_y_screen_pix = int(height_screen_pix / 2)

        # Create grids in pixels then in cm from the center
        width_array = np.arange(-half_x_screen_pix, half_x_screen_pix, step=1, dtype=int)
        height_array = np.arange(-half_y_screen_pix, half_y_screen_pix, step=1, dtype=int)
        x, y = np.meshgrid(width_array, height_array)  # coordinates in pixels
        cm_per_pixel = height_screen_cm / height_screen_pix  # cm
        eccen_screen_cm = np.sqrt((x * cm_per_pixel) ** 2 + (y * cm_per_pixel) ** 2)

        # Create grids in °VA
        eccen_screen = np.degrees(np.arctan(eccen_screen_cm / distance_from_screen))

        # Create screen grid corresponding to putative cortical distance
        e_cort = cort_eccen_mm(eccen_screen)  # in mm of cortex

        return eccen_screen, e_cort

    def __create_stim_inducer(self, e_cort):
        """
        Create the visual stimulus presented on the screen, which should induce cortical waves.

        Args:
            screen_config: A named tuple with the following fields:
                - width (int): The width of the screen in pixels.
                - height (int): The height of the screen in pixels.
                - distanceFrom (float): The distance from the screen in cm.
                - heightCM (float): The height of the screen in cm.
            times (ndarray): An array of time points.
            params: A named tuple with the following fields:
                - amplitude (float): The amplitude of the stimulus.
                - freq_spatial (float): The spatial frequency of the stimulus.
                - freq_temp (float): The temporal frequency of the stimulus.
                - phase_offset (float): The phase offset of the stimulus.
            e_cort (ndarray): An array of cortical distances.
            stim (str): The type of stimulation, one of `TRAV_OUT`, `STANDING`, or `TRAV_IN`.

        Returns:
            An ndarray containing the screen luminance values for each time point and pixel.
        """
        params = self.simulation_params
        if self.stim == TRAV_OUT:
            @jit(nopython=True)
            def func(t):
                return params.amplitude * \
                       np.sin(2 * np.pi * params.freq_spatial *
                              e_cort - 2 * np.pi * params.freq_temp * t + params.phase_offset)
        elif self.stim == STANDING:
            @jit(nopython=True)
            def func(t):                    
                return params.amplitude * \
                       np.sin(2 * np.pi * params.freq_spatial * e_cort + params.phase_offset) * \
                       np.cos(2 * np.pi * params.freq_temp * t)
        elif self.stim == TRAV_IN:
            @jit(nopython=True)
            def func(t):                      
                return params.amplitude * \
                       np.sin(2 * np.pi * params.freq_spatial *
                              e_cort + 2 * np.pi * params.freq_temp * t + params.phase_offset)
        elif self.stim == TRAV_OUT_STAND:
            @jit(nopython=True)
            def func(t):
                return params.amplitude * \
                       np.sin(2 * np.pi * params.freq_spatial *
                              e_cort - 2 * np.pi * params.freq_temp * t + params.phase_offset) +  params.amplitude * \
                              np.sin(2 * np.pi * params.freq_spatial * e_cort + params.phase_offset) * \
                              np.cos(2 * np.pi * params.freq_temp * t)
        else:
            raise ValueError('Incorrect stimulation value')  # needs to be InputStimError

        sin_inducer = np.zeros((len(self.times), self.screen_params.height, self.screen_params.width))
        # apply func on times
        for idx, time in enumerate(self.times):
            sin_inducer[idx] = func(time)
            
        # add decay from the eccentricity e0
        e_cort_e0 = cort_eccen_mm(params.e0)
        decay_area = e_cort >= e_cort_e0
        sin_inducer[:,decay_area] = sin_inducer[:,decay_area] * np.exp(-params.decay * (e_cort[decay_area] - e_cort_e0)) 
        
        return sin_inducer

    def __create_wave_stims(self, sin_inducer, eccen_screen, angle_label, eccen_label):
        """
        Map stim values on voxel label (for lh and rh labels)
        And return wave_label depending on c_space (full, quad, fov)
            Used with apply_tuple, avoiding to have to handle tuple inside
        """

        times = self.times
        @jit(float64[:, :](float64[:]), nopython=True)
        def __create_wave_label_single_hemi(eccen_label_hemi):
            """
            Returns 1 hemisphere of wave label
            To be used with apply_tuple
            """
            wave_label_h = np.zeros((len(eccen_label_hemi), len(times)), dtype=np.float64)
            max_eccen = np.max(eccen_screen)
            for ind_l, l in enumerate(eccen_label_hemi):
                if l > max_eccen:
                    continue
                imin = np.argmin(np.abs(eccen_screen - eccen_label_hemi[ind_l]))
                ind_stim = (imin // eccen_screen.shape[1], imin % eccen_screen.shape[1])  # np.unravel_index(imin, np.shape(eccen_screen)) Not supported by numba
                wave_label_h[ind_l] = sin_inducer[:, ind_stim[0], ind_stim[1]]

            return wave_label_h

        wave_label = apply_tuple(eccen_label, __create_wave_label_single_hemi)
        if self.c_space == 'full':
            return wave_label
        if self.c_space == 'quad':
            # Create wave stim for quadrant condition (session 1)
            wave_quad = deepcopy(wave_label[LEFT_HEMI])
            mask_quad = (angle_label[LEFT_HEMI] > 90 + 5) & (angle_label[LEFT_HEMI] < 180 - 5)
            mask = np.invert(mask_quad)  # right lower quadrant
            wave_quad[mask, :] = 0
            return wave_quad
        elif self.c_space == 'fov':
            # Create wave stim for foveal condition (session 2)
            wave_fov = apply_tuple(wave_label, deepcopy)
            wave_fov[LEFT_HEMI][eccen_label[0] > 5, :] = 0
            wave_fov[RIGHT_HEMI][eccen_label[1] > 5, :] = 0
            return wave_fov
        else:  # handle wrong c_space
            return wave_label

    def __create_stc(self, verbose=False):
        # Load forward model
        fwd = mne.read_forward_solution(self.forward_model, verbose=verbose)
        src = fwd['src']  # source space

        # Create empty stc
        # need to indicate the directory of the freesurfer files for given subject
        labels = mne.read_labels_from_annot(self.freesurfer.name, subjects_dir=self.freesurfer.parent,
                                            parc='aparc.a2009s', verbose=verbose)  # aparc.DKTatlas
        n_labels = len(labels)
        signal = np.zeros((n_labels, len(self.times)))
        return mne.simulation.simulate_stc(src, labels, signal, self.times[0], self.tstep, allow_overlap = True,
                                           value_fun=lambda x: x)  # labels or label_sel

    def __fill_stc(self, stc_gen, inds_label, angle_label, eccen_label, wave_label):
        stc_angle = stc_gen.copy()  
        stc_eccen = stc_gen.copy()
        tmp = None

        if (self.c_space == 'full') | (self.c_space == 'fov'):
            tmp = stc_gen.copy()
            for i in inds_label[LEFT_HEMI]:
                if i in stc_gen.lh_vertno:
                    i_stc = np.where(i == stc_gen.lh_vertno)[0][0]
                    tmp.lh_data[i_stc] = wave_label[LEFT_HEMI][inds_label[LEFT_HEMI] == i]
                    stc_eccen.lh_data[i_stc] = eccen_label[LEFT_HEMI][inds_label[LEFT_HEMI] == i]
                    stc_angle.lh_data[i_stc] = angle_label[LEFT_HEMI][inds_label[LEFT_HEMI] == i]

            for i in inds_label[RIGHT_HEMI]:
                if i in stc_gen.rh_vertno:
                    i_stc = np.where(i == stc_gen.rh_vertno)[0][0]
                    tmp.rh_data[i_stc] = wave_label[RIGHT_HEMI][inds_label[RIGHT_HEMI] == i]
                    stc_eccen.rh_data[i_stc] = eccen_label[RIGHT_HEMI][inds_label[RIGHT_HEMI] == i]
                    stc_angle.rh_data[i_stc] = angle_label[RIGHT_HEMI][inds_label[RIGHT_HEMI] == i]

        elif self.c_space == 'quad':
            tmp = stc_gen.copy()
            for i in inds_label[LEFT_HEMI]:
                if i in stc_gen.lh_vertno:
                    i_stc = np.where(i == stc_gen.lh_vertno)[0][0]
                    tmp.lh_data[i_stc] = wave_label[inds_label[0] == i]

        return tmp

    def generate_simulation(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        # à revoir
        labels = self.__load_retino()
        inds_label, angle_label, eccen_label = labels

        # Create the visual stimulus presented on the screen, which should induced cortical waves
        eccen_screen, e_cort = self.__create_screen_grid()
        stim_inducer = self.__create_stim_inducer(e_cort)

        # return wave_label depending on c_space (full, quad or fov)
        wave_label = self.__create_wave_stims(stim_inducer, eccen_screen, angle_label, eccen_label)

        stc_gen = self.__create_stc(verbose=verbose)

        # only wave_label (which depends on c_space)
        stc = self.__fill_stc(stc_gen, *labels, wave_label)

        return stc
