##############################################################################
#### encodingModel.py
##############################################################################
# -*- coding: utf-8 -*-
"""
Compute a source estimate with activity in a given label and project this onto
sensors space.
Created on Thu Apr 22 10:07:06 2021

@author: laeti
"""

import os
import os.path as op
import preproc
import mne
import numpy as np
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import nibabel.freesurfer.mghformat as mgh
import scipy
from scipy.optimize import curve_fit
from copy import deepcopy
import matplotlib.animation as animation
import utils


##############################################################################
### Functions ###
##############################################################################

def e_mm(x):
    ''' calculate distance from center of fovea in the cortex (cort_eccen in mm) given the 
    % eccentricity in visual field (x in d.v.a)
    % cort_eccen = M.x  
    % magnification inverse: M-1 = M0-1(1 + ax) with M0 = 23.07 and E2 = 0.75(from Fig. 9, Strasburger et al., 2011, values from  Horton and Hoyt (1991))
    % hence cort_eccen = x/M-1
    '''

    cort_eccen = np.sign(x) * (17.3 * np.abs(x) / (np.abs(x) + 0.75))

    return cort_eccen


def cort_dist(X, M0, E2):
    '''
    Equation for cortical magnification: xcort = x. M0*E2/(x+E2)
    Used to fit cortical distance (mm) to eccentricity (°VA) with M0 and E2
    as free parameters

    Parameters
    ----------
    X : 1*N array
        Array with eccentricity (N voxels).
    b : FLOAT
        Weight to be fitted.
    theta : FLOAT
        Phase reference to be fitted.
    inter : FLOAT
        Intercept to be fitted.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return X * M0 * E2 / (E2 + X)


def magnif_inv(X, M0, E2):
    '''
    Equation for inverse of cortical magnification: M-1 = 1/M0*(1+x/E2)
    Used to fit cortical distance (mm) to eccentricity (°VA) with M0 and E2
    as free parameters.

    Parameters
    ----------
    X : 1*N array
        Array with eccentricity (N voxels).
    b : FLOAT
        Weight to be fitted.
    theta : FLOAT
        Phase reference to be fitted.
    inter : FLOAT
        Intercept to be fitted.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return 1 / M0 * (1 + X / E2)


##############################################################################
### Parameters ###
##############################################################################

from config import datapath, preprocpath, subjects, subjects_dir, resultspath, simupath, cond_waves, cond_space

plot_stc = False  # plot stc or not

# Screen parameters used in the MEEG experiment
distanceFromScreen = 78  # cm
heightScreenCM = 44.2  # cm
widthScreenPix = 1920  # pixels
heightScreenPix = 1080  # pixels
halfXscreenPix = int(widthScreenPix / 2)
halfYscreenPix = int(heightScreenPix / 2)
cmPerPixel = heightScreenCM / heightScreenPix  # cm
degsPerPixel = np.degrees(
    2 * np.arctan(heightScreenCM / (2 * heightScreenPix * distanceFromScreen)))  # deg of visual angle

# conditions
hemis = ['lh', 'rh']

# using the namedtuple data structure to store simulated signal parameters
params = utils.simulation_params(5, 0.05, 10e-9, np.pi / 2)

# simulated signal parameters
tfreq = params.freq_temp  # temporal frequency of the wave_inducer: 5 Hz (cycles/s)
sfreq = params.freq_spacial  # spatial freq wave_inducer: 0.05 cycles/mm #  with 5Hz it corresponds to v = 0.1 m/s wavelength = 20 mm
A = params.amplitude  # 10e-9 amplitude 10nA  because 10e-9 means (10 * 10^-9) # GM - but this was making the signla
# go from +10 nA to -10nA, so the _wave_ amplitude (peak to trough) was actually 20 nA. If we want
# this "A" to be the wave amplitude then we have to divide by 2 in the equation
phi = params.phase_offset  # initial phase so that center is not a (null) node for standing wave

snr = 20
noise_amp = A / snr  # noise amplitude corresponding to snr
rng = np.random.RandomState()

##############################################################################
### Main ###
##############################################################################


#### Read data and restrict to a given label
##############################################################################

# loop across subject
speed_label_all, global_speed_all = {}, {}
for s in range(len(subjects)):
    subject = subjects[s]
    print(subject)

    # Read info from raw data --- this is used only to find 'sfreq' (== sample rate)
    fname_dic = preproc.read_filesName(datapath + subject)
    sessions = list(fname_dic.keys())
    sname = fname_dic['session1']['func'][0][
            :-4] + '_preproc_raw_tsss.fif'  # replace 'run01.fif' by 'run01_preproc...tsss.fif'
    info = mne.io.read_info(
        os.path.join(preprocpath, fname_dic['session1']['subj'], fname_dic['session1']['ses'], sname))

    # Load retinotopy, visual phase and eccentricity for labels of both hemi
    inds_label = []
    angle_label = []
    eccen_label = []
    dist = []
    pRF_label = []
    for ind_h, hemi in enumerate(hemis):
        # Load retinotopy
        fname = op.join(subjects_dir, subject, 'prfs', hemi + '.inferred_varea.mgz')
        retino_labels = mgh.load(fname)

        # Select V1 (according to the codes used in varea)
        # 1	V1 / 2	V2 / 3	V3 / 4	hV4 / 5	VO1 / 6	VO2 / 7	LO1 / 8	LO2 / 9	TO1
        # 10	TO2 / 11	V3b / 12	V3a
        lab_ind = 1
        msk_label = retino_labels.get_fdata() == lab_ind
        inds_label.append(np.where(np.squeeze(retino_labels.get_fdata()) == lab_ind)[0])

        # Load pRF values
        fname = op.join(subjects_dir, subject, 'prfs', hemi + '.inferred_angle.mgz')
        angle = mgh.load(fname)
        angle_label.append(angle.get_fdata()[msk_label])

        # Load angle within the selected label (in °VA from the fovea)
        # unused
        fname = op.join(subjects_dir, subject, 'prfs', hemi + '.inferred_sigma.mgz')
        pRF = mgh.load(fname)
        pRF_label.append(angle.get_fdata()[msk_label])

        # Load eccentricity within the selected label (in °VA from the fovea)
        fname = op.join(subjects_dir, subject, 'prfs', hemi + '.inferred_eccen.mgz')
        eccen = mgh.load(fname)
        eccen_label.append(eccen.get_fdata()[msk_label])

        # convert visual angle into cm (distance on screen)
        dist.append(distanceFromScreen * np.arctan(np.radians(eccen.get_fdata()[msk_label])))  # in cm

    # Time Parameters for the source signal
    # can be replace by 'sensor' files --> sensor[sfreq] == preproc[sfreq] / 5 -- 5 == 'decim'
    tstep = 1 / (info['sfreq'] / 5)
    times = np.arange(2 / tstep + 1) * tstep


    ####  Create source activity based on stimulus
    ##############################################################################
    # Create screen grid: each screen voxel has its eccentricity value in cm
    widthArray = np.arange(-halfXscreenPix, halfXscreenPix, step=1, dtype=int)
    heightArray = np.arange(-halfYscreenPix, halfYscreenPix, step=1, dtype=int)
    [x, y] = np.meshgrid(widthArray, heightArray)  # coordinates in pixels
    # what is eccen_screen and theta_screen ??
    eccen_screen = np.sqrt((x * cmPerPixel) ** 2 + (y * cmPerPixel) ** 2)
    theta_screen = np.rad2deg(2 * np.arctan(y / (x + np.sqrt(x ** 2 + y ** 2))))

    # Create screen grid corresponding to putative cortical distance
    e_cort = e_mm(np.sqrt((x * degsPerPixel) ** 2 + (y * degsPerPixel) ** 2))  # in mm of cortex

    # Recreate stim inducer
    sin_inducer = np.zeros((3, len(times), heightScreenPix, widthScreenPix))
    for ind_t, t in enumerate(times):
        # traveling wave (out)
        sin_inducer[0, ind_t] = A * np.sin(2 * np.pi * sfreq * e_cort - 2 * np.pi * tfreq * t + phi)

        # standing wave
        sin_inducer[1, ind_t] = A * np.sin(2 * np.pi * sfreq * e_cort + phi) * np.cos(2 * np.pi * tfreq * t)

        # traveling wave (in)
        sin_inducer[2, ind_t] = A * np.sin(2 * np.pi * sfreq * e_cort + 2 * np.pi * tfreq * t + phi)

    # Map stim values on voxel label (for lh and rh labels)
    wave_label = []
    for ind_h, hemi in enumerate(hemis):
        wave_label.append(np.zeros((3, len(eccen_label[ind_h]), len(times))))
        for ind_l, l in enumerate(eccen_label[ind_h]):
            # voxels above the maximum eccentricity of the screen stays zero
            if l <= np.max(eccen_screen):
                imin = np.argmin(np.abs(eccen_screen - eccen_label[ind_h][ind_l]))
                ind_stim = np.unravel_index(imin, np.shape(eccen_screen))
                wave_label[ind_h][0, ind_l] = sin_inducer[0, :, ind_stim[0], ind_stim[1]]  # trav
                wave_label[ind_h][1, ind_l] = sin_inducer[1, :, ind_stim[0], ind_stim[1]]  # stand
                wave_label[ind_h][2, ind_l] = sin_inducer[2, :, ind_stim[0], ind_stim[1]]  # trav in

    # Create wave stim for quadrant condition (session 1)
    wave_quad = deepcopy(wave_label[0])
    mask_quad = (angle_label[0] > 90 + 5) & (angle_label[0] < 180 - 5)
    mask = np.invert(mask_quad)  # right lower quadrant
    wave_quad[:, mask, :] = 0

    # Create wave stim for foveal condition (session 2)
    wave_fov = deepcopy(wave_label)
    wave_fov[0][:, eccen_label[0] > 5, :] = 0  # lh
    wave_fov[1][:, eccen_label[1] > 5, :] = 0  # rh

    # Create stim for half left stimulated V1 = cond, other half = other cond
    wave_halfHalf = np.zeros(np.shape(wave_label[0]))
    wave_halfHalf[0] = wave_label[0][1];
    wave_halfHalf[1] = wave_label[0][0]  # switch wave types
    wave_halfHalf[:, mask_quad, :] = wave_quad[:, mask_quad, :]

    # Loop across sessions to create simulation for each condition
    for ses in sessions:
        # Load forward model
        fname = op.join(preprocpath, subject, 'forwardmodel',
                        subject + '_' + ses + '_ico5-fwd.fif')  # check if can be replaced by sensors
        fwd = mne.read_forward_solution(fname)
        src = fwd['src']  # source space

        # Create empty stc
        labels = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir, parc='aparc.a2009s')  # aparc.DKTatlas
        n_labels = len(labels)
        signal = np.zeros((n_labels, len(times)))
        stc_gen = mne.simulation.simulate_stc(fwd['src'], labels, signal, times[0], tstep,
                                              value_fun=lambda x: x)  # labels or label_sel

        # Fill wave activity in stc
        # the Nth time series in stc.lh_data corresponds to the Nth value in stc.lh_vertno and src[0][‘vertno’] respectively, 
        # which in turn map the time series to a specific location on the surface, 
        # represented as the set of cartesian coordinates stc.lh_vertno[N] in src[0]['rr'].

        # FULLFIELD_TRAV, FULLFIELD_STAND (session1) or TRAV_OUT, STAND (session2)
        stc_angle = stc_gen.copy()  # only for left hemisphere
        stc_eccen = stc_gen.copy()
        stc_waves = []
        for ind_c, cond in enumerate(cond_waves + ['trav_in']):
            ## full field
            tmp = stc_gen.copy()
            for i in inds_label[0]:  # lh
                if i in stc_gen.lh_vertno:
                    i_stc = np.where(i == stc_gen.lh_vertno)[0][0]
                    tmp.lh_data[i_stc] = wave_label[0][ind_c, inds_label[0] == i]
                    stc_eccen.lh_data[i_stc] = eccen_label[0][inds_label[0] == i]
                    stc_angle.lh_data[i_stc] = angle_label[0][inds_label[0] == i]

            for i in inds_label[1]:  # rh
                if i in stc_gen.rh_vertno:
                    i_stc = np.where(i == stc_gen.rh_vertno)[0][0]
                    tmp.rh_data[i_stc] = wave_label[1][ind_c, inds_label[1] == i]
                    stc_eccen.rh_data[i_stc] = eccen_label[1][inds_label[1] == i]
                    stc_angle.rh_data[i_stc] = angle_label[1][inds_label[1] == i]
            stc_waves.append(tmp)

            # # add noise on each vertex
            # noise_lh = noise_amp * rng.randn(np.shape(tmp.lh_data)[0], np.shape(tmp.lh_data)[1])
            # tmp.lh_data[:] += noise_lh  
            # noise_rh = noise_amp * rng.randn(np.shape(tmp.rh_data)[0], np.shape(tmp.rh_data)[1])
            # tmp.rh_data[:] += noise_rh

            # save stc according to session
            if ses == 'session1':
                fname = op.join(simupath, '{}_simulated_source_{}_fullField'.format(subject, cond))
            elif (ses == 'session2') & (cond == 'trav'):
                fname = op.join(simupath, '{}_simulated_source_trav_out'.format(subject))
            elif (ses == 'session2') & (cond in ['stand', 'trav_in']):
                fname = op.join(simupath, '{}_simulated_source_{}'.format(subject, cond))
            tmp.save(fname)

            # QUAD_TRAV or QUAD_STAND (session1)
        if ses == 'session1':
            stc_quad = []
            for ind_c, cond in enumerate(cond_waves):
                ## quadrant
                tmp = stc_gen.copy()
                for i in inds_label[0]:  # lh
                    if i in stc_gen.lh_vertno:
                        i_stc = np.where(i == stc_gen.lh_vertno)[0][0]
                        tmp.lh_data[i_stc] = wave_quad[ind_c, inds_label[0] == i]

                stc_quad.append(tmp)

                # Save stc
                fname = op.join(simupath, '{}_simulated_source_{}_quadField'.format(subject, cond))
                tmp.save(fname)

        # FOV_OUT (session2)
        if ses == 'session2':
            stc_fov = []
            tmp = stc_gen.copy()
            for i in inds_label[0]:  # lh
                if i in stc_gen.lh_vertno:
                    i_stc = np.where(i == stc_gen.lh_vertno)[0][0]
                    tmp.lh_data[i_stc] = wave_fov[0][0, inds_label[0] == i]
            for i in inds_label[1]:  # rh
                if i in stc_gen.rh_vertno:
                    i_stc = np.where(i == stc_gen.rh_vertno)[0][0]
                    tmp.rh_data[i_stc] = wave_fov[1][0, inds_label[1] == i]

            # Save stc
            fname = op.join(simupath, '{}_simulated_source_fov_out'.format(subject))
            tmp.save(fname)

        if plot_stc:
            # plot eccentricity map
            mne.viz.set_3d_backend('pyvista')
            brain = stc_eccen.plot(subject=subject, subjects_dir=subjects_dir, colormap='rainbow', views='caudal')

            # plot angle map
            brain = stc_angle.plot(subject=subject, subjects_dir=subjects_dir, colormap='rainbow', views='caudal')

            # Plot stc (interactive)
            brain = stc_waves[0].plot(subject=subject, subjects_dir=subjects_dir, hemi='both')  # trav
            brain = stc_waves[1].plot(subject=subject, subjects_dir=subjects_dir, hemi='both')  # stand
            brain = stc_quad[0].plot(subject=subject, subjects_dir=subjects_dir, hemi='both')  # trav
            brain = stc_quad[1].plot(subject=subject, subjects_dir=subjects_dir, hemi='both')  # stand

            # save movie (do it manually to change the brain position)
            mne.viz.set_3d_backend('mayavi')
            for ind_c, cond in enumerate(cond_waves):
                brain = stc_waves[ind_c].plot(subject=subject, subjects_dir=subjects_dir, views='caudal',
                                              clim={'kind': 'value', 'pos_lims': [0, A / 2, A]}, colormap='mne',
                                              smoothing_steps=2)
                fname = op.join(resultspath, 'gif', subject + '_{}Wave_fullField_V1.gif'.format(cond))
                # brain.save_movie(fname, time_dilation=20, tmin=0, tmax=1, interpolation='linear', framerate = 10)

                brain = stc_quad[ind_c].plot(subject=subject, subjects_dir=subjects_dir, views='caudal',
                                             clim={'kind': 'value', 'pos_lims': [0, A / 2, A]}, colormap='mne',
                                             smoothing_steps=2)
                fname = op.join(resultspath, 'gif', subject + '_{}Wave_quadField_V1.gif'.format(cond))
                # brain.save_movie(fname, time_dilation=20, tmin=0, tmax=1, interpolation='linear', framerate = 10)

        ## Fill stc with half V1 stimulated
        if ses == 'session1':
            for ind_c, cond in enumerate(cond_waves):
                ## left V1 fully activated
                tmp = stc_gen.copy()
                for i in inds_label[0]:  # lh
                    if i in stc_gen.lh_vertno:
                        i_stc = np.where(i == stc_gen.lh_vertno)[0][0]
                        tmp.lh_data[i_stc] = wave_label[0][ind_c, inds_label[0] == i]

                # # add noise on each vertex
                # noise_lh = noise_amp * rng.randn(np.shape(tmp.lh_data)[0], np.shape(tmp.lh_data)[1])
                # tmp.lh_data[:] += noise_lh  
                # noise_rh = noise_amp * rng.randn(np.shape(tmp.rh_data)[0], np.shape(tmp.rh_data)[1])
                # tmp.rh_data[:] += noise_rh

                # Save left V1 
                fname = op.join(simupath, '{}_simulated_source_{}_leftV1_full'.format(subject, cond))
                tmp.save(fname)

                ## V1 with half cond, the other half other cond
                tmp = stc_gen.copy()
                for i in inds_label[0]:  # lh
                    if i in stc_gen.lh_vertno:
                        i_stc = np.where(i == stc_gen.lh_vertno)[0][0]
                        tmp.lh_data[i_stc] = wave_halfHalf[ind_c, inds_label[0] == i]

                # # add noise on each vertex
                # noise_rh = noise_amp * rng.randn(np.shape(tmp.rh_data)[0], np.shape(tmp.rh_data)[1])
                # tmp.rh_data[:] += noise_rh
                # noise_lh = noise_amp * rng.randn(np.shape(tmp.lh_data)[0], np.shape(tmp.lh_data)[1])
                # tmp.lh_data[:] += noise_lh

                # Save stc
                fname = op.join(simupath, '{}_simulated_source_leftV1_stimulated{}_otherHalf{}'.format(subject, cond,
                                                                                                       cond_waves[not (
                                                                                                           ind_c)]))
                tmp.save(fname)

                #### Sanity check: compute actual speed wave
        ###########################################################################

        # Compute the speed profile of the full label in the wave direction
        speed_label = [];
        global_speed = [];
        global_speed_label = []
        for ind_h, hemi in enumerate(hemis):
            # Compute real cortical speed
            vert_coord = src[ind_h]['rr'][
                inds_label[ind_h]]  # coordinates of each vertex of the label Freesurfer surface RAS in m
            vert_dis = scipy.spatial.distance.cdist(vert_coord, vert_coord,
                                                    metric='euclidean') * 1000  # distance for each pair of vertices in mm

            # select vertices corresponding to each angle
            speed_label.append(np.ones(np.shape(angle_label[ind_h])) * -1)
            global_speed.append(np.ones(181) * -1)  # global speed per angle (between fovea and most peripheric voxel)
            global_speed_label.append(np.ones(np.shape(angle_label[ind_h])) * -1)
            computed_speed = []
            inds = []
            i1_list = [];
            i2_list = []
            for a in np.arange(0, 181):
                # select vertices corresponding to each angle (phase)
                ind_horiz = np.where(np.round(angle_label[ind_h]) == a)[0]

                # calculate distance between each pair
                dist_horiz = np.zeros(len(ind_horiz) - 1)  # mm
                phase_diff = np.zeros(len(ind_horiz) - 1)  # rad
                for i in range(len(ind_horiz) - 1):
                    # calculate distance between each pair
                    dist_horiz[i] = vert_dis[ind_horiz[i], ind_horiz[i + 1]]
                    # calculate phase difference between the two temporal traveling signals
                    s1 = wave_label[ind_h][0, ind_horiz[i]];
                    s2 = wave_label[ind_h][0, ind_horiz[i + 1]]
                    phase_diff[i] = np.arccos(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))

                    if all(s1 == s2):  # no phase difference if identical signals
                        phase_diff[i] = 0

                # compute local speed profile (expected: 0.05 cycles/mm)
                speed_label[ind_h][ind_horiz[:-1]] = phase_diff / (2 * np.pi * dist_horiz)  # cycle/mm
                computed_speed.append(phase_diff / (2 * np.pi * dist_horiz))
                inds.append(ind_horiz)

                # compute global speed
                i1 = 0;
                i2 = -1
                s1 = wave_label[ind_h][0, ind_horiz[i1]];
                s2 = wave_label[ind_h][0, ind_horiz[i2]]
                while np.sum(s1) == 0:  # select closest voxel near the fovea with non zero signal
                    i1 = i1 + 1;
                    s1 = wave_label[ind_h][0, ind_horiz[i1]];
                while np.sum(s2) == 0:  # select closest voxel near the periph with non zero signal
                    i2 = i2 - 1;
                    s2 = wave_label[ind_h][0, ind_horiz[i2]]
                dist_global = vert_dis[ind_horiz[i1], ind_horiz[i2]]
                phase_global = np.arccos(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))
                global_speed[ind_h][a] = phase_global / (2 * np.pi * dist_global)  # cycle/mm
                i1_list.append(i1);
                i2_list.append(i2)

                # Fill global speed for visualization on stc
                global_speed_label[ind_h][ind_horiz[:-1]] = global_speed[ind_h][a]

            # check if there is missing index
            inds = np.concatenate(inds)
            miss_inds = list(set(inds) ^ set(np.arange(0, len(angle_label))))
            print('There are {:.1f} missing indices'.format(np.shape(miss_inds)[0]))

            # Mean/median/max speed
            computed_speed = np.concatenate(computed_speed)
            speed_mean = np.nanmean(np.array(computed_speed))
            print('Mean speed for the vertices (all angles) = {:.3f} cycles/mm (expected: 0.05)'.format(speed_mean))
            print('Median speed  = {:.3f} cycles/mm (expected: 0.05)'.format(np.nanmedian(np.array(computed_speed))))
            print('Range speed  = [{:.3f}, {:.3f}] cycles/mm'.format(np.nanmin(computed_speed),
                                                                     np.nanmax(computed_speed)))
            print('Global mean speed  = {:.3f} cycles/mm and median = {:.3f}'.format(np.mean(global_speed),
                                                                                     np.median(global_speed)))

        # Mean speed for both labels
        np.mean(global_speed, axis=1)
        np.mean(global_speed)

        speed_label_all[subject] = speed_label  # speed for each voxel of the label
        global_speed_all[subject] = global_speed  # speed for each angle and hemisphere

        # Plot stc with speed profile
        if plot_stc:
            stc_speed = stc_gen.copy()
            for i in inds_label[0]:  # lh
                if i in stc_gen.lh_vertno:
                    i_stc = np.where(i == stc_gen.lh_vertno)[0][0]
                    stc_speed.lh_data[i_stc] = speed_label[0][inds_label[0] == i]
            for i in inds_label[1]:  # rh
                if i in stc_gen.rh_vertno:
                    i_stc = np.where(i == stc_gen.rh_vertno)[0][0]
                    stc_speed.rh_data[i_stc] = speed_label[1][inds_label[1] == i]

            stc_speed.lh_data[np.isnan(stc_speed.lh_data)] = -1  # visualization does not work with nan
            stc_speed.rh_data[np.isnan(stc_speed.rh_data)] = -1
            mne.viz.set_3d_backend('pyvista')
            brain = stc_speed.plot(subject=subject, subjects_dir=subjects_dir, colormap='rainbow', hemi='both')

            # Plot stc with global speed profile (computed for a given visual field angle)
            stc_speed_global = stc_gen.copy()
            for i in inds_label[0]:  # lh
                if i in stc_gen.lh_vertno:
                    i_stc = np.where(i == stc_gen.lh_vertno)[0][0]
                    stc_speed_global.lh_data[i_stc] = global_speed_label[0][inds_label[0] == i]
            for i in inds_label[1]:  # rh
                if i in stc_gen.rh_vertno:
                    i_stc = np.where(i == stc_gen.rh_vertno)[0][0]
                    stc_speed_global.rh_data[i_stc] = global_speed_label[1][inds_label[1] == i]
            stc_speed_global.lh_data[np.isnan(stc_speed_global.lh_data)] = -1  # visualization does not work with nan
            stc_speed_global.rh_data[np.isnan(stc_speed_global.rh_data)] = -1
            brain = stc_speed_global.plot(subject=subject, subjects_dir=subjects_dir, colormap='rainbow', hemi='both')

# Group average for cortical speed
# speed_label_all = {}; global_speed_all = {}; global_speed_label_all = {}
speed_mean = np.zeros(len(speed_label_all.keys()))
speed_median = np.zeros(len(speed_label_all.keys()))
speed_range = np.zeros((len(speed_label_all.keys()), 2))
glob_speed = np.zeros(len(speed_label_all.keys()))
for s, subj in enumerate(speed_label_all.keys()):
    speed_bothHemis = np.concatenate((speed_label_all[subj][0], speed_label_all[subj][1]))
    speed_bothHemis[speed_bothHemis == -1] = np.nan
    speed_mean[s] = np.nanmean(speed_bothHemis)  # both hemis
    speed_median[s] = np.nanmedian(speed_bothHemis)
    speed_range[s] = [np.nanmin(speed_bothHemis), np.nanmax(speed_bothHemis)]
    glob_speed[s] = np.mean(global_speed_all[subj])
print('Mean speed for the vertices (all angles) = {:.3f} cycles/mm (expected: 0.05)'.format(np.mean(speed_mean)))
print('Median speed  = {:.3f} cycles/mm (expected: 0.05)'.format(np.mean(speed_median)))
print('Range speed  = [{:.3f}, {:.3f}] cycles/mm'.format(np.mean(speed_range, 0)[0], np.mean(speed_range, 0)[1]))
print('Global mean speed  = {:.3f} cycles/mm and median = {:.3f}'.format(np.mean(glob_speed), np.median(glob_speed)))
