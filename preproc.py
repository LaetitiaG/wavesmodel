# -*- coding: utf-8 -*-
"""
Functions used for preprocessing (in preproc_main)
Created on Thu Dec 17 20:21:15 2020

@author: laeti
"""

import os
import numpy as np
import mne
from mne.preprocessing import find_bad_channels_maxwell
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

########################################
def read_filesName(path):
    """
    Read file names and path of fif files of a given subject. Return a dictionnary
    for each session with the following keys:
        - func: functional runs
        - empty: empty room recording
        - rest: resting-state

    Parameters
    ----------
    path : STR : path of the subject folder.

    Returns
    -------
    filenames : Dictionary with session1/session2... as keys, then 
    path/func/rest/empty as keys

    """

    filenames = {}
    sessions = os.listdir(path)
    for s, session in enumerate(sessions):
        filenames['session' + str(s+1)] = {}
        # Functional runs (wave inducer)
        filenames['session' + str(s+1)]['func'] = [
            f
            for f in os.listdir(os.path.join(path,session))
            if (f[:3]=='run') & (f[-3:]=='fif') # block instead of run for P01
        ]
        
        # Resting state
        filenames['session' + str(s+1)]['rest'] = [
            f
            for f in os.listdir(os.path.join(path,session))
            if (f[:4]=='rest') & (f[-3:]=='fif')
        ]
        
        # Empty room
        filenames['session' + str(s+1)]['empty'] = [
            f
            for f in os.listdir(os.path.join(path,session))
            if (f[:5]=='empty') & (f[-3:]=='fif')
        ]
        
        # Add path 6subject and session info for these files
        filenames['session' + str(s+1)]['path'] = os.path.join(path,session)
        filenames['session' + str(s+1)]['ses'] = os.path.split(filenames['session' + str(s+1)]['path'])[1]
        filenames['session' + str(s+1)]['subj'] = os.path.split(os.path.split(filenames['session' + str(s+1)]['path'])[0])[1]
        
    return filenames
    

######################################## 
def detect_MEGbadChannel(raw, fname, fine_cal_file, crosstalk_file, coord_frame,report):
    """
    Detect bad channels with automatic detection based on maxfiler. 
    Udpate the report with bad channels scores.
    Modify the raw.info['bads'] section in the raw files.

    Parameters
    ----------
    raw : Raw
        raw files without maxfilter, with empty bad channels info.
    fname : STR
        file name of the raw
    fine_cal_file : STR
        name of the fine calibration file for maxfilter.
    crosstalk_file : STR
        name of the crosstalk file for maxfilter.
    coord_frame : STR
        coordinate frame used in maxfilter: 'head' for data, 'meg' for empty room
    report : Report
        report file.

    Returns
    -------
    None. 

    """
    # automatic detection of bad channels with maxfilter
    raw_check = raw.copy()
    auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
        raw_check, cross_talk=crosstalk_file, calibration=fine_cal_file,
        coord_frame=coord_frame,return_scores=True, verbose=True)
    print(auto_noisy_chs)  
    print(auto_flat_chs)  
        
    # Loop across all channel types
    for ch_type in ['grad', 'mag']:
        ch_subset = auto_scores['ch_types'] == ch_type
        ch_names = auto_scores['ch_names'][ch_subset]
        scores = auto_scores['scores_noisy'][ch_subset]
        limits = auto_scores['limits_noisy'][ch_subset]
        bins = auto_scores['bins']  # The the windows that were evaluated.
        # We will label each segment by its start and stop time, with up to 3
        # digits before and 3 digits after the decimal place (1 ms precision).
        bin_labels = [f'{start:3.3f} – {stop:3.3f}'
                      for start, stop in bins]
        
        # We store the data in a Pandas DataFrame. The seaborn heatmap function
        # we will call below will then be able to automatically assign the correct
        # labels to all axes.
        data_to_plot = pd.DataFrame(data=scores,
                                    columns=pd.Index(bin_labels, name='Time (s)'),
                                    index=pd.Index(ch_names, name='Channel'))
        
        # First, plot the "raw" scores.
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f'Automated noisy channel detection: {ch_type}',
                     fontsize=16, fontweight='bold')
        sns.heatmap(data=data_to_plot, cmap='Reds', cbar_kws=dict(label='Score'),
                    ax=ax[0])
        [ax[0].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
            for x in range(1, len(bins))]
        ax[0].set_title('All Scores', fontweight='bold')
        
        # Now, adjust the color range to highlight segments that exceeded the limit.
        sns.heatmap(data=data_to_plot,
                    vmin=np.nanmin(limits),  # bads in input data have NaN limits
                    cmap='Reds', cbar_kws=dict(label='Score'), ax=ax[1])
        [ax[1].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
            for x in range(1, len(bins))]
        ax[1].set_title('Scores > Limit', fontweight='bold')
        
        # The figure title should not overlap with the subplots.
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        report.add_figs_to_section(fig, captions='scores_badCh_'+ch_type, section='1-tsss_bads', 
                                   comments='Auto noisy: '+str(auto_noisy_chs) 
                                   + 'auto flat: ' +str(auto_flat_chs))
        
        # mark bad channels
        raw.info['bads'] += auto_noisy_chs + auto_flat_chs
    
        
########################################     
def init_report(fname_dic):   
    '''
    Create a new report with a report path being path/log/subj/session.
    Non-existing directories are created.

    Parameters
    ----------
    fname_dic : File names as created in read_filesName

    Returns
    -------
    report : report file
    report_path : STR
        report path

    '''
    from config import preprocpath
    
    # Create report for preprocessing
    report_path = os.path.join(preprocpath,'log', fname_dic['subj'], fname_dic['ses'])
    report = mne.Report(verbose=True)

    # Create output dir if it doesn't exist
    if not os.path.isdir(os.path.join(preprocpath,fname_dic['subj'])):
        os.mkdir(os.path.join(preprocpath,fname_dic['subj']))
    if not os.path.isdir(os.path.join(preprocpath,fname_dic['subj'], fname_dic['ses'])):
        os.mkdir(os.path.join(preprocpath,fname_dic['subj'], fname_dic['ses']))     
    if not os.path.isdir(os.path.join(preprocpath,'log', fname_dic['subj'])):
        os.mkdir(os.path.join(preprocpath,'log', fname_dic['subj']))        
    if not os.path.isdir(os.path.join(preprocpath,'log', fname_dic['subj'], fname_dic['ses'])):
        os.mkdir(os.path.join(preprocpath,'log', fname_dic['subj'], fname_dic['ses']))
        
    return report, report_path
        

def choose_refRun(fname_dic):
    '''
    Select among functional runs the run whose head position is closest to the
    average head position. It will be used for head compensation done by maxfilter.

    Parameters
    ----------
    fname_dic : DIC
        Files name, as created in read_filenames.

    Returns
    -------
    TheFile : STR
        File name of the reference run.

    '''
    
    fnames = fname_dic['func']
    # retrieve head coordinates
    coord = np.zeros((len(fnames), 3))
    for i, fname in enumerate(fnames):
        raw = mne.io.read_raw_fif(os.path.join(fname_dic['path'],fname), allow_maxshield=True)
        coord[i] = raw.info['dev_head_t']['trans'][:3,3]
    
    # average across runs
    coord_av = np.mean(coord,0)
    
    # compute distance of each run to average
    distances= np.sqrt(np.sum((coord-coord_av)**2, axis=1))
    
    # pick the closest run to be the reference runs
    TheFile= np.array(fnames)[distances==np.min(distances)][0]
    print('The reference run used for head compensation is: ' + TheFile)
    
    return TheFile

######################################## 
def applytsss(fname_dic, fname, report, visu):
    """
    For a given fif file, mark bad channels and apply tsss.

    Parameters
    ----------
    fname_dic : DICT
        Files name, as created in read_filenames.
    fname : STR
        Name of the file to preprocess.
    report : REPORT
        Report file to fill.
    visu : BOOL
        True: allow interactive raw plot to be plotted for bad channels.
        False: do not plot interactive raw plot.

    Returns
    -------
    None.

    """
    
    from config import st_correlation, badC_MEG
    
         
    # Read calibration file and crosstalk compensation file needed for tsss
    cal_files = os.listdir(os.path.join(fname_dic['path'], 'sss_config'))
    fine_cal_file = os.path.join(fname_dic['path'], 'sss_config', [f for f in cal_files if 'sss_cal' in f][0])
    crosstalk_file  = os.path.join(fname_dic['path'], 'sss_config', [f for f in cal_files if 'ct_sparse' in f][0])
    
    # read files
    raw = mne.io.read_raw_fif(os.path.join(fname_dic['path'],fname), allow_maxshield=True)
    
    # raw.plot(duration=10, n_channels = 30) # interactive viewing
    if visu:
        raw.plot()
        print(fname + ' before sss')
        breakpoint() # when done, type continue() 

    
    # specify coordinate frame for maxfilter and reference run to align head position
    if "empty" in fname:
        coord_frame = 'meg'
        destination = None
    else:
        coord_frame = 'head'
        fname_ref = choose_refRun(fname_dic)
        destination = os.path.join(fname_dic['path'], fname_ref) # check raw.info['dev_head_t']
        
    # mark bad channels after automatic detection
    detect_MEGbadChannel(raw, fname, fine_cal_file, crosstalk_file, coord_frame, report)       
    
    # add bad channels manually detected
    raw.info['bads'] += badC_MEG[fname_dic['subj']][fname]        
    
    # apply tsss
    st_duration = len(raw)/1000 # in secs
    raw_sss = mne.preprocessing.maxwell_filter(
        raw, cross_talk=crosstalk_file, calibration=fine_cal_file, coord_frame=coord_frame,
        verbose=True, st_duration=st_duration, st_correlation=st_correlation, destination=destination)
    
    if visu:
        raw_sss.plot()
        print(fname + ' after sss')
        breakpoint() # when done, type continue() 
        
    # visualize before/after maxfilter
    fig1= raw.copy().pick(['meg']).plot(duration=2, butterfly=True)
    fig2= raw_sss.copy().pick(['meg']).plot(duration=2, butterfly=True)
    report.add_figs_to_section(fig1, captions='raw', section='1-tsss_bads')
    report.add_figs_to_section(fig2, captions='tsss', section='1-tsss_bads')
    
    return raw_sss, report


######################################## 
def interpolate_badEEGch(raw_sss, fname_dic, fname, report):
    '''
    Interpolate bad EEG channels after manual detection (visu in applytsss)
    Parameters
    ----------
    raw_sss : FIFF
        raw fif file.
    fname_dic : DICT
        Files name, as created in read_filenames.
    fname : STR
        Name of the file to preprocess.
    report : REPORT
        Report file to fill. 

    Returns
    -------
    None.

    '''
    
    from config import badC_EEG
    
    # Interpolate bad EEG channel 
    raw_sss.info['bads'] += badC_EEG[fname_dic['subj']][fname]    
    raw_sss.interpolate_bads()    
        
    # Plot interpolated channels
    fig1= raw_sss.copy().pick(badC_EEG[fname_dic['subj']][fname]).plot(duration=2, butterfly=True)
    report.add_figs_to_section(fig1, captions='interpolated channels', 
                               section='2-EEGbads', comments = str(badC_EEG[fname_dic['subj']][fname]))     


######################################## 
def clean_artefacts(raw_sss, fname_dic, fname, report, visu):
    
    from config import ICA_remove_inds
    
    # Visualize EOGs artefact
    for i, chEOG in enumerate(['BIO001', 'BIO002']):
        eog_evoked = mne.preprocessing.create_eog_epochs(raw_sss, ch_name=chEOG).average()
        eog_evoked.apply_baseline(baseline=(None, -0.2))
        fig1, fig2, fig3 = eog_evoked.plot_joint()
        
        report.add_figs_to_section(fig1, captions='EOG{} artefact- EEG'.format(i+1), section='3-Artefacts')
        report.add_figs_to_section(fig2, captions='EOG{} artefact- MAG'.format(i+1), section='3-Artefacts')
        report.add_figs_to_section(fig3, captions='EOG{} artefact- GRAD'.format(i+1), section='3-Artefacts')
    
    # Visualize ECG artefact
    ecg_evoked = mne.preprocessing.create_ecg_epochs(raw_sss, ch_name='BIO003').average()
    ecg_evoked.apply_baseline(baseline=(None, -0.2))
    fig1, fig2, fig3 = ecg_evoked.plot_joint()
    report.add_figs_to_section(fig1, captions='ECG artefact- EEG', section='3-Artefacts')
    report.add_figs_to_section(fig2, captions='ECG artefact- MAG', section='3-Artefacts')
    report.add_figs_to_section(fig3, captions='ECG artefact- GRAD', section='3-Artefacts')
    
    # Filter to avoid problems for ICA
    filt_raw = raw_sss.copy()
    filt_raw.load_data().filter(l_freq=1., h_freq=None)
    
    # Do the ICA decomposition for each sensor type
    for chs_type in ['mag', 'grad', 'eeg']:
        # Fit ICA component
        ica = mne.preprocessing.ICA(n_components=50, random_state=42)
        ica.fit(filt_raw, picks = chs_type)
   
        # Visualize
        if visu:
            ica.plot_sources(raw_sss)
            ica.plot_components(inst=raw_sss) 
            print(chs_type)
            breakpoint() # then type continue()
        
        figs = ica.plot_components(inst=raw_sss)        
        for i,fig in enumerate(figs):
            report.add_figs_to_section(fig, captions='ICA components {}- {}'.format(i+1, chs_type), section='3-Artefacts')
        
        ## Detect automatically ICA corresponding to EOG
        eog_indices = []
        for i, chEOG in enumerate(['BIO001', 'BIO002']):
            eog_ind, eog_scores = ica.find_bads_eog(raw_sss,  ch_name = chEOG, measure = 'correlation', threshold = 0.25) # find which ICs match the EOG pattern
            eog_indices += eog_ind
    
            fig1 = ica.plot_scores(eog_scores) # barplot of ICA component "EOG match" scores
            report.add_figs_to_section(fig1, captions='ICA_toremove_scores - EOG{}- {}'.format(i+1, chs_type), section='3-Artefacts')
        
        eog_indices = list(np.unique(eog_indices))
        if eog_indices:
            ica.exclude = eog_indices
        
            figs = ica.plot_properties(raw_sss, picks=eog_indices) # plot diagnostics
            for j,fig in enumerate(figs):
                report.add_figs_to_section(fig, captions='ICA{}_toremove_prop - EOG- {}'.format(j+1, chs_type), section='3-Artefacts')    
        
            # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
            fig1 = ica.plot_sources(eog_evoked)
            report.add_figs_to_section(fig1, captions='ICA_toremove_sources - EOG- {}'.format(chs_type), section='3-Artefacts')   
            
            # Plot signal 
            fig1 = ica.plot_overlay(raw_sss, exclude=ica.exclude, picks='mag')
            report.add_figs_to_section(fig1, captions='ICA_before_after - EOG- {}'.format(chs_type), 
                                       comments = 'ICA excluded: ' + str(ica.exclude), section='3-Artefacts') 
            
            # Check that automatically detected ICA are written in the config file
            if not(set(eog_indices).issubset(set(ICA_remove_inds[fname_dic['subj']][fname][chs_type]))):
                print('EOG ICA are not all included in the config file!!')
                print(chs_type, eog_indices)
                print('manual ICA: ', ICA_remove_inds[fname_dic['subj']][fname][chs_type])
                #breakpoint() # then type continue()
                
        else:
            print('No EOG artefact detected in ' + chs_type)    
    
        ## Detect automatically ICA corresponding to ECG
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw_sss,ch_name= 'BIO003', method='correlation',
                                                    threshold='auto') # find which ICs match the ECG pattern       
        if ecg_indices:
            ica.exclude = ecg_indices
            
            # barplot of ICA component "ECG match" scores
            fig1 = ica.plot_scores(ecg_scores)
            report.add_figs_to_section(fig1, captions='ICA_toremove_scores - ECG- {}'.format(chs_type), section='3-Artefacts')
            
            # plot diagnostics
            figs = ica.plot_properties(raw_sss, picks=ecg_indices)
            for j,fig in enumerate(figs):
                report.add_figs_to_section(fig, captions='ICA{}_toremove_prop - ECG- {}'.format(j+1, chs_type), section='3-Artefacts')   
                
            # plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
            fig1 = ica.plot_sources(ecg_evoked)
            report.add_figs_to_section(fig1, captions='ICA_toremove_sources - ECG- {}'.format(chs_type), section='3-Artefacts') 
        
            # Plot signal with/without ICa
            fig1 = ica.plot_overlay(raw_sss, exclude=ecg_indices, picks='mag')
            report.add_figs_to_section(fig1, captions='ICA_before_after - ECG- {}'.format(chs_type),
                                       comments = 'ICA excluded: ' + str(ecg_indices), section='3-Artefacts') 
            
            # Check that automatically detected ICA are written in the config file
            if not(set(ecg_indices).issubset(set(ICA_remove_inds[fname_dic['subj']][fname][chs_type]))):
                ica.plot_components(inst=raw_sss) 
                print('ECG ICA are not all included in the config file!!')
                print(fname_dic['subj'], fname, chs_type, ecg_indices)
                print('manual ICA: ', ICA_remove_inds[fname_dic['subj']][fname][chs_type])
                #breakpoint() # then type continue()
                
        else:
            print('No ECG artefact detected in ' + chs_type)
        
        # Print automatically detected components
        print('Automatically detected EOG compoents: {}'.format(eog_indices))
        print('Automatically detected ECG compoents: {}'.format(ecg_indices))
        print('Please add them manually to the config file to remove them!')
        
        # Remove manually detected components
        ica.exclude = ICA_remove_inds[fname_dic['subj']][fname][chs_type]
        
        # Plot all components
        if ICA_remove_inds[fname_dic['subj']][fname][chs_type]:
            fig1 = ica.plot_components(inst=raw_sss, picks = ICA_remove_inds[fname_dic['subj']][fname][chs_type])
            report.add_figs_to_section(fig1, captions='manual_det_ICA- {}'.format(chs_type),
                                       comments = 'ICA excluded: ' + str(ICA_remove_inds[fname_dic['subj']][fname][chs_type]), section='3b-Excluded ICA') 
            
        # Plot all ICA components applied to data
        fig1 = ica.plot_sources(raw_sss, picks=list(np.unique(eog_indices + ecg_indices)))
        report.add_figs_to_section(fig1, captions='allICA_to remove - {}'.format(chs_type),
                                   comments = 'ICA excluded: ' + str(eog_indices + ecg_indices), section='3b-Excluded ICA')
         
        # Apply ICA correction to data
        ica.apply(raw_sss)
        
    # plot
    if visu:
        raw_sss.plot()
        print(fname + ' after cleaning artefact')
        breakpoint() # when done, type continue()
            
######################################## 
def reref_EEG(raw_sss, fname_dic, fname, report):
    '''
    Re-reference the EEG electrode to the average (use a projector)

    Parameters
    ----------
    raw_sss : FIFF
        raw fif file.
    fname_dic : DICT
        Files name, as created in read_filenames.
    fname : STR
        Name of the file to preprocess.
    report : REPORT
        Report file to fill.

    Returns
    -------
    None.

    '''
      
    # average-reference-as-projection (recommended for source estimates)
    # recorded: mastoid 1 connected to mastoid 2 is the recording reference
    raw_sss.set_eeg_reference('average', projection=True)
    print(raw_sss.info['projs'])
    
    # add report
    fig1= raw_sss.copy().pick(['eeg']).plot(proj=False)
    fig2= raw_sss.copy().pick(['eeg']).plot(proj=True)
    report.add_figs_to_section(fig1, captions='before reref', section='4-EEG_reref')
    report.add_figs_to_section(fig2, captions='after average reref', section='4-EEG_reref')     