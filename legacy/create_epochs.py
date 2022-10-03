##############################################################################
#### create_epochs.py
##############################################################################
# -*- coding: utf-8 -*-
"""
From preprocessed raw data, concatenate runs filter, decimate and create 
epochs and evoked locked on the start of a trial. Create epochs/evoked for 
each condition (quadrant/full and standing/traveling).


@author: lgrabot
"""

import os
import preproc
import mne
import numpy as np
from datetime import datetime

##############################################################################
### Parameters ###
##############################################################################

from config import datapath, preprocpath, preprocpath_EEG, sensorspath,sensorspath_EEG, resultspath, \
                    subjects, subjects_INCC, l_freq, h_freq, tmin_ep, tmax_ep, \
                    decim, reject,cond_space,cond_waves, trig_list, delay_trig, Nruns

# Choose which epochs type you want to create
# SSVEP: 1-45Hz, -1 to 2.5s, decim = 5
# 5Hzfilt: 4-6Hz, -1 to 2.5s, decim = 5
epoch_type = "SSVEP" # SSVEP 5Hzfilt
ep_name = {'SSVEP': '-', '5Hzfilt': '_5Hzfilt-'}
acquis = 'ICM' # 'ICM': MEEG at ICM or 'INCC':EEG at INCC   

if acquis == 'ICM':
    subjs = subjects
elif  acquis == 'INCC':
    subjs = subjects_INCC
  

##############################################################################
### Main ###
##############################################################################

n_trials_ses1 = np.zeros((len(subjs), len(cond_space), len(cond_waves)))
n_trials_ses2 = np.zeros((len(subjs), 4))
    
for s in range(len(subjs)):
    # Create report
    now = datetime.now()
    report_path = os.path.join(resultspath,'create_epochs', now.strftime("%Y%m%d_%H%M%S"))
    if not os.path.isdir(report_path):
            os.mkdir(report_path)
    report = mne.Report(verbose=True)
        
    # Read filenames of raw data
    if acquis == 'ICM':
        filenames = preproc.read_filesName(datapath + subjs[s])
        #preprocpath_sub = os.path.join(preprocpath,filenames['session1']['subj'], filenames['session1']['ses'])
        #fnames = filenames['session1']['func']
        #fnames = [fname[:-4] + '_preproc_raw_tsss.fif' for fname in fnames]
        #sensorspath = sensorspath
    elif  acquis == 'INCC':
        fnames = []
        for iR in np.arange(Nruns[subjs[s]]):
            fnames.append('LG_'+  subjs[s]+ "_B"+ str(iR+1) + '_preproc_raw.fif')
        preprocpath_sub = os.path.join(preprocpath_EEG, subjs[s])
        sensorspath = sensorspath_EEG
    
    # Loop across session
    for ses, val in filenames.items():
        fnames = filenames[ses]['func']
        preprocpath_sub = os.path.join(preprocpath,filenames[ses]['subj'], filenames[ses]['ses'])
        fnames = [fname[:-4] + '_preproc_raw_tsss.fif' for fname in fnames]
        sensorspath = sensorspath    
        
        # Concatenate runs
        raws=[]
        for fname in fnames:
            raws.append(mne.io.read_raw_fif(os.path.join(preprocpath_sub, fname)))
            
        raw = mne.io.concatenate_raws(raws)
            
        # low-pass filtering the raw data
        raw.load_data()
        raw.filter(l_freq=l_freq[epoch_type], h_freq=h_freq[epoch_type])
        
        # Find events
        if acquis == 'ICM':
            events = mne.find_events(raw, stim_channel='STI101', min_duration = 2/raw.info['sfreq'])
            
            # Correct events timing with the measured delay between trigger and photodiode signal (measure_timing_delays.py)
            events = mne.event.shift_time_events(events, [11,12,21,22],delay_trig, sfreq = raw.info['sfreq'])           
        elif  acquis == 'INCC':
            custom_mapping = {'Stimulus/S 11': 11, 'Stimulus/S 12': 12, 'Stimulus/S 21': 21, 'Stimulus/S 22': 22}
            events, ev_id = mne.events_from_annotations(raw, event_id=custom_mapping)
        
        # Define conditions/triggers
        triggers = trig_list[ses]  

        # Create epochs for each condition
        if ses=='session1':
            for cs, c_space in enumerate(cond_space):
                for cw,c_waves in enumerate(cond_waves):                                
                    # Reject epoch with too high amplitude and decimate (5: 1000 to 200Hz)
                    epochs = mne.Epochs(raw, events, event_id= triggers[c_space][c_waves], 
                                        tmin=tmin_ep, tmax=tmax_ep, decim=decim, reject=reject[acquis])
                    
                    # Reject epochs with too high amplitude (count final nber of epochs)
                    epochs.drop_bad()
                    n_trials_ses1[s,cs,cw] = len(epochs)
                    
                    # Save epochs
                    fname = '{}_{}Field_{}{}epo.fif'.format(subjs[s], c_space, c_waves, ep_name[epoch_type])
                    epochs.save(os.path.join(sensorspath, fname), overwrite=True)
                    
                    # Diagnostic plot
                    fig1 = epochs.plot(butterfly =True)
                    report.add_figs_to_section(fig1, captions='Epochs', section= '{}Field_{}'.format(c_space,c_waves))
                    
                    fig1 = epochs.plot_psd(fmax=40)
                    report.add_figs_to_section(fig1, captions='PSD', section='{}Field_{}'.format(c_space,c_waves))
                    
                    # Average and save evoked
                    evoked = epochs.average()
                    fname = '{}_{}Field_{}{}ave.fif'.format(subjs[s], c_space, c_waves, ep_name[epoch_type])
                    evoked.save(os.path.join(sensorspath, fname))     
                    
        elif ses=='session2':
            for cond, trig in triggers.items():            
                # Reject epoch with too high amplitude and decimate (5: 1000 to 200Hz)
                epochs = mne.Epochs(raw, events, event_id= trig, 
                                    tmin=tmin_ep, tmax=tmax_ep, decim=decim, reject=reject[acquis])
                
                # Reject epochs with too high amplitude (count final nber of epochs)
                epochs.drop_bad()
                c = list(triggers).index(cond)
                n_trials_ses2[s,c] = len(epochs)
                
                # Save epochs
                fname = '{}_{}{}epo.fif'.format(subjs[s], cond, ep_name[epoch_type])
                epochs.save(os.path.join(sensorspath, fname), overwrite=True)
                
                # Diagnostic plot
                fig1 = epochs.plot(butterfly =True)
                report.add_figs_to_section(fig1, captions='Epochs', section= cond)
                
                fig1 = epochs.plot_psd(fmax=40)
                report.add_figs_to_section(fig1, captions='PSD', section=cond)
                
                # Average and save evoked
                evoked = epochs.average()
                fname = '{}_{}{}ave.fif'.format(subjs[s], cond, ep_name[epoch_type])
                evoked.save(os.path.join(sensorspath, fname))                     
            
        # Create artificial epochs on resting state for MEG
        if acquis == 'ICM':
            fnames_rs = filenames[ses]['rest'] 
            for fname in fnames_rs:
                # read raw resting state
                sname = fname[:-4] + '_preproc_raw_tsss.fif'
                raw = mne.io.read_raw_fif(os.path.join(preprocpath_sub, sname))
                
                # low-pass filtering the raw data
                raw.load_data()
                raw.filter(l_freq=l_freq[epoch_type], h_freq=h_freq[epoch_type])
                
                # create fake events, each event is separated by 2secs
                events = mne.make_fixed_length_events(raw, duration =2) 
                
                # Create epochs
                epochs = mne.Epochs(raw, events, event_id= 1, 
                                        tmin=-1.5, tmax=+1.5, decim=decim, reject=reject[acquis])
                epochs.drop_bad()
            
                # Save epochs
                sname = '{}_{}_{}{}epo.fif'.format(subjs[s], fname[:-4], ses, ep_name[epoch_type])
                epochs.save(os.path.join(sensorspath, sname), overwrite=True)
        
    # Save report
    report.save(os.path.join(report_path, 'report_' + subjs[s] +'.html'), overwrite=True)
    
    
# Cout nber of epochs per condition
n_trials_tot = np.zeros((len(subjects), len(cond_space), len(cond_waves)))
for s,subj in enumerate(subjects):
    for cs, c_space in enumerate(cond_space):
        for cw,c_waves in enumerate(cond_waves):
            fname = '{}_{}Field_{}{}epo.fif'.format(subj, c_space, c_waves, ep_name[epoch_type])
            epochs = mne.read_epochs(os.path.join(sensorspath, fname))
            epochs.drop_bad()
            n_trials_tot[s,cs,cw] = len(epochs)