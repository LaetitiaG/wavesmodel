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

##############################################################################
### Parameters ###
##############################################################################

from config import datapath, preprocpath, subjects, subjects_dir, resultspath, \
    simupath, cond_waves, col_cond, cond_space

# Subject
subjects = ['UNI5M7', 'H9W70R', '5P57E6','XZJ7KI', 'S1VW75', 'JRRWDT'] # subjects[0]

# Create report and directories if not existing
if not os.path.isdir(os.path.join(resultspath, 'encodingModel_retino_proj')):
    os.mkdir(os.path.join(resultspath, 'encodingModel_retino_proj'))
report_path = os.path.join(resultspath, 'encodingModel_retino_proj')
  
# topomap scale
ch_types = ['mag','grad','eeg']
scale = {'mag': 1200, 'grad': 400, 'eeg': 40}
chan_toplot = {'mag':['MEG2111'],'grad':['MEG2112'],'eeg': ['EEG070']}

##############################################################################
### Main ###
##############################################################################
  
for subject in subjects: 
    report = mne.Report(verbose=True)
    # Read info from raw data
    fname_dic = preproc.read_filesName(datapath + subject)
    sname = fname_dic['session1']['func'][0][:-4] + '_preproc_raw_tsss.fif'
    info = mne.io.read_info(os.path.join(preprocpath,fname_dic['session1']['subj'], fname_dic['session1']['ses'], sname))
    
    # Load forward model
    fname = op.join(preprocpath, subject, 'forwardmodel', subject + '_session1_ico5-fwd.fif')
    fwd = mne.read_forward_solution(fname)
    src = fwd['src'] # source space    
    
    for i_s, c_space  in enumerate(cond_space):
            
        evokeds= []
        for ind_c, cond in enumerate(cond_waves):
            # Read simulated stc created in encodingModel_retino_sources
            fname = op.join(simupath,subject + '_simulated_source_{}_{}Field'.format(cond, c_space))
            stc_gen = mne.read_source_estimate(fname)
            
            # Project into sensor space
            nave = np.inf # number of averaged epochs - controlled the amount of noise
            evoked_gen = mne.simulation.simulate_evoked(fwd, stc_gen, info, cov=None, nave=nave,
                                         random_state=42)
            f = evoked_gen.plot()
            report.add_figs_to_section(f, captions='evoked_gen before rescaling', section=cond + ' ' + c_space)
            
            
            # Save projections
            sname = subject + '_proj_' + cond +  'Wave_' + c_space + 'Field-ave.fif'
            evoked_gen.save(os.path.join(simupath,sname))
            evokeds.append(evoked_gen)
            
            # Plot evoked
            # mag and grad
            f = evoked_gen.plot_topo()
            report.add_figs_to_section(f, captions='topo meg ', section=cond + ' ' + c_space)
                
            # topo EEG
            tmp = evoked_gen.copy().pick_types(meg=False, eeg=True)
            f = tmp.plot_topo()
            report.add_figs_to_section(f, captions='topo eeg ', section=cond + ' ' + c_space)
            
            for chan in ch_types:    
                #f = evoked_gen.plot_topomap(times = [0.05, 0.1, 0.15, 0.20, 0.25], ch_type = chan, vmax = scale[chan], vmin = -scale[chan])
                f = evoked_gen.plot_topomap(times = [0.0, 0.60, 0.120], ch_type = chan, vmax = scale[chan], vmin = -scale[chan])
                report.add_figs_to_section(f, captions=chan + ' ', section=cond + ' ' + c_space)
                
        
        # time course for a chosen sensor
        times = evoked_gen.times
        f, ax = plt.subplots(figsize=(10,9))
        for ch,ch_type in enumerate(ch_types):
            ax = plt.subplot(3,1,ch+1)
            plt.plot(times, evokeds[0].copy().pick_channels(chan_toplot[ch_type]).data[0], col_cond[cond_waves[0]])
            plt.plot(times, evokeds[1].copy().pick_channels(chan_toplot[ch_type]).data[0], col_cond[cond_waves[1]])
            plt.hlines(y=0, xmin=-0.2, xmax=2, color='k', linestyles='dotted')
            plt.axvline(x=0, color='k', linestyle='dotted')
            plt.xlim((-0.2,2))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.ylabel(chan_toplot[ch_type][0] + '-' + c_space)
            if ch== 2: plt.legend(cond_waves, frameon=False); plt.xlabel('time (s)')
        report.add_figs_to_section(f, captions='timecourse at selected sensor', section='bothcond ' + c_space)        
    
    
    ### Do the same for left V1 whole or half/half
    for ind_c, cond in enumerate(cond_waves):
        leftV1 = ['leftV1_stimulated{}_otherHalf{}'.format(cond, cond_waves[not(ind_c)]), '{}_leftV1_full'.format(cond)]
        for condi in leftV1:
            # Read simulated stc created in encodingModel_retino_sources
            fname = op.join(simupath,subject + '_simulated_source_' + condi )
            stc_gen = mne.read_source_estimate(fname)
            
            # Project into sensor space
            nave = np.inf # number of averaged epochs - controlled the amount of noise
            evoked_gen = mne.simulation.simulate_evoked(fwd, stc_gen, info, cov=None, nave=nave,
                                         random_state=42)
            f = evoked_gen.plot()
            report.add_figs_to_section(f, captions='evoked_gen before rescaling', section=condi)
            
            
            # Save projections
            sname = subject + '_proj_' + condi + '-ave.fif'
            evoked_gen.save(os.path.join(simupath,sname))
            evokeds.append(evoked_gen)
            
            # Plot evoked
            # mag and grad
            f = evoked_gen.plot_topo()
            report.add_figs_to_section(f, captions='topo meg ', section=condi)
                
            # topo EEG
            tmp = evoked_gen.copy().pick_types(meg=False, eeg=True)
            f = tmp.plot_topo()
            report.add_figs_to_section(f, captions='topo eeg ', section=condi)
            
            for chan in ch_types:    
                f = evoked_gen.plot_topomap(times = [0.05, 0.1, 0.15, 0.20, 0.25], ch_type = chan, vmax = scale[chan])
                report.add_figs_to_section(f, captions=chan + ' ', section=condi)            
        
        
    # Save report
    report.save(os.path.join(report_path, 'report_' + subject +'.html'), overwrite=True)    
        
