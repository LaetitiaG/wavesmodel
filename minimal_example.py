# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:54:11 2023
This script simulates traveling waves in V1, project the activity onto MEG and EEG sensors, 
and compare the predicted activity in sensors to empirical data.


@author: laeti
"""

from toolbox.simulation import create_sim_from_entry
from toolbox.projection import project_wave
from toolbox.comparison import compare_meas_simu
from toolbox.comparison import compare_meas_simu_concat
from toolbox.entry import Entry


# My path **TO CHANGE ACCORDINGLY**
wdir = "D:/Data/wave_model/scripts_python/3_example_dataset/" 
subject = 'EWTO6I'

## Specify the parameters of the model you want to test within an entry object 
# (see help for Entry object to have the details of required parameters)
entry = Entry()
# Screen parameters
width = 1920 # pixels
height = 1080 # pixels
distancefrom = 78 # cm
heightcm = 44.2 # cm
entry.screen_params = [width, height, distancefrom, heightcm]

# Waves parameters
freq_temp = 5 # Hz
freq_spatial = 0.05 # cycles per mm of cortex
amplitude = 1e-08 # A.m
phase_offset = 1.5708 # rad
decay = 0 # between 0 (no decay) to 1 (full decay)
e0 = 5 # degree of visual angle (DVA)

condition_meas = 'trav_out' 
condition_simu = 'trav_out' # 'trav_out' 'trav_in' 'stand' 'fov_out'
if condition_simu == 'trav_out' :
    entry.stim = 'TRAV_OUT'
    entry.c_space = 'full'
elif condition_simu == 'trav_in':
    entry.stim = 'TRAV_IN'
    entry.c_space = 'full'
elif condition_simu == 'stand' :
    entry.stim = 'STANDING'
    entry.c_space = 'full'    
elif condition_simu == 'fov_out':
    entry.stim = 'TRAV_OUT'
    entry.c_space = 'fov'  
entry.simulation_params = [freq_temp, freq_spatial, amplitude, phase_offset, decay, e0]

# Paths parameters
entry.freesurfer = wdir + 'data_MRI/preproc/freesurfer/' + subject
entry.forward_model = wdir + 'data_MEEG/preproc/'+ subject+'/forwardmodel/' + subject + '_ico5-fwd.fif'
entry.measured = wdir + 'data_MEEG/sensors/' + subject + '_' + condition_meas +'-ave.fif'

## Launch modeling
sim = create_sim_from_entry(entry)
stc = sim.generate_simulation() # simulate sources
ev_proj = project_wave(entry, stc) # predicted sensors activity

## Compare predicted and measured data
compare = compare_meas_simu(entry, ev_proj)
R = compare[7] # correlation coefficient between predicted and measured data

        
