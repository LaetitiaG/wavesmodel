# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:54:11 2023

@author: laeti
"""


from toolbox.simulation import create_sim_from_entry
from toolbox.projection import project_wave
from toolbox.comparison import compare_meas_simu
from toolbox.comparison import compare_meas_simu_V2
from toolbox.entry import Entry

subject = 'QNP39U'
session = 'session2'

# my path
wdir = "D:/Data/wave_model/" 
sensorspath = wdir + 'data_MEEG/sensors/' 


# Minimal example

entry = Entry(c_space='full')
width = 1920
height = 1080
distancefrom = 78
heightcm = 44.2
entry.screen_params = [width,height,distancefrom,heightcm]
freq_temp = 5
freq_spatial = 0.05
amplitude = 1e-08
phase_offset = 1.5707963267948966
decay = 0
e0 = 5
verbose = True 
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
entry.freesurfer = wdir + 'data_MRI/preproc/freesurfer/' + subject
entry.forward_model = wdir + 'data_MEEG/preproc/'+ subject+'/forwardmodel/' + subject + '_session2_ico5-fwd.fif'
entry.measured = wdir + 'data_MEEG/sensors/' + subject + '_' + condition_meas +'-ave.fif'
sim = create_sim_from_entry(entry)
stc = sim.generate_simulation()
ev_proj = project_wave(entry, stc, verbose)

compare = compare_meas_simu(entry, ev_proj, SNR_threshold = 1)
compare = compare_meas_simu_V2(entry, ev_proj)
compare[7]
        
