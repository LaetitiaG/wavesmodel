##############################################################################
#### estimate_free_parameters.py
##############################################################################
# -*- coding: utf-8 -*-

"""
Estimate free parameters for each participant.

Created on Mon Jun 19 14:39:34 2023

@author: laeti
"""


from scipy.optimize import curve_fit, least_squares
import numpy as np

from toolbox.simulation import create_sim_from_entry
from toolbox.projection import project_wave
from toolbox.comparison import compare_meas_simu, create_RSA_matrices, from_meas_evoked_to_matrix
import time
from numba import jit
import os
import pickle
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import scipy.stats
from toolbox.entry import Entry

subject = 'OF4IP5'
session = 'session2'
comp = 'matched'
cond = 'full_out'

p = 0.05

epoch_type = "SSVEP" # SSVEP 5Hzfilt
ep_name = {'SSVEP': '-', '5Hzfilt': '_5Hzfilt-'}

# my path
wdir = 'C:/Users/laeti/Data/wave_model/'
sensorspath = wdir + 'data_MEEG/sensors/' 

###### FUNCTIONS ##############################################################

    
def func(parameter, entry, verbose = False):
    ''' Free parameter p = temporal frequency'''
    
    start_time = time.time()
    freq_temp = parameter
    freq_spatial = 0.05
    amplitude = 1e-08
    phase_offset = 1.5707963267948966
    
    entry.simulation_params = [freq_temp, freq_spatial, amplitude, phase_offset]
    sim = create_sim_from_entry(entry)
    stc = sim.generate_simulation()
    ev_proj = project_wave(entry, stc, verbose) 
    compare = compare_meas_simu(entry, ev_proj, verbose=verbose)
    
    # select sum of squared residuals for all channel types and the complex comparison
    SSR = compare[-1] # compare[-1][:,2]
    R = compare[5] # compare[5][:,2]
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return SSR, R  

def func_sF(parameter, entry, res_meas, ch_types, verbose = False):
    ''' Free parameter p = spatial frequency'''
    
    freq_temp = 5
    freq_spatial = parameter
    amplitude = 1e-08
    phase_offset = 1.5707963267948966
    
    entry.simulation_params = [freq_temp, freq_spatial, amplitude, phase_offset]
    sim = create_sim_from_entry(entry)
    stc = sim.generate_simulation()
    ev_proj = project_wave(entry, stc, verbose) 
    compare = compare_meas_simu(entry, ev_proj, ch_types = ch_types, meas = res_meas, verbose=verbose)
    
    # select sum of squared residuals for all channel types and the complex comparison
    SSR = compare[-1][:,2] # only complex data SSR
    
    return SSR

def fit_tempFreq(entry, subject, condition, parameters_to_test):
    '''
    

    Parameters
    ----------
    subject : str
        DESCRIPTION.
    condition : str
        'TRAV_OUT', 'TRAV_IN'

    Returns
    -------
    best_parameter : float
        DESCRIPTION.

    '''
        
    # Change condition
    entry.stim = condition
    if condition == 'TRAV_OUT':
        cond = 'trav_out'
    elif condition == 'TRAV_IN':
        cond = 'trav_in'
    
    # Change subject
    #entry.measured = wdir + 'data_MEEG/sensors/' + subject + '_' + cond +'-ave.fif' # TO PUT BACK
    #entry.measured = wdir + 'data_MEEG/simulation/' + subject + '_proj_V1_session2_' + cond +'-ave.fif' # test pure 5Hz
    #entry.measured = wdir + 'data_MEEG/simulation/' + subject + '_proj_V1_session2_5plus10Hz_' + cond +'-ave.fif' # pure 5 + 10Hz
    entry.measured = wdir + 'data_MEEG/simulation/' + subject + '_proj_V1_session2_5plus7Hz_' + cond +'-ave.fif' # pure 5 + 10Hz

    entry.freesurfer = wdir + 'data_MRI/preproc/freesurfer/' + subject
    entry.forward_model = wdir + 'data_MEEG/preproc/'+ subject+'/forwardmodel/' + subject + '_session2_ico5-fwd.fif'
    
    # Initialize
    ch_types = ['mag', 'grad', 'eeg']
    sumsq = np.zeros((len(ch_types), 3,len(parameters_to_test)))
    R_all = np.zeros((len(ch_types), 3,len(parameters_to_test)))
    best_parameter =np.zeros((len(ch_types), 3))
    
    # Iterate over the parameters and store sum of sq
    for p,parameter in enumerate(parameters_to_test):
        # Calculate the objective function for the current parameter
        res = func(parameter, entry)
        sumsq[:,:,p] = res[0]
        R_all[:,:,p] = res[1]
        #print('Parameter tested: {} Hz'.format(parameter))
        # Find the best-fit parameter for each channel type
    for c in range(len(ch_types)):
        best_parameter[c] = parameters_to_test[np.argmin(sumsq[c],1)]
    print('--Participant {} optimized--'.format(subject))
    

    # Save fitting parameters results
    fit_result = {"best_fit": best_parameter,
                  "SSR":  sumsq,
                  "R":  R_all,
                   "param_name": 'tempFreq',
                   "condition": condition,
                   "subject": subject
                   }
    fname = "{}_fit_param_tempFreq_session2_5plus7Hz_{}.pkl".format(subject,condition)
    a_file = open(os.path.join(wdir, 'data_MEEG', 'freeparam', fname), "wb")
    pickle.dump(fit_result, a_file)
    a_file.close() 
    
    return best_parameter, sumsq, R_all

############ for parameters where we know the parameters space (temporal frequency) ############

# test for one participant
start_time = time.time()
subject = 'JRRWDT'
condition='TRAV_OUT'
parameters_to_test = np.arange(2., 7., 3)
bestP = fit_tempFreq(entry, subject, condition, parameters_to_test)
print("--- %s seconds ---" % (time.time() - start_time))

# test with joblib
subjects_ses2 = ['EWTO6I', 'QNP39U', 'OF4IP5', 'XZJ7KI', '03TPZ5','UEGW36',\
            'S1VW75', 'QFLDFC', 'JRRWDT', 'Q95PQG', '90WCLR', 'TX0JPL',\
            'U2XPZV', '575ZC9', 'D1AQHG','8KYLY7', 'O3YE19', 'DOYJLH', 'NOT7EZ']
subjects_ses2 = ['EWTO6I', 'QNP39U']
    
condition = 'TRAV_OUT' # 'TRAV_OUT' or 'TRAV_IN'
parameters_to_test = np.arange(2., 15., 3) # freqs  np.arange(2., 50., 0.5)

# Create entry
entry = Entry(c_space='full')
width = 1920
height = 1080
distancefrom = 78
heightcm = 44.2
entry.screen_params = [width,height,distancefrom,heightcm]

# Loop across participants

res_allSubj = Parallel(n_jobs=4)(
    delayed(fit_tempFreq)(entry, subject, condition, parameters_to_test)
    for subject in subjects_ses2)

############ Simulated measures ############

# Create entry
entry = Entry(c_space='full')
width = 1920
height = 1080
distancefrom = 78
heightcm = 44.2
entry.screen_params = [width,height,distancefrom,heightcm]
subject = 'JRRWDT'
condition='TRAV_OUT'
parameters_to_test = np.arange(2., 35., 1)
bestP = fit_tempFreq(entry, subject, condition, parameters_to_test)

# Pure sinusoid + noise
freq_temp = 10
freq_spatial = 0.05
amplitude = 1e-08
phase_offset = 1.5707963267948966
verbose = True 
entry.stim = 'TRAV_OUT'
entry.simulation_params = [freq_temp, freq_spatial, amplitude, phase_offset]
entry.freesurfer = wdir + 'data_MRI/preproc/freesurfer/' + subject
entry.forward_model = wdir + 'data_MEEG/preproc/'+ subject+'/forwardmodel/' + subject + '_session2_ico5-fwd.fif'
entry.measured = wdir + 'data_MEEG/sensors/' + subject + '_trav_out-ave.fif'
sim = create_sim_from_entry(entry)
stc = sim.generate_simulation()
ev_proj10 = project_wave(entry, stc, verbose) 

freq_temp = 5
phase_offset = np.pi/4
entry.simulation_params = [freq_temp, freq_spatial, amplitude, phase_offset]
sim = create_sim_from_entry(entry)
stc = sim.generate_simulation()
ev_proj5 = project_wave(entry, stc, verbose)

ev_proj = ev_proj5.copy()
ev_proj.data = (ev_proj5.data + ev_proj10.data)/2

############ for parameters where we don't know the parameters space ############


ch_types = ['mag']
bounds = (0.01,1.5) # spatial frequency
p0 = 0.01 # initial parameter

# Matrix for measured data
res_meas = from_meas_evoked_to_matrix(entry, ch_types, verbose=verbose)
results = least_squares(func_sF, x0=p0, args=(entry, res_meas, ch_types), bounds = bounds, verbose=2)   

    

'''
############# plot 
mat_meas = compare[7]['mag'][2]
mat_simu = compare[8]['mag'][2] 
diff = mat_meas-mat_simu
np.sum(np.abs(diff**2))
np.sum((diff.real**2 + diff.imag**2)**2)
np.sum(diff.real**2 + diff.imag**2) # this is equal to np.abs(diff)


import matplotlib.pyplot as plt
plt.figure()
plt.subplot(211)
plt.plot(parameters_to_test, sumsq[0,0])
plt.ylabel('SSR - amp')
plt.subplot(212)
plt.plot(parameters_to_test, R_all[0,0])
plt.ylabel('R')
plt.xlabel('Temp freq')
plt.figure()
plt.subplot(211)
plt.plot(parameters_to_test, sumsq[0,1])
plt.ylabel('SSR - phase')
plt.subplot(212)
plt.plot(parameters_to_test, R_all[0,1])
plt.ylabel('R')
plt.xlabel('Temp freq')
plt.figure()
plt.subplot(211)
plt.plot(parameters_to_test, sumsq[0,2])
plt.ylabel('SSR - cplx')
plt.subplot(212)
plt.plot(parameters_to_test, R_all[0,2])
plt.ylabel('R')
plt.xlabel('Temp freq')



start_time = time.time()
wave_label = create_wave_stims(c_space, times, stim_inducer, eccen_screen, angle_label, eccen_label)
print("--- %s seconds ---" % (time.time() - start_time))


###### TESTS ##################################################################

from toolbox.simulation import Simulation, create_sim_from_entry
from toolbox.projection import project_wave
from toolbox.comparison import compare_meas_simu, create_RSA_matrices, compare_meas_simu_oneChType
from toolbox.entry import Entry
import time

# Use of scipy.optimize.curve_fit(f, xdata, ydata)
# f should take as input xdata (fixed parameters of the simu) and the parameters 
# to change, then give as output the complex matrice of the simulated data.
# ydata should be of the same shape.

entry_file = "C:\\Users\\laeti\\Data\\wave_model\\scripts_python\\WAVES\\test\\entry\\entry.ini"
entry_list = read_entry_config(entry_file)
entry = entry_list[0]





simulation_params = {'freq_temp':5, 'freq_spatial':0.05, 'amplitude':1e-08, 'phase_offset':1.5707963267948966}
screen_params = {'width':1920, 'height':1080, 'distanceFrom':78, 'heightCM':44.2}

entry_dic = {
    'wdir' : 'C://Users//laeti//Data//wave_model//',
    'measured' : wdir + 'data_MEEG//sensors//2XXX72_fullField_stand-ave.fif',
    'freesurfer' : wdir +  'data_MRI//preproc//freesurfer//2XXX72',
    'fwd_model' : wdir +  'data_MEEG//preproc//2XXX72//forwardmodel//2XXX72_session1_ico5-fwd.fif', 
    'stim' : 'TRAV_OUT',
    'c_space'  : 'full', 
    'simulation_params' : simulation_params, 
    'screen_params' : screen_params}

entry = Entry().load_entry(entry_dic)

simulation_config_section='0cfg',
screen_config_section='cfg1',

'''
