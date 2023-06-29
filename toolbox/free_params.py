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

from toolbox import configIO
from toolbox.simulation import generate_simulation
from toolbox.projection import project_wave
from toolbox.comparison import compare_meas_simu, create_RSA_matrices, compare_meas_simu_oneChType
import time
from numba import jit
import os
import pickle
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import scipy.stats
from toolbox.configIO import read_entry_config



subject = 'OF4IP5'
session = 'session2'
comp = 'matched'
cond = 'full_out'

p = 0.05

epoch_type = "SSVEP" # SSVEP 5Hzfilt
ep_name = {'SSVEP': '-', '5Hzfilt': '_5Hzfilt-'}

# my path
wdir = 'C:\\Users\\laeti\\Data\\wave_model\\'
sensorspath = wdir + 'data_MEEG\\sensors\\' 

###### FUNCTIONS ##############################################################

    
def func(p, entry, verbose = False):
    ''' Free parameter p = temporal frequency'''
    
    start_time = time.time()
    freq_temp = p
    freq_spacial = 0.05
    amplitude = 1e-08
    phase_offset = 1.5707963267948966
    
    entry.simulation_params = [freq_temp, freq_spacial, amplitude, phase_offset]
    sim = create_sim_from_entry(entry)
    stc = sim.generate_simulation()
    proj = project_wave(entry, stc, verbose) 
    compare = compare_meas_simu(entry, proj, verbose)
    
    # select sum of squared residuals for all channel types and the complex comparison
    SSR = compare[-1] # compare[-1][:,2]
    R = compare[5] # compare[5][:,2]
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return SSR, R  


def fit_tempFreq(subject, condition):
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
    start_time1 = time.time()
    wdir = "C:\\Users\\laeti\\Data\\wave_model\\"
    entry_file = wdir + "scripts_python\\WAVES\\test\\entry\\entry.ini"
    entry_list = configIO.read_entry_config(entry_file)
    entry = entry_list[0]
    
    # # Change condition
    # entry.stim = condition
    # if condition == 'TRAV_OUT':
    #     cond = 'trav_out'
    # elif condition == 'TRAV_IN':
    #     cond = 'trav_in'
    
    # # Change subject
    # entry.measured = wdir + 'data_MEEG\\sensors\\' + subject + '_' + cond +'-ave.fif'
    # entry.freesurfer = wdir + 'data_MRI\\preproc\\freesurfer\\' + subject
    # entry.fwd_model = wdir + 'data_MEEG\\preproc\\'+ subject+'\\forwardmodel\\' + subject + '_session2_ico5-fwd.fif'
    
    # Initialize
    ch_types = ['mag', 'grad', 'eeg']
    parameters_to_test = np.arange(2., 30., 1) # freqs  np.arange(2., 50., 0.5)
    sumsq = np.zeros((len(ch_types), 3,len(parameters_to_test)))
    R_all = np.zeros((len(ch_types), 3,len(parameters_to_test)))
    best_parameter = [None,None,None]
    
    # Iterate over the parameters and store sum of sq
    for p,parameter in enumerate(parameters_to_test):
        # Calculate the objective function for the current parameter
        res = func(parameter, entry)
        sumsq[:,:,p] = res[0]
        R_all[:,:,p] = res[1]
        print('Parameter tested: {} Hz'.format(parameter))
        # Find the best-fit parameter for each channel type
    for c in range(len(ch_types)):
        best_parameter[c] = parameters_to_test[np.argmin((sumsq[c]))]
    print("--- %s seconds ---" % (time.time() - start_time1))
   
    
    return best_parameter

############ for parameters where we know the parameters space (temporal frequency) ############

# test for one participant
start_time = time.time()
subject = 'OF4IP5'
condition='TRAV_OUT'
bestP = fit_tempFreq(subject, condition)
print("--- %s seconds ---" % (time.time() - start_time))

# test with joblib
subjects_ses2 = ['EWTO6I', 'QNP39U', 'OF4IP5', 'XZJ7KI', '03TPZ5','UEGW36',\
            'S1VW75', 'QFLDFC', 'JRRWDT', 'Q95PQG', '90WCLR', 'TX0JPL',\
            'U2XPZV', '575ZC9', 'D1AQHG','8KYLY7', 'O3YE19', 'DOYJLH', 'NOT7EZ']
    
condition = 'TRAV_OUT' # 'TRAV_OUT' or 'TRAV_IN'

# Loop across participants
best_param_allSubj = Parallel(n_jobs=2)(
    delayed(fit_tempFreq)(subject, condition)
    for subject in subjects_ses2)

# Save fitting parameters results
fit_result = {"best_fit": best_param_allSubj,
               "param_name": 'tempFreq',
               "condition": condition
               }
fname = "fit_param_tempFreq_session2_{}.pkl".format(condition)
a_file = open(os.path.join(wdir, 'results', fname), "wb")
pickle.dump(fit_result, a_file)
a_file.close()         

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



'''
start_time = time.time()
wave_label = create_wave_stims(c_space, times, stim_inducer, eccen_screen, angle_label, eccen_label)
print("--- %s seconds ---" % (time.time() - start_time))

############ for parameters where we don't know the parameters space ############
entry_file = "C:\\Users\\laeti\\Data\\wave_model\\scripts_python\\WAVES\\test\\entry\\entry.ini"
entry_list = read_entry_config(entry_file)
entry = entry_list[0]

ch_type= 'mag'
bounds = (2.,49.5) # temporal frequency
p0 = 2 # initial parameter
results = least_squares(func, x0=p0, args=(entry, ch_type), bounds = bounds, verbose=2)   

    
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





simulation_params = {'freq_temp':5, 'freq_spacial':0.05, 'amplitude':1e-08, 'phase_offset':1.5707963267948966}
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
