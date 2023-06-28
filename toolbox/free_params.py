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
import matplotlib.pyplot as plt
import scipy.stats

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


    
def func(p, entry, ch_type):
    ''' Free parameter p = temporal frequency'''

    start_time = time.time()
    freq_temp = p
    freq_spacial = 0.05
    amplitude = 1e-08
    phase_offset = 1.5707963267948966
    entry.set_simulation_params([freq_temp, freq_spacial, amplitude, phase_offset])
       
    stc = generate_simulation(entry)
    proj = project_wave(entry, stc) 
    compare = compare_meas_simu_oneChType(entry, proj, ch_type)
    
    # select residuals for one channel type and the complex comparison
    residuals = compare[-1][2]
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return residuals 


def cost_fun(p, entry, ch_type):
    residuals = func(p, entry, ch_type)
    
    return np.sum(residuals**2)


############ for parameters where we know the parameters space (temporal frequency) ############
entry_file = "C:\\Users\\laeti\\Data\\wave_model\\scripts_python\\WAVES\\test\\entry\\entry.ini"
entry_list = configIO.read_entry_config(entry_file)
entry = entry_list[0]

parameters_to_test = np.arange(2., 50., 0.5) # freqs
best_cost_value = np.inf
best_parameter = None

# Iterate over the parameters and find the best-fit parameter
for parameter in parameters_to_test:
    # Calculate the objective function for the current parameter
    cost_value = cost_fun(parameter, entry, ch_type)
    
    # Update the best parameter if the current objective function value is better
    if cost_value < best_cost_value:
        best_cost_value = cost_value
        best_parameter = parameter
    print('Tested parameter: {} - best parameter: {}'.format(parameter,best_parameter))


############ for parameters where we don't know the parameters space ############
entry_file = "C:\\Users\\laeti\\Data\\wave_model\\scripts_python\\WAVES\\test\\entry\\entry.ini"
entry_list = configIO.read_entry_config(entry_file)
entry = entry_list[0]

ch_type= 'mag'
bounds = (2.,49.5) # temporal frequency
p0 = 2 # initial parameter
results = least_squares(func, x0=p0, args=(entry, ch_type), bounds = bounds, verbose=2)   

    
###### TESTS ##################################################################

from toolbox import configIO
from toolbox.simulation import generate_simulation
from toolbox.projection import project_wave
from toolbox.comparison import compare_meas_simu, create_RSA_matrices, compare_meas_simu_oneChType
import time

# Use of scipy.optimize.curve_fit(f, xdata, ydata)
# f should take as input xdata (fixed parameters of the simu) and the parameters 
# to change, then give as output the complex matrice of the simulated data.
# ydata should be of the same shape.

entry_file = "C:\\Users\\laeti\\Data\\wave_model\\scripts_python\\WAVES\\test\\entry\\entry.ini"
entry_list = configIO.read_entry_config(entry_file)
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

entry = utils.Entry.load_entry(entry_dic)

    simulation_config_section='0cfg', 
    screen_config_section='cfg1',


# xdata = xdata(subject, session, cond, sensorspath, epoch_type, ep_name)

xdata = np.zeros(26284,)
ydata = ydata_cplx(subject, session, comp, cond, sensorspath)

# phases = f_phase(xdata,p)

popt, pcov = curve_fit(f_phase, xdata, ydata) # popt = 0.999975 pcov = 1.12099937e-10

# curve_fit takes as inputs
# - f so that f(xdata,p1) = ydatafit
# - xdata -> no need for them
# - ydata -> vector of measured data
# return p1 optimized so that f(xdata,p1) is fitted to ydata

# On amplitude
ydata = ydata_ampl(subject, session, comp, cond, sensorspath) # measured
# note ydata is sometimes equal to 0, why? full matrix considered (after taking log) CORRECTED
xdata = np.zeros(np.shape(ydata))
#ampls = f_ampl(xdata,p)

popt, pcov = curve_fit(f_ampl, xdata, ydata) # popt = 1. pcov = [inf]

# plot fitted data
ydata_fit = f_ampl(xdata, popt)
ydata_fixed = f_ampl(xdata, 5) # fixed temporal frequency

R2_fixed, pval_fixed = scipy.stats.spearmanr(ydata, ydata_fixed)
R2_fit, pval_fit = scipy.stats.spearmanr(ydata, ydata_fit)

plt.plot(ydata, ydata_fit, 'o')
plt.plot(ydata_fixed, ydata_fit, 'o')