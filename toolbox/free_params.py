##############################################################################
#### estimate_free_parameters.py
##############################################################################
# -*- coding: utf-8 -*-

"""
Estimate free parameters for each participant.

Created on Mon Jun 19 14:39:34 2023

@author: laeti
"""


from scipy.optimize import curve_fit, least_squares,differential_evolution
import numpy as np
from copy import deepcopy

from toolbox.simulation import create_sim_from_entry
from toolbox.projection import project_wave
from toolbox.comparison import compare_meas_simu, create_RSA_matrices, from_meas_evoked_to_matrix
import time
import os
import pickle
from joblib import Parallel, delayed
from toolbox.entry import Entry

subject = 'OF4IP5'
session = 'session2'
comp = 'matched'
cond = 'full_out'

p = 0.05

epoch_type = "SSVEP" # SSVEP 5Hzfilt
ep_name = {'SSVEP': '-', '5Hzfilt': '_5Hzfilt-'}

# my path
wdir = '/home/lgrabot/wave_model/'
sensorspath = wdir + 'data_MEEG/sensors/' 

###### FUNCTIONS ##############################################################

######################## old ######################## 

def func(parameter, entry, ch_types, verbose = False):
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
    res_meas = from_meas_evoked_to_matrix(entry, ch_types, verbose=False)
    compare = compare_meas_simu(entry, ev_proj, ch_types = ch_types, meas = res_meas, verbose=verbose)
    
    # select sum of squared residuals for all channel types and the complex comparison
    SSR = compare[-1] # compare[-1][:,2]
    R = compare[4] # compare[5][:,2]
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return SSR, R  

def func_sF(parameter, param_name, entry, res_meas, ch_types, out = 'SSR', verbose = False):
    ''' 
    Parameters
    ----------
    parameter :  ndarray of shape (k,) 
        k free parameters to fit.
    param_name : list of string
        List of lenght k with the name of the free parameters ('spatialFreq', 'decay')
        
    Free parameter p = spatial frequency'''
    
    if type(entry) == Entry: # only one condition
        # attribute parameter
        freq_temp = 5
        amplitude = 1e-08
        phase_offset = 1.5707963267948966
        e0 = 5 # dva
        if param_name[0] == 'spatialFreq':
            freq_spatial = parameter
        else: 
            freq_spatial = 0.05
        if param_name[0] == 'decay':
            decay = parameter
        else: 
            decay = 0 # no decay            
        
        entry.simulation_params = [freq_temp, freq_spatial, amplitude, phase_offset, decay, e0]
        sim = create_sim_from_entry(entry)
        stc = sim.generate_simulation()
        ev_proj = project_wave(entry, stc, verbose) 
        compare = compare_meas_simu(entry, ev_proj, ch_types = ch_types, meas = res_meas, verbose=verbose)
        
        # select sum of squared residuals for all channel types and the complex comparison
        SSR = compare[-1][:,2] # only complex data SSR
        R = compare[4][:,2]
    
    elif type(entry) == list: # several conditions to combine
        SSR = 0
        R = []
        for ent, res in zip(entry, res_meas):
            if out == 'SSR':
                SSR += func_sF(parameter, ent, res, ch_types, out)
            elif out == 'R':
                R.append(func_sF(parameter, ent, res, ch_types, out))

    if out == 'SSR':
        return SSR
    elif out == 'R':
        return R


def fit_tempFreq(entry, subject, condition_meas, condition_simu, parameters_to_test):
    '''
    

    Parameters
    ----------
    subject : str
        DESCRIPTION.
    condition_meas : str
        'trav_out', 'trav_in', 'fov_out'
    condition_simu : str
        'trav_out', 'trav_in', 'fov_out'
    Returns
    -------
    best_parameter : float
        DESCRIPTION.

    '''
        
    # Change condition for simulation
    if condition_simu == 'trav_out' :
        entry.stim = 'TRAV_OUT'
        entry.c_space = 'full'
    elif condition_simu == 'trav_in':
        entry.stim = 'TRAV_IN'
        entry.c_space = 'full'
    elif condition_simu == 'fov_out':
        entry.stim = 'TRAV_OUT'
        entry.c_space = 'fov'       
    
    # Change subject
    entry.measured = wdir + 'data_MEEG/sensors/' + subject + '_' + condition_meas +'-ave.fif' # TO PUT BACK
    #entry.measured = wdir + 'data_MEEG/simulation/' + subject + '_proj_V1_session2_' + condition +'-ave.fif' # test pure 5Hz
    #€entry.measured = wdir + 'data_MEEG/simulation/' + subject + '_proj_V1_session2_5plus10Hz_' + condition +'-ave.fif' # noisy 5 + 10Hz
    #entry.measured = wdir + 'data_MEEG/simulation/' + subject + '_proj_V1_session2_5plus7Hz_' + condition +'-ave.fif' # pure 5 + 10Hz

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
                   "condition_meas": condition_meas,
                   "condition_simu": condition_simu,
                   "subject": subject
                   }
    fname = "{}_fit_param_tempFreq_session2_meas_{}_simu_{}.pkl".format(subject, condition_meas, condition_simu)
    a_file = open(os.path.join(wdir, 'data_MEEG', 'freeparam', fname), "wb")
    pickle.dump(fit_result, a_file)
    a_file.close() 
    
    return best_parameter, sumsq, R_all

def fit_spatFreq(entry, subject, param_name, condition_meas, condition_simu, ch_types):
    '''
    

    Parameters
    ----------
    subject : str
        DESCRIPTION.
    condition_meas : str
        'trav_out', 'trav_in', 'fov_out'
        If a list of conditions is passed, the SSR for each condition is calculated then summed.
    condition_simu : str
        'trav_out', 'trav_in', 'fov_out'
        If a list is passed, it should match the lenght of the list in condition_meas.

    Returns
    -------
    best_parameter : float
        DESCRIPTION.

    '''
    
    # Update subject
    entry.freesurfer = wdir + 'data_MRI/preproc/freesurfer/' + subject
    entry.forward_model = wdir + 'data_MEEG/preproc/'+ subject+'/forwardmodel/' + subject + '_session2_ico5-fwd.fif'
        
    if type(condition_meas) == str:   # only one condition
        cond_meas_name = condition_meas; cond_simu_name = condition_simu
        # Update simulation condition     
        if condition_simu == 'trav_out' :
            entry.stim = 'TRAV_OUT'
            entry.c_space = 'full'
        elif condition_simu == 'trav_in':
            entry.stim = 'TRAV_IN'
            entry.c_space = 'full'
        elif condition_simu == 'fov_out':
            entry.stim = 'TRAV_OUT'
            entry.c_space = 'fov'             
        
        # Update measured condition
        entry.measured = wdir + 'data_MEEG/sensors/' + subject + '_' + condition_meas +'-ave.fif' # TO PUT BACK
        entries = entry
        
        # Matrix for measured data
        res_meas = from_meas_evoked_to_matrix(entry, ch_types, verbose=False)
        
    elif type(condition_meas) == list:  # several conditions to combine
        entries = []
        res_meas = []
        cond_meas_name = 'comb_' ; cond_simu_name = 'comb_' 
        for cond_meas, cond_simu in zip(condition_meas, condition_simu):
            # # Update simulated condition     
            if cond_simu == 'trav_out' :
                entry.stim = 'TRAV_OUT'
                entry.c_space = 'full'
            elif cond_simu == 'trav_in':
                entry.stim = 'TRAV_IN'
                entry.c_space = 'full'
            elif cond_simu == 'fov_out':
                entry.stim = 'TRAV_OUT'
                entry.c_space = 'fov'             
            
            # Update measured condition
            entry.measured = wdir + 'data_MEEG/sensors/' + subject + '_' + cond_meas +'-ave.fif' # TO PUT BACK
            entries.append(deepcopy(entry))
            cond_meas_name += cond_meas + '_'
            cond_simu_name += cond_simu + '_'
            
            # Matrix for measured data
            res_meas.append(from_meas_evoked_to_matrix(entry, ch_types, verbose=False))            
        
    # Fit parameter
    ftol = 1e-10
    results = least_squares(func_sF, x0=p0, args=([param_name],entries, res_meas, ch_types), bounds = bounds, ftol=ftol, verbose=2)   
    print('--Participant {} optimized--'.format(subject))   
    from scipy.optimize import differential_evolution, minimize
    start = time.time()
    res = differential_evolution(func_sF, args=([param_name],entries, res_meas, ch_types), bounds = [bounds], workers = 20)   
    end = time.time()
    print( f'Optimisation finished after {end - start} seconds')
    res_mini = minimize(func_sF, x0=p0, args=([param_name],entries, res_meas, ch_types), bounds = [bounds], method="Nelder-Mead")
    diff_step = 0.33
    results = least_squares(func_sF, x0=p0, args=([param_name],entries, res_meas, ch_types), bounds = bounds, diff_step=diff_step, verbose=2)   


    # Save fitting parameters results
    fit_result = {"best_fit": results.x,
                  "SSR":  results.fun,
                  "full_results":  results,
                   "param_name": param_name,
                   "condition_meas": condition_meas,
                   "condition_simu": condition_simu,
                   "subject": subject
                   }
    fname = "{}_fit_param_{}_session2_meas_{}_simu_{}_{}.pkl".format(subject, param_name, cond_meas_name, cond_simu_name, ch_types[0])
    a_file = open(os.path.join(wdir, 'data_MEEG', 'freeparam', fname), "wb")
    pickle.dump(fit_result, a_file)
    a_file.close() 
                  
    return 


################################################

def fun_tF(parameter, param_name, entry, ch_types, out = 'SSR', verbose = False):
    ''' Free parameter p = temporal frequency'''
    
    if type(entry) == Entry: # only one condition

        amplitude = 1e-08
        phase_offset = 1.5707963267948966
        e0 = 5 # dva
        for p_name in param_name:
            if p_name == 'sfreq':
                freq_spatial = parameter
            else: 
                freq_spatial = 0.05
            if p_name == 'decay':
                decay = parameter
            else: 
                decay = 0 # no decay   
            if p_name == 'tfreq':
                freq_temp = parameter
            else: 
                freq_temp = 5
                
        entry.simulation_params = [freq_temp, freq_spatial, amplitude, phase_offset, decay, e0]
        sim = create_sim_from_entry(entry)
        stc = sim.generate_simulation()
        ev_proj = project_wave(entry, stc, verbose) 
        res_meas = from_meas_evoked_to_matrix(entry, ch_types, verbose=False)
        compare = compare_meas_simu(entry, ev_proj, ch_types = ch_types, meas = res_meas, verbose=verbose)
        
        # select sum of squared residuals for all channel types and the complex comparison
        SSR = compare[-1][:,2] # compare[-1][:,2]
        R = compare[4][:,2] # compare[5][:,2]

    elif type(entry) == list: # several conditions to combine
        SSR = 0
        R = []
        for ent in entry:
            if out == 'SSR':
                SSR += fun_tF(parameter, param_name, ent, ch_types,out)
            elif out == 'R':
                R.append(fun_tF(parameter, param_name, ent, ch_types, out))

    if out == 'SSR':
        return SSR
    elif out == 'R':
        return R


def fun(parameter, param_name, entry, res_meas, ch_types, out = 'SSR', verbose = False):
    ''' 

    Parameters
    ----------
    parameter :  ndarray of shape (k,) 
        k free parameters to fit.
    param_name : list of string
        List of lenght k with the name of the free parameters ('tfreq', 'sfreq', 'decay')
        
    Free parameter p = spatial frequency'''
    
    if type(entry) == Entry: # only one condition
        # attribute parameter
        amplitude = 1e-08
        phase_offset = 1.5707963267948966
        e0 = 5 # dva
        for p_name in param_name:
            if p_name == 'sfreq':
                freq_spatial = parameter
            else: 
                freq_spatial = 0.05
            if p_name == 'decay':
                decay = parameter
            else: 
                decay = 0 # no decay   
            if p_name == 'tfreq':
                freq_temp = parameter
            else: 
                freq_temp = 5
        
        entry.simulation_params = [freq_temp, freq_spatial, amplitude, phase_offset, decay, e0]
        sim = create_sim_from_entry(entry)
        stc = sim.generate_simulation()
        ev_proj = project_wave(entry, stc, verbose) 
        compare = compare_meas_simu(entry, ev_proj, ch_types = ch_types, meas = res_meas, verbose=verbose)
        
        # select sum of squared residuals for all channel types and the complex comparison
        SSR = compare[-1][:,2] # only complex data SSR
        R = compare[4][:,2]
    
    elif type(entry) == list: # several conditions to combine
        SSR = 0
        R = []
        for ent, res in zip(entry, res_meas):
            if out == 'SSR':
                SSR += fun(parameter, param_name, ent, res, ch_types)
            elif out == 'R':
                R.append(fun(parameter, param_name, ent, res, ch_types))

    if out == 'SSR':
        return SSR
    elif out == 'R':
        return R

def calculate_AIC_BIC(SSR, k, n):
    '''
    Calculate Akaike information criterion and Bayesian information criterion based on the sum of 
    squared residuals. Used to compare models.

    Parameters
    ----------
    SSR : float
        Sum of Squared Residuals.
    k : int
        Number of free parameters.
    n : int
        Number of observations.

    Returns
    -------
    AIC : float
        Akaike information criterion
    BIC : float
        Bayesian information criterion     
    '''
    AIC = 2*k + n*np.log(SSR/(n - k))
    BIC = n*np.log(SSR/n) + k*np.log(n)   
    
    return AIC, BIC
    
def fit_params(entry, subject, param_name, condition_meas, condition_simu, ch_types, bounds, workers):
    '''
    Final function to parallelize.
    Call fun_fT or fun as minimization function

    Parameters
    ----------
    subject : str
        Participant name.
    param_name : list of str
        Parameter(s) to fit. Temporal frequency: 'tfreq', 
        spatial frequency: 'sfreq', decay parameter: 'decay'
    condition_meas : str
        'trav_out', 'trav_in', 'fov_out'
        If a list of conditions is passed, the SSR for each condition is calculated then summed.
    condition_simu : str
        'trav_out', 'trav_in', 'fov_out'
        If a list is passed, it should match the lenght of the list in condition_meas.
    ch_types : list        
        List of channels to process( 'mag', 'grad', 'eeg').
     bounds : list  of tuples      
         List of min and max boundaries for each parameter to test: [(min1, max1), (min2, max2)]      
     workers : int     
         Number of parallel processing done in differential_evolution 
         
    Returns
    -------
    None
    '''
    
    # Update subject
    entry.freesurfer = wdir + 'data_MRI/preproc/freesurfer/' + subject
    entry.forward_model = wdir + 'data_MEEG/preproc/'+ subject+'/forwardmodel/' + subject + '_session2_ico5-fwd.fif'
    
    # Parameters
    if not type(param_name) == list:
        raise ValueError('param_name should be a list.')
    if len(param_name) > 1:
        p_name = '_'.join(param_name)
    else :
        p_name = param_name[0]
            
        
    if type(condition_meas) == str:   # only one condition
        cond_meas_name = condition_meas; cond_simu_name = condition_simu
        # Update simulation condition     
        if condition_simu == 'trav_out' :
            entry.stim = 'TRAV_OUT'
            entry.c_space = 'full'
        elif condition_simu == 'trav_in':
            entry.stim = 'TRAV_IN'
            entry.c_space = 'full'
        elif condition_simu == 'fov_out':
            entry.stim = 'TRAV_OUT'
            entry.c_space = 'fov'             
        
        # Update measured condition
        entry.measured = wdir + 'data_MEEG/sensors/' + subject + '_' + condition_meas +'-ave.fif' # TO PUT BACK
        entries = entry
        
        # Matrix for measured data
        res_meas = from_meas_evoked_to_matrix(entry, ch_types, verbose=False)
        Nchan = np.zeros((len(ch_types)))
        for ch, ch_type in enumerate(ch_types):
            Nchan[ch] = np.shape(res_meas[3][ch_type])[-1]
            
        
    elif type(condition_meas) == list:  # several conditions to combine

        cond_meas_name = '_'.join(condition_meas)
        cond_simu_name = '_'.join(condition_simu)
        entries = []
        res_meas = []
        for cond_meas, cond_simu in zip(condition_meas, condition_simu):
            # # Update simulated condition     
            if cond_simu == 'trav_out' :
                entry.stim = 'TRAV_OUT'
                entry.c_space = 'full'
            elif cond_simu == 'trav_in':
                entry.stim = 'TRAV_IN'
                entry.c_space = 'full'
            elif cond_simu == 'fov_out':
                entry.stim = 'TRAV_OUT'
                entry.c_space = 'fov'             
            
            # Update measured condition
            entry.measured = wdir + 'data_MEEG/sensors/' + subject + '_' + cond_meas +'-ave.fif' # TO PUT BACK
            entries.append(deepcopy(entry))
            
            # Matrix for measured data
            res = from_meas_evoked_to_matrix(entry, ch_types, verbose=False)
            res_meas.append(res)     
            Nchan = np.zeros((len(ch_types)))
            for ch, ch_type in enumerate(ch_types):
                Nchan[ch] = np.shape(res[3][ch_type])[-1]
        
    # Fit parameter 
    if 'tfreq' in param_name:
        results = differential_evolution(fun_tF, args=(param_name,entries, ch_types), bounds = bounds, seed=42, workers = workers )   
    else: 
        results = differential_evolution(fun, args=(param_name,entries, res_meas, ch_types), bounds = bounds, seed=42, workers = workers )     
    print('--Participant {} optimized--'.format(subject))   

    # Compute R at bestfit
    R = fun_tF(results.x, param_name, entries, ch_types, out='R')
    
    # Compute AIC/BIC and reference  AIC/BIC (with fixed parameters)
    SSR_fixed = fun_tF(results.x, [None], entries, ch_types, out='SSR')
    AIC = np.zeros((len(ch_types))); BIC = np.zeros((len(ch_types)));
    AIC_fixed = np.zeros((len(ch_types))); BIC_fixed = np.zeros((len(ch_types)));
    for ch, ch_type in enumerate(ch_types):
        n =  int(Nchan[ch]*(Nchan[ch]+1)/2) # nber of observations (upper triangle matrix)
        AIC[ch], BIC[ch] = calculate_AIC_BIC(SSR = results.fun, k = len(param_name) , n=n)
        AIC_fixed[ch], BIC_fixed[ch] = calculate_AIC_BIC(SSR = SSR_fixed , k = 0, n=n)
     
    # Save fitting parameters results
    fit_result = {"best_fit": results.x,
                  "SSR":  results.fun,
                  "full_results":  results,
                   "param_name": param_name,
                   'AIC': AIC,
                   'BIC': BIC,
                   'AIC_fixedParam' : AIC_fixed,
                   'BIC_fixedParam' : BIC_fixed,
                   'R_bestFit': R,
                   "condition_meas": condition_meas,
                   "condition_simu": condition_simu,
                   "param_bounds": bounds,
                   "subject": subject
                   }
    fname = "{}_fit_param_diffev_{}_session2_meas_{}_simu_{}_{}.pkl".format(subject, p_name, cond_meas_name, cond_simu_name, ch_types[0])
    a_file = open(os.path.join(wdir, 'data_MEEG', 'freeparam', fname), "wb")
    pickle.dump(fit_result, a_file)
    a_file.close() 
                  
    return 

