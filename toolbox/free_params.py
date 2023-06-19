##############################################################################
#### estimate_free_parameters.py
##############################################################################
# -*- coding: utf-8 -*-

"""
Estimate free parameters for each participant.

Created on Mon Jun 19 14:39:34 2023

@author: laeti
"""

from CL_config import sensorspath
from CL_xdata import xdata
from CL_ydata import ydata_phase, ydata_ampl , ydata_cplx
from CL_f import f_phase, f_ampl



from scipy.optimize import curve_fit
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

# copy from compute_compare_proj_data_RSA
def compare_meas_simu(fname_meas, fname_simu):
    '''
    Compare the relationships between pairs of sensors in term of amplitude, phase
    or complex values between measured and simulated signal.
    
    Parameters
    ----------
    fname_meas : STRING
        Path and file name for the measured data file to compare (should contain an evoked).
    fname_simu : STRING
        Path and file name for the simulated data file to compare (should contain an evoked).

    Returns
    -------
    phases : ARRAY meas/simu*channels*time points
        Instantaneous phase at 5 Hz for each signal, channel and time point.
    ampls : ARRAY meas/simu*channels*time points
        Instantaneous amplitude at 5 Hz for each signal, channel and time point.
    times : ARRAY
        Times of the evoked files.
    evoked.info : INFO file
        Infor from evoked files.
    zscores : ARRAY channels*amp/phase/cplx
        Zscores of the comparison between measured vs simulated signal, for each 
        channel and each comparison type (amplitude, phase or complex).
    R2_all : ARRAY channels*amp/phase/cplx
        Correlation coefficient of the comparison between measured vs simulated 
        signal, for each channel and each comparison type (amplitude, phase or complex).
    pval_all : ARRAY channels*amp/phase/cplx
        P-value from the comparison between measured vs simulated 
        signal, for each channel and each comparison type (amplitude, phase or complex).
    matrices : DICT {channel type: amp/phase/cplx*meas/pred*chan*chan}
        Amplitude/phase/complexe matrices for each signal and each channel pair.

    '''
    # initiate outputs
    zscores = np.zeros((len(ch_types), 3))  # zscore of the correlation ch_type*amp/phase/cplx  
    R2_all = np.zeros((len(ch_types), 3)) 
    pval_all = np.zeros((len(ch_types), 3)) 
    matrices = {'eeg': np.zeros((3,2,74,74),dtype = 'complex_'), # amp/phase/cplx *meas/pred*chan*chan
                'mag': np.zeros((3,2, 102,102),dtype = 'complex_'),   
                'grad': np.zeros((3, 2, 102,102),dtype = 'complex_')} # complex connectivity matrices for each channel type

    # Read measured data evoked
    evoked = mne.read_evokeds(fname_meas)[0]
    evoked.crop(tmin=0.5, tmax = 2)
    Nt = len(evoked.times)
    sf = evoked.info['sfreq']
    
    # Read projections         
    ev_proj = mne.read_evokeds(fname_simu)[0]
    ev_proj.crop(tmin=0.5, tmax = 2)
    evokeds = [evoked, ev_proj] # 'measured', 'simulated' 

    # Extract phase and amplitude at each time point at 5Hz
    phases = np.zeros((len(evokeds), np.shape(evoked.data)[0], Nt)) # train, test proj
    ampls = np.zeros((len(evokeds), np.shape(evoked.data)[0], Nt)) 
    for e, ev in enumerate(evokeds):  
        # extract instantaneous phase and amplitude
        dat = ev.data[np.newaxis,:]
        tfr = mne.time_frequency.tfr_array_morlet(dat, sfreq=sf, freqs=freqs, n_cycles=n_cycles,output='complex')
        phase = np.angle(tfr)
        amp =  np.abs(tfr)
        
        # extract instantaneous phase and amplitude at frequency of interest
        phases[e] = np.squeeze(phase[0,:,freqs == fstim])
        ampls[e] = np.squeeze(amp[0,:,freqs == fstim])

    times = evoked.times
    msk_t = times > tmin_crop  
            
    ###### AMPLITUDE ###### 
    # Compute a matrix of relative amplitude between sensors for each sensor type
    cov_amp = [] # log matrices for each sensor type
    for ch,ch_type in enumerate(ch_types):
        
        if ch_type == 'grad':
            # do the average of relative amplitude between grad1 and grad2
            inds1 = mne.pick_types(evoked.info, meg = 'planar1', eeg = False)
            inds2 = mne.pick_types(evoked.info, meg = 'planar2', eeg = False)
            ind_grad = [inds1, inds2]
            cov_tmp = np.zeros((2,len(evokeds), len(inds1), len(inds2)))
            for j in range(len(ind_grad)):
                inds = ind_grad[j]
                for i in range(len(evokeds)): # for measured and predicted data
                    ampls_ch = ampls[i,inds][:,msk_t]    #ampls[i,inds,msk_t]
                    # loop across rows
                    for r in range(len(inds)):
                        for c in range(len(inds)): # loop across column
                            #cov_tmp[j,i,r,c] = np.mean(ampls_ch[r]/ampls_ch[c]) # average across time OLD
                            cov_tmp[j,i,r,c] = np.mean(np.log(ampls_ch[r]/ampls_ch[c])) # so that cov is a symetrical matrix
            cov_ch = cov_tmp.mean(0)
        else:
            inds = mne.pick_types(evoked.info, meg = choose_ch[ch][0], eeg = choose_ch[ch][1])
            cov_ch = np.zeros((len(evokeds), len(inds), len(inds)))
            
            for i in range(len(evokeds)): # for measured and predicted data
                ampls_ch = ampls[i,inds][:,msk_t] 
                # loop across rows
                for r in range(len(inds)):
                    for c in range(len(inds)): # loop across column
                        #cov_ch[i,r,c] = np.mean(ampls_ch[r]/ampls_ch[c]) # average across time OLD
                        cov_ch[i,r,c] = np.mean(np.log(ampls_ch[r]/ampls_ch[c])) # so that cov is a symetrical matrix
                        
                                        
        #cov_amp.append(np.log(cov_ch)) # OLD
        cov_amp.append(cov_ch)
            
        # Correlate measured vs predicted matrix
        msk_tri = np.triu(np.ones((len(inds), len(inds)), bool), k = 1) # zero the diagonal and all values below
        meas = cov_ch[0][msk_tri]
        simu = cov_ch[1][msk_tri]
        R2, pval = scipy.stats.spearmanr(meas,simu)
        zscores[ch, 0] = 0.5*np.log((1+R2)/(1-R2)) # fisher Z (ok for spearman when N>10)
        R2_all[ch, 0] = R2
        pval_all[ch, 0] = pval            
            
    ###### PHASE ######
    # Compute a matrix of relative phase between sensors for each sensor type
    cov_phase = []
    for ch,ch_type in enumerate(ch_types):  
        if ch_type == 'grad':
            # do the average of relative amplitude between grad1 and grad2
            inds1 = mne.pick_types(evoked.info, meg = 'planar1', eeg = False)
            inds2 = mne.pick_types(evoked.info, meg = 'planar2', eeg = False)
            inds = inds2
            cov_ch = np.zeros((len(evokeds), len(inds), len(inds)))
            for i in range(len(evokeds)): # for measured and predicted data
                phases_1 = phases[i,inds1][:,msk_t]  
                phases_2 = phases[i,inds2][:,msk_t] 
                # loop across rows
                for r in range(len(inds)):
                    for c in range(len(inds)): # loop across column
                        d1 = phases_1[r] - phases_1[c]
                        d11 = (d1 + np.pi) % (2 * np.pi) - np.pi # put between -pi and pi
                        
                        d2 = phases_2[r] - phases_2[c]
                        d22 = (d2 + np.pi) % (2 * np.pi) - np.pi # put between -pi and pi   
                        
                        cov_ch[i,r,c] = circular_mean(np.array([circular_mean(d11), circular_mean(d22)]))

        else:                    
            inds = mne.pick_types(evoked.info, meg = choose_ch[ch][0], eeg = choose_ch[ch][1])
            cov_ch = np.zeros((len(evokeds), len(inds), len(inds)))
        
            for i in range(len(evokeds)): # for measured and predicted data
                phases_ch = phases[i,inds][:,msk_t]
                # loop across rows
                for r in range(len(inds)):
                    for c in range(len(inds)): # loop across column
                        d1 = phases_ch[r] - phases_ch[c]
                        d = (d1 + np.pi) % (2 * np.pi) - np.pi # put between -pi and pi
                        cov_ch[i,r,c] = circular_mean(d)
                                
        cov_phase.append(cov_ch)            
        
        # Correlate measured vs predicted matrix
        msk_tri = np.triu(np.ones((len(inds), len(inds)), bool), k = 1) # zero the diagonal and all values below
        meas = cov_ch[0][msk_tri]
        simu = cov_ch[1][msk_tri]                    
        R2, zscore, pval = circular_corr(meas, simu)
        zscores[ch, 1] = zscore
        R2_all[ch, 1] = R2
        pval_all[ch, 1] = pval            
            
    ###### CPLEX NBER ######
    # Compute a complex matrix combinining amplitude and phase
    for ch,ch_type in enumerate(ch_types): 
        if ch_type == 'grad':
            inds = mne.pick_types(evoked.info, meg = 'planar1', eeg = choose_ch[ch][1])
        else:
            inds = mne.pick_types(evoked.info, meg = choose_ch[ch][0], eeg = choose_ch[ch][1])
            
        # Complex matrix
        covX = cov_amp[ch]*np.exp(cov_phase[ch]*1j)   
        matrices[ch_type][0] = cov_amp[ch]
        matrices[ch_type][1] = cov_phase[ch]
        matrices[ch_type][2] = covX
        
        # Correlate matrix using conjugate
        msk_tri = np.triu(np.ones((len(inds), len(inds)), bool), k = 1) # zero the diagonal and all values below
        meas = covX[0][msk_tri]
        simu = covX[1][msk_tri]        
        
        # amplitude of the complex correlation coefficient
        R2 = np.abs(np.corrcoef(meas,simu))[1,0]
        zscores[ch, 2] = 0.5*np.log((1+R2)/(1-R2)) # fisher Z (ok for spearman when N>10)  Kanji 2006 as ref too
        df = len(meas)-2
        tscore = R2*np.sqrt(df)/np.sqrt(1-R2*R2) # Kanji 2006
        pval = scipy.stats.t.sf(tscore, df)
        pval_all[ch, 2] = pval
        R2_all[ch, 2] = R2

    return phases, ampls, times, evoked.info, zscores, R2_all, pval_all, matrices


def f_model(entry, p):
    ''' Free parameter p = temporal frequency'''
    freq_temp = p
    freq_spacial = 0.05
    amplitude = 1e-08
    phase_offset = 1.5707963267948966
    entry.set_simulation_params([freq_temp, freq_spacial, amplitude, phase_offset])
       
    stc = generate_simulation(entry)
    proj = project_wave(entry, stc)
    compare = compare_meas_simu(entry, proj)  
    matrices = create_RSA_matrices(entry, evoked)
    mat_mag = matrices['mag'][2] # pick the cplx one
    
    # select complex mag matrices
    matrices = compare[7]['mag']

    return 
    
    
    
    
###### TESTS ##################################################################

from toolbox import configIO
from toolbox.simulation import generate_simulation
from toolbox.projection import project_wave
from toolbox.comparison import compare_meas_simu, create_RSA_matrices

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