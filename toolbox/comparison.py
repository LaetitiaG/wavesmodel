import numpy as np
import mne
import scipy.stats as scistats
from math import sqrt, atan2, pi, floor, exp

# global variables
ch_types = ['mag', 'grad', 'eeg'] 
chansel = {'mag': ['mag', False],'grad': ['grad', False],'grad1': ['planar1', False],'grad2': ['planar2', False],'eeg': [False, True]}


def complex_corr(meas, simu):
    R2 = np.abs(np.corrcoef(meas, simu))[1, 0]
    # fisher Z (ok for spearman when N>10)  Kanji 2006 as ref too
    zscore = 0.5 * np.log((1 + R2) / (1 - R2))  
    df = len(meas) - 2
    tscore = R2 * np.sqrt(df) / np.sqrt(1 - R2 * R2)  # Kanji 2006
    pval = scistats.t.sf(tscore, df) 
    
    return R2, zscore, pval
                

def circular_corr(phases1, phases2):
    # from Topics in Circular Statistics (Jammalamadaka & Sengupta, 2001, World Scientific)
    m1 = circular_mean(phases1)
    m2 = circular_mean(phases2)

    sin1 = np.sin(phases1 - m1)
    sin2 = np.sin(phases2 - m2)
    rho = np.sum(sin1 * sin2) / np.sqrt(np.sum(sin1 ** 2) * np.sum(sin2 ** 2))

    # Statistical test
    l20 = np.sum(sin1 ** 2)
    l02 = np.sum(sin2 ** 2)
    l22 = np.sum(sin1 ** 2 * sin2 ** 2)
    zcorr = rho * np.sqrt(l20 * l02 / l22)
    p_value = 2 * (1 - scistats.norm.cdf(abs(zcorr)))

    return np.real(rho), np.real(zcorr), p_value


def circular_mean(phases):
    """
    Compute the circular mean of a phase vector in radian

    Parameters
    ----------
    phases : ARRAY
        Phases vector.

    Returns
    -------
    Circular mean in radian.

    """
    ang = np.mean(np.exp(1j*phases))
    return atan2(ang.imag, ang.real)


def circular_diff(phases1, phases2):
    """
    Calculates the circular difference between two arrays of phases.

    Parameters:
    phases1 (numpy.ndarray or list): First array of phase values in radians.
    phases2 (numpy.ndarray or list): Second array of phase values in radians.

    Returns:
    numpy.ndarray: The array of circular differences between phases1 and phases2 in radians.
    """

    # Convert inputs to numpy arrays if they are not already
    phases1 = np.array(np.real(phases1))
    phases2 = np.array(np.real(phases2))

    # Ensure phases are within the range of 0 to 2*pi
    phases1 = phases1 % (2 * np.pi)
    phases2 = phases2 % (2 * np.pi)
    diff = phases2 - phases1

    # Adjust differences outside of the range -pi to pi
    diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)
    diff = np.where(diff < -np.pi, diff + 2 * np.pi, diff)

    return diff

  
def compare_meas_simu(entry, ev_proj, verbose=False):
    '''
    Compare the measured and projected signals after rereferencing
    each signal to a global reference (average across all sensors).
    Return single-sensor correlations coefficients between rereferenced signals.

    Parameters
    ----------
    entry : class
        Class for entry values
    ev_proj : EVOKED
        Evoked instance of the model predictions.

    Returns 
    -------
    R2: ARRAY of shape Nsensors
        Correlation coefficient between data and realigned predicted time series for each sensor.
    pval: ARRAY of shape Nsensors
        Associated p-value.
    SSR: ARRAY of shape Nsensors
        Associated sum of squared residuals.
    fft_meas: ARRAY (size Nsensors*Nfreq)
        Fast Fourier Transform (FFT) spectrum of the measured data.
    freqs: ARRAY (size Nfreq)
        Frequency range used in the FFT.
    thetaRef: ARRAY of shape Nchannel
        Phase reference used to realigned the predicted time series. One for each channel types (MAG, GRAD, EEG).
    aRef: ARRAY of shape Nchannel
        Amplitude correction used to realigned the predicted time series. One for each channel types (MAG, GRAD, EEG).
    R2_glob: list (size Nchannel)
        Correlation coefficient between measured and predicted data for each channel type (mag, grad, eeg).
    pval_glob:
        Associated p-value.  
    reref_proj: ARRAY of Nsensors*Ntimes or Ncond*Nsensors*Ntimes
        Realigned predicted time series.
    phase_shift:  ARRAY of Nsensors or Ncond*Nsensors
        Phase shifts between measured and predicted data, for each sensor.
    ampl_meas: ARRAY (size Nsensors)
        Amplitude of measured data at the frequency of interst fstim
    phase_meas: ARRAY (size Nsensors) 
        Phase of measured data at the frequency of interst fstim
    ampl: ARRAY (size Nsensors) 
        Amplitude of predicted data at the frequency of interst fstim
    phase: ARRAY (size Nsensors)
        Phase of predicted data at the frequency of interst fstim  
    '''
    
    ampl_meas, phase_meas, ampl, phase, fft_meas, freqs, fstim, evoked = calculate_phase_ampls(entry, ev_proj, verbose=verbose)

    # Create signal to be correlated (complex correlation)
    meas = ampl_meas*np.exp(phase_meas*1j)
    pred = ampl*np.exp((phase)*1j)       
    R2_glob, pval_glob, MSE = calculate_corr_coef_cplx(meas, pred, evoked)

    # Correlations per sensor between reconstructed measure and predicted signals
    R2, pval, SSR, thetaRef, aRef, reref_proj, phase_shift = calculate_corr_coef_rerefSignal(ampl_meas, phase_meas, ampl, phase, fstim, evoked, ev_proj) 

    return R2, pval, SSR, fft_meas, freqs, thetaRef, aRef, R2_glob, pval_glob, reref_proj, phase_shift, ampl_meas, phase_meas, ampl, phase
    

def calculate_phase_ampls(entry, ev_proj, verbose=False):
    '''
    Calculate the phase and amplitude of the measured and predicted data.

    Parameters
    ----------
    entry : class
        Class for entry values
    ev_proj : EVOKED
        Evoked instance of the model predictions.

    Returns 
    -------
    ampl_meas: ARRAY (size Nchannel)
        Amplitude of measured data at the frequency of interst fstim
    phase_meas: ARRAY (size Nchannel) 
        Phase of measured data at the frequency of interst fstim
    ampl: ARRAY (size Nchannel) 
        Amplitude of predicted data at the frequency of interst fstim
    phase: ARRAY (size Nchannel)
        Phase of predicted data at the frequency of interst fstim
    fft_meas: ARRAY (size Nchannel*Nfreq)
        Fast Fourier Transform (FFT) spectrum of the measured data.
    freqs: ARRAY (size Nfreq)
        Frequency range used in the FFT.
    fstim: Float
        Stimulation frequency (Hz).
    evoked: Evoked instance
        Measured data
    '''

    # Read measured data evoked
    evoked = mne.read_evokeds(entry.measured, verbose=verbose)[0]
    evoked.crop(tmin=0, tmax=2)
    Nt = len(evoked.times); Fs = evoked.info['sfreq']
    Nchan = len(evoked.data)
    
    # Calculate FFT
    fft_meas = np.fft.fft(evoked.data, axis = 1)
    freqs = np.fft.fftfreq(Nt, 1/Fs)

    # Extract phase and amplitude at stimulation freq
    fstim = entry.simulation_params.freq_temp
    ind = np.nanargmin(np.abs(freqs - fstim)) # find closest freq
    phase_meas = np.angle(fft_meas[:,ind])  
    ampl_meas = 2*np.abs(fft_meas[:,ind])/Nt

    # Calculate FFT on predicted data
    hamm = np.hamming(Nt)
    fft = np.fft.fft(hamm *ev_proj.data, axis = 1)
    ampl = 2*np.abs(fft[:,ind])/Nt
    phase = np.angle(fft[:,ind]) 

    return ampl_meas, phase_meas, ampl, phase, fft_meas, freqs, fstim, evoked
        

def calculate_corr_coef_cplx(meas, pred, evoked):
    '''
    Calculate the complex correlation between measured and predicted data,
    from the amplitude and phase.

    Parameters
    ----------
    entry : class
        Class for entry values
    ev_proj : EVOKED
        Evoked instance of the model predictions.

    Returns
    -------
    R2_glob: list (size Nchannel)
        Correlation coefficient between measured and predicted data for each channel type (mag, grad, eeg).
    pval_glob:
        Associated p-value.
    MSE: ARRAY (size Nsensors)
        Normalized mean squared error between measured and predicted data, for each sensor.
    '''

    R2_glob = np.zeros(len(ch_types));  pval_glob = np.zeros(len(ch_types))
    MSE  = np.zeros((len(evoked.data)))
    for c, ch_type in enumerate(ch_types):
        inds_chan = mne.pick_types(evoked.info, meg = chansel[ch_type][0], eeg = chansel[ch_type][1])
        # Single correlation for all sensor together (no rereferencing needed) 
        x = pred[inds_chan]
        y = meas[inds_chan]
        R2_glob[c], z, pval_glob[c] = complex_corr(y, x)

        # Calculate MSE (normalized mean squared error)
        res = scistats.linregress(x,y) 
        y_fit = res.slope*x + res.intercept        
        MSE[inds_chan] = np.sqrt( (y - y_fit)*np.conjugate(y - y_fit)/(y_fit*np.conjugate(y_fit)) ) # mse
                  
    return R2_glob, pval_glob, MSE


def calculate_corr_coef_rerefSignal(ampl_meas, phase_meas, ampl, phase, fstim, evoked, ev_proj):
    '''
    Calculate the correlation between the reconstructed measured data at fstim
    and the realigned predicted signal, for each sensor.

    Handle multiple conditions, by calculating an average phase reference and amplitude ratio correction
    across conditions, and realign each condition separately. The correlation is calculated after concatenating
    all conditions

    Parameters
    ----------
    ampl_meas: ARRAY of size Ncond*Nsensors or size Nsensors
        Amplitude of measured data at the frequency of interst fstim
    phase_meas: ARRAY of size Ncond*Nsensors or size Nsensors 
        Phase of measured data at the frequency of interst fstim
    ampl: ARRAY of size Ncond*Nsensors or size Nsensors 
        Amplitude of predicted data at the frequency of interst fstim
    phase: ARRAY of size Ncond*Nsensors or size Nsensors
        Phase of predicted data at the frequency of interst fstim
    ev_proj: EVOKED
        Evoked instance of the model predictions.
    fstim: DOUBLE
        Frequency of interest.
    evoked: instance of evoked or lists of evokeds
        Measured data.
    ev_proj: instance of evoked or lists of evokeds
        Predicted data.

    Returns  
    -------
    R2: ARRAY of shape Nsensors
        Correlation coefficient between data and realigned predicted time series for each sensor.
    pval: ARRAY of shape Nsensors
        Associated p-value.
    SSR: ARRAY of shape Nsensors
        Associated sum of squared residuals.
    thetaRef: ARRAY of shape Nchannel
        Phase reference used to realigned the predicted time series. One for each channel types (MAG, GRAD, EEG).
    aRef: ARRAY of shape Nchannel
        Amplitude correction used to realigned the predicted time series. One for each channel types (MAG, GRAD, EEG).
    reref_proj: ARRAY of Nsensors*Ntimes or Ncond*Nsensors*Ntimes
        Realigned predicted time series.
    phase_shift:  ARRAY of Nsensors or Ncond*Nsensors
        Phase shifts between measured and predicted data, for each sensor.
    '''
   
    # Store phase shift meas-proj 
    phase_shift = circular_diff(phase_meas, phase) 

    if len(np.shape(ampl_meas)) ==2 :
        reref_proj = np.zeros((np.shape(ampl_meas)[0], np.shape(ev_proj[0].data)[0], np.shape(ev_proj[0].data)[1]))
        data = np.zeros((np.shape(ampl_meas)[0], np.shape(ev_proj[0].data)[0], np.shape(ev_proj[0].data)[1]))
        Nchan = len(evoked[0].data)
    else: 
        reref_proj = np.zeros(np.shape(ev_proj.data))
        Nchan = len(evoked.data)
    thetaRef = np.zeros((len(ch_types)), dtype='complex_'); aRef = np.zeros(len(ch_types))
    R2 = np.zeros((Nchan)); pval = np.zeros((Nchan)); SSR = np.zeros((Nchan))

    for c, ch_type in enumerate(ch_types):

        if len(np.shape(ampl_meas)) ==2 : # if there are several conditions to concatenate
            inds_chan = mne.pick_types(evoked[0].info, meg = chansel[ch_type][0], eeg = chansel[ch_type][1])
            cplex_mean_meas_cd = []; cplex_mean_proj_cd = []
            for n in range(np.shape(ampl_meas)[0]):
                cplex_mean_meas_cd.append( np.mean(ampl_meas[n,inds_chan]*np.exp(phase_meas[n,inds_chan]*1j)) ) 
                cplex_mean_proj_cd.append( np.mean(ampl[n,inds_chan]*np.exp(phase[n,inds_chan]*1j)) )
            cplex_mean_meas = np.mean(cplex_mean_meas_cd); cplex_mean_proj = np.mean(cplex_mean_proj_cd)
            aRef[c] = np.mean(np.mean(ampl_meas[:,inds_chan])/np.mean(ampl[:,inds_chan]))
        else: # if there is one single condition
            inds_chan = mne.pick_types(evoked.info, meg = chansel[ch_type][0], eeg = chansel[ch_type][1])
            cplex_mean_meas = np.mean(ampl_meas[inds_chan]*np.exp(phase_meas[inds_chan]*1j)) 
            cplex_mean_proj = np.mean(ampl[inds_chan]*np.exp(phase[inds_chan]*1j)) 
            aRef[c] = np.mean(ampl_meas[inds_chan])/np.mean(ampl[inds_chan])

        thetaRef[c] = circular_diff( np.angle(cplex_mean_meas), np.angle(cplex_mean_proj) )
        
        # Correlate sensor per sensor
        for ch in inds_chan:
            if len(np.shape(ampl_meas)) ==2 : # if there are several conditions to concatenate
                # Reconstruct predicted data after phase-rereferencing
                for n in range(np.shape(ampl_meas)[0]):
                    reref_proj[n,ch] = (aRef[c]*ampl[n,ch])*np.cos(2*np.pi*fstim*ev_proj[n].times + phase[n,ch] + thetaRef[c])
                    data[n,ch] = evoked[n].data[ch]
                # Correlate measured with reref predicted signals
                R2[ch], pval[ch] = scistats.spearmanr(data[:,ch].flatten(), reref_proj[:,ch].flatten())
                SSR[ch] = np.sum((data[:,ch].flatten()-reref_proj[:,ch].flatten())**2)
            else: 
                # Reconstruct predicted data after phase-rereferencing
                reref_proj[ch] = (aRef[c]*ampl[ch])*np.cos(2*np.pi*fstim*ev_proj.times + phase[ch] + thetaRef[c])
            
                # Correlate measured with reref predicted signals
                R2[ch], pval[ch] = scistats.spearmanr(evoked.data[ch], reref_proj[ch])
                SSR[ch] = np.sum((evoked.data[ch]-reref_proj[ch])**2)

    return R2, pval, SSR, thetaRef, aRef, reref_proj, phase_shift


def compare_meas_simu_concat(entries, ev_projs, verbose=False):
    '''
    Compare the measured and projected signals using complex correlations 
    between sensors, after concatenated the different conditions.

    Parameters
    ----------
    entries : list of class
        List of class for entry values
    ev_projs : list of EVOKED
        List of Evoked instance for the model predictions.

    Returns
    -------
    R2: ARRAY of shape Nsensors
        Correlation coefficient between data and realigned predicted time series for each sensor.
    pval: ARRAY of shape Nsensors
        Associated p-value.
    SSR: ARRAY of shape Nsensors
        Associated sum of squared residuals.
    fft_meas: ARRAY (size Nsensors*Nfreq)
        Fast Fourier Transform (FFT) spectrum of the measured data.
    freqs: ARRAY (size Nfreq)
        Frequency range used in the FFT.
    thetaRef: ARRAY of shape Nchannel
        Phase reference used to realigned the predicted time series. One for each channel types (MAG, GRAD, EEG).
    aRef: ARRAY of shape Nchannel
        Amplitude correction used to realigned the predicted time series. One for each channel types (MAG, GRAD, EEG).
    R2_glob: list (size Nchannel)
        Correlation coefficient between measured and predicted data for each channel type (mag, grad, eeg).
    pval_glob:
        Associated p-value.  
    reref_proj: ARRAY of Nsensors*Ntimes or Ncond*Nsensors*Ntimes
        Realigned predicted time series.
    phase_shift:  ARRAY of Nsensors or Ncond*Nsensors
        Phase shifts between measured and predicted data, for each sensor.
    ampl_meas_all, phase_meas_all: array Ncond*Nsensors
        Amplitude and phase of the measured data, for each tested condition.
    ampl_pred_all, phase_pred_all: array Ncond*Nsensors
        Amplitude and phase of the predicted data, for each tested condition.    
    MSE_all: array Ncond*Nsensors
        Normalized mean squared error for each tested condition.
    '''
    
    if len(entries) != len(ev_projs):
        raise ValueError('There should be the same number of entry and ev_proj.')
    N = len(entries) # number of conditions to concatenate

    # Compare measured with simulated data for each entry and projections
    meas = np.zeros((N, len(ev_projs[0].data)), dtype='complex_')
    pred = np.zeros((N, len(ev_projs[0].data)), dtype='complex_') 
    ampl_meas_all = np.zeros((N, len(ev_projs[0].data)), dtype='complex_') 
    phase_meas_all = np.zeros((N, len(ev_projs[0].data)), dtype='complex_') 
    ampl_pred_all = np.zeros((N, len(ev_projs[0].data)), dtype='complex_') 
    phase_pred_all = np.zeros((N, len(ev_projs[0].data)), dtype='complex_')     
    evokeds = []
    for i in range(N): 
        ampl_meas, phase_meas, ampl, phase, fft_meas, freqs, fstim, evoked = calculate_phase_ampls(entries[i], ev_projs[i], verbose=verbose)
        meas[i] = ampl_meas*np.exp(phase_meas*1j)
        pred[i] = ampl*np.exp((phase)*1j)       
        ampl_meas_all[i] = ampl_meas
        phase_meas_all[i] = phase_meas
        ampl_pred_all[i] = ampl
        phase_pred_all[i] = phase
        evokeds.append(evoked)

    # Calculate complex R on concatenated data
    R2_glob = np.zeros(len(ch_types));  pval_glob = np.zeros(len(ch_types))
    MSE_all  = np.zeros((len(ev_projs[0].data)))
    for c, ch_type in enumerate(ch_types):            
        inds_chan = mne.pick_types(ev_projs[0].info, meg = chansel[ch_type][0], eeg = chansel[ch_type][1])
        x = pred[:,inds_chan].flatten()
        y = meas[:,inds_chan].flatten()
        
        R2_glob[c], z, pval_glob[c] = complex_corr(y, x)

        # MSE
        res = scistats.linregress(x,y) 
        y_fit = res.slope*x + res.intercept
        MSE = np.sqrt( (y - y_fit)*np.conjugate(y - y_fit)/(y_fit*np.conjugate(y_fit)) ) 

        # Average MSE across each condition 
        MSE_mean = np.reshape(MSE, np.shape(pred[:,inds_chan])).mean(0)
        MSE_all[inds_chan] = MSE_mean

    # Calculate R per sensor between reconstructed measured and predicted time series
    R2, pval, SSR, thetaRef, aRef, reref_proj, phase_shift = calculate_corr_coef_rerefSignal(ampl_meas_all, phase_meas_all, ampl_pred_all, phase_pred_all, fstim, evokeds, ev_projs) 

    return R2, pval, SSR, fft_meas, freqs, thetaRef, aRef, R2_glob, pval_glob, reref_proj, phase_shift, ampl_meas_all, phase_meas_all, ampl_pred_all, phase_pred_all, MSE_all

