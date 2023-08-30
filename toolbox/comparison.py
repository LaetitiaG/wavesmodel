import numpy as np
import mne
import scipy.stats as scistats
from math import sqrt, atan2, pi, floor, exp
from numba import jit


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

def TS_SS(v1, v2):
    '''
    Calculate Triangle Similarity-Sector Similarity (TS-SS) between two vectors.
    This distance metric accont for difference in phase and magnitude.
    
    Source: A. Heidarian and M. J. Dinneen, "A Hybrid Geometric Approach for 
    Measuring Similarity Level Among Documents and Document Clustering," 2016 
    IEEE Second International Conference on Big Data Computing Service and 
    Applications (BigDataService), Oxford, UK, 2016, pp. 142-151, 
    doi: 10.1109/BigDataService.2016.14.

    Parameters
    ----------
    v1 : ARRAY
        Array of vectors in complex form.
    v2 : ARRAY
        Array of vectors in complex form.

    Returns
    -------
    TS_SS : ARRAY
        TS-SS metric, between 0 and +infinite. The more similar, the closer to zero.

    '''
    
    # Euclidean distance
    ED = np.linalg.norm(v1[:, np.newaxis]-v2[:, np.newaxis], axis = 1)

    # Angle between v1 and v2
    V = np.real(v1*np.conjugate(v2).T)/(np.linalg.norm(v1[:, np.newaxis], axis=1) * np.linalg.norm(v2[:, np.newaxis], axis=1))
    theta = np.arccos(V) + np.radians(10)    
    theta[theta > np.pi] = np.pi # correct the 10° adjustement so that theta is between 0 and pi
    
    # Magnitude difference
    MD = abs((np.linalg.norm(v1[:, np.newaxis], axis = 1) - np.linalg.norm(v2[:, np.newaxis], axis = 1)))
    
    # Triangle's area similarity
    TS = ((np.linalg.norm(v1[:, np.newaxis], axis = 1) * np.linalg.norm(v2[:, np.newaxis], axis = 1)) * np.sin(theta))/2
    
    # Sector’s Area Similarity
    SS = np.pi * (ED + MD)**2 * theta/360    
    
    return TS*SS
    

    
def create_RSA_matrices(entry, evoked, ch_type, verbose=False):
    """
    Create RSA matrices for phase and amplitude relationship between each pair
    of sensors of the given channel type.

    Parameters
    ----------
    entry : class
        Class for entry values
    evoked : class
        Evoked data.
    ch_type : STRING
        'mag', 'grad', 'eeg'
    Returns
    -------
    phases : ARRAY meas/simu*Nchan*time points
        Instantaneous phase at 5 Hz for each signal, channel and time point.
    ampls : ARRAY meas/simu*Nchan*time points
        Instantaneous amplitude at 5 Hz for each signal, channel and time point.
    times : ARRAY
        Times of the evoked files.    
    matrices : ARRAY amplitude/phase/complex*Nchan*Nchan
        RSA-like matrices for each feature of the signal (phase/ampl/cplx).

    """

    # Parameters for Morlet analysis for the SSVEP extraction
    fstim = entry.simulation_params.freq_temp
    freqs = np.arange(2., 50., 0.5)
    n_cycles = freqs / 2.
    tmin_crop = 0.5    
    sf = evoked.info['sfreq']
    
    # Select only channels of interest
    ev_ch = evoked.copy()
    ev_ch.pick(ch_type)
    if ch_type == 'grad':
        Nchan = int(len(ev_ch.data)/2)
    else:
        Nchan = len(ev_ch.data)
    
    # Initiate outputs (complex connectivity matrices)
    matrices = np.zeros((3, Nchan, Nchan), dtype='complex_') # amp/phase/cplx*chan*chan 

    # Extract phase and amplitude at each time point at 5Hz
    dat = ev_ch.data[np.newaxis, :]
    tfr = mne.time_frequency.tfr_array_morlet(dat, sfreq=sf, freqs=freqs, n_cycles=n_cycles, output='complex', n_jobs=4, verbose=verbose)
    phase = np.angle(tfr)
    amp = np.abs(tfr)

    # extract instantaneous phase and amplitude at frequency of interest
    fclose = freqs[np.nanargmin(np.abs(freqs - fstim))] # find closest freq
    phases = np.squeeze(phase[0, :, freqs == fclose])
    ampls = np.squeeze(amp[0, :, freqs == fclose])

    times = ev_ch.times
    msk_t = times > tmin_crop

    ###### AMPLITUDE ######
    # Compute a matrix of relative amplitude between sensors for each sensor type
    if ch_type == 'grad':
        # do the average of relative amplitude between grad1 and grad2
        inds1 = mne.pick_types(ev_ch.info, meg='planar1', eeg=False)
        inds2 = mne.pick_types(ev_ch.info, meg='planar2', eeg=False)
        ind_grad = [inds1, inds2]
        cov_tmp = np.zeros((2, len(inds1), len(inds2)))
        for j in range(len(ind_grad)):
            inds = ind_grad[j]
            ampls_ch = ampls[inds][:, msk_t]
            # loop across rows
            for r in range(len(inds)):
                for c in range(len(inds)):  # loop across column
                    cov_tmp[j, r, c] = np.mean(np.log(ampls_ch[r] / ampls_ch[c]))
        matrices[0] = cov_tmp.mean(0)
    else:
        ampls_ch = ampls[:, msk_t]
        # loop across rows
        for r in range(Nchan):
            for c in range(Nchan):  # loop across column
                # average across time and take the log to have a symmetrical matrix
                matrices[0][r, c] = np.mean(np.log(ampls_ch[r] / ampls_ch[c]))  
    
    ###### PHASE ######
    # Compute a matrix of relative phase between sensors for each sensor type
    if ch_type == 'grad':
        # do the average of relative amplitude between grad1 and grad2
        inds1 = mne.pick_types(ev_ch.info, meg='planar1', eeg=False)
        inds2 = mne.pick_types(ev_ch.info, meg='planar2', eeg=False)
        phases_1 = phases[inds1][:, msk_t]
        phases_2 = phases[inds2][:, msk_t]
        # loop across rows
        for r in range(len(inds1)):
            for c in range(len(inds1)):  # loop across column
                d1 = phases_1[r] - phases_1[c]
                d11 = (d1 + np.pi) % (2 * np.pi) - np.pi  # put between -pi and pi

                d2 = phases_2[r] - phases_2[c]
                d22 = (d2 + np.pi) % (2 * np.pi) - np.pi  # put between -pi and pi

                matrices[1][r, c] = circular_mean(np.array([circular_mean(d11), circular_mean(d22)]))

    else:
        phases_ch = phases[:, msk_t]
        # loop across rows
        for r in range(Nchan):
            for c in range(Nchan):  # loop across column
                d1 = phases_ch[r] - phases_ch[c]
                d = (d1 + np.pi) % (2 * np.pi) - np.pi  # put between -pi and pi
                matrices[1][r, c] = circular_mean(d)

    ###### CPLEX NBER ######
    # Compute a complex matrix combining amplitude and phase
    matrices[2] = matrices[0] * np.exp(matrices[1] * 1j)
    
    return phases, ampls, times,matrices    


def from_meas_evoked_to_matrix(entry, ch_types = ['mag', 'grad', 'eeg'], verbose=False):
    
    # Check that ch_types is a list
    if not(isinstance(ch_types, list)):
        print('ch_types should be a list, e.g. ["mag", "grad", "eeg"]')
        
    # Read measured data evoked
    evoked = mne.read_evokeds(entry.measured, verbose=verbose)[0]
    evoked.crop(tmin=0, tmax=2)
    
    # Create RSA matrices for the required channels
    matrices = {} 
    phases = {}; ampls= {}    
    for ch, ch_type in enumerate(ch_types):
        phases[ch_type], ampls[ch_type], times, matrices[ch_type] = create_RSA_matrices(entry, evoked, ch_type, verbose)
        
    return phases, ampls, times, matrices
    
# def compare_meas_simu_oneChType(entry, ev_proj, ev_meas, ch_type, verbose=False): ### OLD
#     '''
#     Compare the relationships between pairs of sensors in terms of amplitude, phase
#     or complex values between measured and simulated signal for a given channel type.

#     Parameters
#     ----------
#     entry : class
#         Class for entry values
#     ev_proj : STRING
#         Path and file name for the simulated data file to compare (should contain an evoked).
#     ch_type : STRING
#         'mag', 'grad', 'eeg'

#     Returns
#     -------
#     phases : ARRAY meas/simu*time points
#         Instantaneous phase at 5 Hz for each signal, channel and time point.
#     ampls : ARRAY meas/simu*time points
#         Instantaneous amplitude at 5 Hz for each signal, channel and time point.
#     times : ARRAY
#         Times of the evoked files.
#     evoked.info : INFO file
#         Infor from evoked files.
#     zscores : ARRAY amp/phase/cplx
#         Zscores of the comparison between measured vs simulated signal, for each
#         channel and each comparison type (amplitude, phase or complex).
#     R2_all : ARRAY amp/phase/cplx
#         Correlation coefficient of the comparison between measured vs simulated
#         signal, for each channel and each comparison type (amplitude, phase or complex).
#     pval_all : ARRAY amp/phase/cplx
#         P-value from the comparison between measured vs simulated
#         signal, for each channel and each comparison type (amplitude, phase or complex).
#     matrices_simu : ARRAY amp/phase/cplx*Nchan*Nchan
#         Amplitude/phase/complexe matrices for simulated data and each channel pair.
#     matrices_simu : ARRAY amp/phase/cplx*Nchan*Nchan
#         Amplitude/phase/complexe matrices for measured data and each channel pair.
#     SSR : ARRAY amp/phase/cplx
#         Sum of Squared Residuals (measured - simulated).
        
#     '''

#     # initiate outputs: amp/phase/cplx
#     zscores = np.zeros((3));     R2_all = np.zeros((3))
#     pval_all = np.zeros((3));

#     # Calculate RSA matrices
#     phases_meas, ampls_meas, times, matrices_meas = create_RSA_matrices(entry, ev_meas, ch_type, verbose)
#     phases_simu, ampls_simu, times, matrices_simu = create_RSA_matrices(entry, ev_proj, ch_type, verbose)
#     phases = np.stack((phases_meas, phases_simu)) # meas/simu
#     ampls = np.stack((ampls_meas, ampls_simu))
    
#     # Calculate correlations 
#     Nchan = np.shape(matrices_meas)[-1]
#     msk_tri = np.triu(np.ones((Nchan, Nchan), bool), k=1)  # zero the diagonal and all values below
    
#     # Initiate SSR (sum of square of residuals) array
#     SSR = np.zeros((3))
    
#     # Correlate measured vs predicted matrix
#     for i, data in enumerate(['amplitude', 'phase', 'complex']):
#         meas = matrices_meas[i][msk_tri]
#         simu = matrices_simu[i][msk_tri]
        
#         if data == 'amplitude':
#             R2, pval = scistats.spearmanr(meas, simu)
#             zscores[i] = 0.5 * np.log((1 + R2) / (1 - R2))  # fisher Z (ok for spearman when N>10)
#             SSR[i] = np.real(np.sum((meas-simu)**2))    
#         elif data == 'phase':
#             R2, zscore, pval = circular_corr(meas, simu)
#             zscores[i] = zscore
#             SSR[i] = np.sum(circular_diff(meas,simu)**2)
#         elif data == 'complex':
#             R2 = np.abs(np.corrcoef(meas, simu))[1, 0]
#             # fisher Z (ok for spearman when N>10)  Kanji 2006 as ref too
#             zscores[i] = 0.5 * np.log((1 + R2) / (1 - R2))  
#             df = len(meas) - 2
#             tscore = R2 * np.sqrt(df) / np.sqrt(1 - R2 * R2)  # Kanji 2006
#             pval = scistats.t.sf(tscore, df) 
#             #SSR[i] = np.sum(np.abs((meas-simu)**2)) # equivalent to np.sum(diff.real**2 + diff.imag**2)
                          
#         # Store statistics    
#         R2_all[i] = R2 
#         pval_all[i] = pval  
        
#     # Compute the SSR for complex values as the sum of SSR for phase and amp
#     SSR[2] = SSR[0] + SSR[1]
    
#     return phases, ampls, times, zscores, R2_all, pval_all, matrices_meas, matrices_simu, SSR

def compare_meas_simu(entry, ev_proj, meas= 'to_compute', ch_types = ['mag', 'grad', 'eeg'], verbose=False):
    '''
    Compare the relationships between pairs of sensors in terms of amplitude, phase
    or complex values between measured and simulated signal for a given channel type.

    Parameters
    ----------
    entry : class
        Class for entry values
    ev_proj : STRING
        Path and file name for the simulated data file to compare (should contain an evoked).
    meas : STRING or output from from_meas_evoked_to_matrix
        If 'to_compute', matrices are calculated from entry. 
    ch_type : STRING
        'mag', 'grad', 'eeg'

    Returns
    -------
    phases : ARRAY meas/simu*time points
        Instantaneous phase at 5 Hz for each signal, channel and time point.
    ampls : ARRAY meas/simu*time points
        Instantaneous amplitude at 5 Hz for each signal, channel and time point.
    times : ARRAY
        Times of the evoked files.
    evoked.info : INFO file
        Infor from evoked files.
    zscores : ARRAY amp/phase/cplx
        Zscores of the comparison between measured vs simulated signal, for each
        channel and each comparison type (amplitude, phase or complex).
    R2_all : ARRAY amp/phase/cplx
        Correlation coefficient of the comparison between measured vs simulated
        signal, for each channel and each comparison type (amplitude, phase or complex).
    pval_all : ARRAY amp/phase/cplx
        P-value from the comparison between measured vs simulated
        signal, for each channel and each comparison type (amplitude, phase or complex).
    matrices_simu : ARRAY amp/phase/cplx*Nchan*Nchan
        Amplitude/phase/complexe matrices for simulated data and each channel pair.
    matrices_simu : ARRAY amp/phase/cplx*Nchan*Nchan
        Amplitude/phase/complexe matrices for measured data and each channel pair.
    SSR : ARRAY amp/phase/cplx
        Sum of Squared Residuals (measured - simulated).
        
    '''

    # Calculate RSA matrices for measured data
    if meas == 'to_compute':
        phases_meas, ampls_meas, times, matrices_meas = from_meas_evoked_to_matrix(entry, ch_types, verbose=False)
    else:
        phases_meas = meas[0];  ampls_meas = meas[1]
        times = meas[2];        matrices_meas = meas[3]
    
    # Calculate RSA matrices for simulated data
    matrices_simu = {}; phases = {}; ampls= {}; TSSS = {}
    zscores = np.zeros((len(ch_types), 3))  # zscore of the correlation ch_type*amp/phase/cplx
    R2_all = np.zeros((len(ch_types), 3)) # ch_type*amp/phase/cplx/combined
    pval_all = np.zeros((len(ch_types), 3))
    SSR = np.zeros((len(ch_types), 3)) 
    for ch, ch_type in enumerate(ch_types):
        phases_simu, ampls_simu, times, matrices_simu[ch_type] = create_RSA_matrices(entry, ev_proj, ch_type, verbose)
        phases[ch_type] = np.stack((phases_meas[ch_type], phases_simu)) # meas/simu
        ampls[ch_type] = np.stack((ampls_meas[ch_type], ampls_simu))
        
        # Calculate correlations 
        Nchan = np.shape(matrices_meas[ch_type])[-1]
        msk_tri = np.triu(np.ones((Nchan, Nchan), bool), k=1)  # zero the diagonal and all values below
        
        # Correlate measured vs predicted matrix
        for i, data in enumerate(['amplitude', 'phase', 'complex']):
            meas = matrices_meas[ch_type][i][msk_tri]
            simu = matrices_simu[ch_type][i][msk_tri]
            
            if data == 'amplitude':
                R2, pval = scistats.spearmanr(meas, simu)
                zscores[ch,i] = 0.5 * np.log((1 + R2) / (1 - R2))  # fisher Z (ok for spearman when N>10)
                SSR[ch,i] = np.real(np.sum((meas-simu)**2))
            elif data == 'phase':
                R2, zscore, pval = circular_corr(meas, simu)
                zscores[ch,i] = zscore
                SSR[ch,i] = np.sum(circular_diff(meas,simu)**2)
            elif data == 'complex':
                R2 = np.abs(np.corrcoef(meas, simu))[1, 0]
                # fisher Z (ok for spearman when N>10)  Kanji 2006 as ref too
                zscores[ch,i] = 0.5 * np.log((1 + R2) / (1 - R2))  
                df = len(meas) - 2
                tscore = R2 * np.sqrt(df) / np.sqrt(1 - R2 * R2)  # Kanji 2006
                pval = scistats.t.sf(tscore, df) 
                
                # Calculate TS-SS distance metric between complex meas and simu
                TSSS[ch] = TS_SS(meas, simu)
                              
            # Store statistics    
            R2_all[ch,i] = R2 
            pval_all[ch,i] = pval  
        
        # Compute the SSR for complex values as the sum of SSR for phase and amp
        SSR[ch,2] = SSR[ch,0] + SSR[ch,1]     
    
    return phases, ampls, times, zscores, R2_all, pval_all, matrices_meas, matrices_simu, SSR, TSSS


# def compare_meas_simu(entry, ev_proj, ev_meas, verbose = False):
#     '''
#     Compare the relationships between pairs of sensors in terms of amplitude, phase
#     or complex values between measured and simulated signal.

#     Parameters
#     ----------
#     entry : class
#         Class for entry values
#     ev_proj : STRING
#         Path and file name for the simulated data file to compare (should contain an evoked).

#     Returns
#     -------
#     phases : ARRAY meas/simu*channels*time points
#         Instantaneous phase at 5 Hz for each signal, channel and time point.
#     ampls : ARRAY meas/simu*channels*time points
#         Instantaneous amplitude at 5 Hz for each signal, channel and time point.
#     times : ARRAY
#         Times of the evoked files.
#     info : INFO file
#         Info from evoked files.
#     zscores : ARRAY channels*amp/phase/cplx
#         Zscores of the comparison between measured vs simulated signal, for each
#         channel and each comparison type (amplitude, phase or complex).
#     R2_all : ARRAY channels*amp/phase/cplx
#         Correlation coefficient of the comparison between measured vs simulated
#         signal, for each channel and each comparison type (amplitude, phase or complex).
#     pval_all : ARRAY channels*amp/phase/cplx
#         P-value from the comparison between measured vs simulated
#         signal, for each channel and each comparison type (amplitude, phase or complex).
#     matrices_meas : DICT {channel type: amp/phase/cplx*chan*chan}
#         Amplitude/phase/complexe matrices for simulated data and each channel pair.
#     matrices_simu : DICT {channel type: amp/phase/cplx*chan*chan}
#         Amplitude/phase/complexe matrices for measured data and each channel pair.
#     SSR : DICT {channel type: amp/phase/cplx*N}
#         Sum of Squared Residuals (measured - simulated).       
#     '''

#     # initiate outputs
#     ch_types = ['mag', 'grad', 'eeg']
#     zscores = np.zeros((len(ch_types), 3))  # zscore of the correlation ch_type*amp/phase/cplx
#     R2_all = np.zeros((len(ch_types), 3))
#     pval_all = np.zeros((len(ch_types), 3))
#     SSR = np.zeros((len(ch_types), 3))
#     matrices_meas = {} ; matrices_simu = {}
#     phases = {}; ampls= {}
#     info = ev_meas.info
    
#     for ch, ch_type in enumerate(ch_types):
#         phases[ch_type], ampls[ch_type], times, zscores[ch], R2_all[ch], pval_all[ch], mat_meas, mat_simu, SSR[ch] = compare_meas_simu_oneChType(entry, ev_proj, ev_meas, ch_type, verbose)
#         matrices_meas[ch_type] = mat_meas
#         matrices_simu[ch_type] = mat_simu            
        
#     return phases, ampls, times, info, zscores, R2_all, pval_all, matrices_meas, matrices_simu, SSR
