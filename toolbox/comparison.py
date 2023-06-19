import numpy as np
import mne
import scipy.stats as scistats
from math import sqrt, atan2, pi, floor, exp


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

    return rho, zcorr, p_value


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


def create_RSA_matrices(entry, evoked):
    """
    Create RSA matrices for phase and amplitude relationship between each pair
    of sensors

    Parameters
    ----------
    entry : class
        Class for entry values
    evoked : class
        Evoked data.

    Returns
    -------
    phases : ARRAY meas/simu*channels*time points
        Instantaneous phase at 5 Hz for each signal, channel and time point.
    ampls : ARRAY meas/simu*channels*time points
        Instantaneous amplitude at 5 Hz for each signal, channel and time point.
    times : ARRAY
        Times of the evoked files.    
    matrices : DICT
        Dictionary with 'mag', 'grad' and 'eeg' keys with the three matrices
        (amplitude, phase, complex).

    """
    fstim = entry.simulation_params.freq_temp
    # Initialize constants
    # Parameters for Morlet analysis for the SSVEP extraction
    freqs = np.arange(2., 50., 0.5)
    n_cycles = freqs / 2.
    ch_types = ['mag', 'grad', 'eeg']
    choose_ch = [['mag', False], ['grad', False], [False, True]]
    tmin_crop = 0.5
    
    Nt = len(evoked.times)
    sf = evoked.info['sfreq']
    # initiate outputs
    matrices = {'eeg': np.zeros((3, 74, 74), dtype='complex_'),  # amp/phase/cplx *meas/pred*chan*chan
                'mag': np.zeros((3, 102, 102), dtype='complex_'),
                'grad': np.zeros((3, 102, 102),
                                 dtype='complex_')}  # complex connectivity matrices for each channel type


    # Extract phase and amplitude at each time point at 5Hz
    dat = evoked.data[np.newaxis, :]
    tfr = mne.time_frequency.tfr_array_morlet(dat, sfreq=sf, freqs=freqs, n_cycles=n_cycles, output='complex')
    phase = np.angle(tfr)
    amp = np.abs(tfr)

    # extract instantaneous phase and amplitude at frequency of interest
    phases = np.squeeze(phase[0, :, freqs == fstim])
    ampls = np.squeeze(amp[0, :, freqs == fstim])

    times = evoked.times
    msk_t = times > tmin_crop

    ###### AMPLITUDE ######
    # Compute a matrix of relative amplitude between sensors for each sensor type
    cov_amp = []  # log matrices for each sensor type
    for ch, ch_type in enumerate(ch_types):

        if ch_type == 'grad':
            # do the average of relative amplitude between grad1 and grad2
            inds1 = mne.pick_types(evoked.info, meg='planar1', eeg=False)
            inds2 = mne.pick_types(evoked.info, meg='planar2', eeg=False)
            ind_grad = [inds1, inds2]
            cov_tmp = np.zeros((2, len(inds1), len(inds2)))
            for j in range(len(ind_grad)):
                inds = ind_grad[j]
                ampls_ch = ampls[inds][:, msk_t]
                # loop across rows
                for r in range(len(inds)):
                    for c in range(len(inds)):  # loop across column
                        cov_tmp[j, r, c] = np.mean(np.log(ampls_ch[r] / ampls_ch[c]))
            cov_ch = cov_tmp.mean(0)
        else:
            inds = mne.pick_types(evoked.info, meg=choose_ch[ch][0], eeg=choose_ch[ch][1])
            cov_ch = np.zeros((len(inds), len(inds)))

            ampls_ch = ampls[inds][:, msk_t]
            # loop across rows
            for r in range(len(inds)):
                for c in range(len(inds)):  # loop across column
                    # average across time and take the log to have a symmetrical matrix
                    cov_ch[r, c] = np.mean(np.log(ampls_ch[r] / ampls_ch[c]))  

        cov_amp.append(cov_ch)

    ###### PHASE ######
    # Compute a matrix of relative phase between sensors for each sensor type
    cov_phase = []
    for ch, ch_type in enumerate(ch_types):
        if ch_type == 'grad':
            # do the average of relative amplitude between grad1 and grad2
            inds1 = mne.pick_types(evoked.info, meg='planar1', eeg=False)
            inds2 = mne.pick_types(evoked.info, meg='planar2', eeg=False)
            inds = inds2
            phases_1 = phases[inds1][:, msk_t]
            phases_2 = phases[inds2][:, msk_t]
            # loop across rows
            cov_ch = np.zeros((len(inds), len(inds)))
            for r in range(len(inds)):
                for c in range(len(inds)):  # loop across column
                    d1 = phases_1[r] - phases_1[c]
                    d11 = (d1 + np.pi) % (2 * np.pi) - np.pi  # put between -pi and pi

                    d2 = phases_2[r] - phases_2[c]
                    d22 = (d2 + np.pi) % (2 * np.pi) - np.pi  # put between -pi and pi

                    cov_ch[r, c] = circular_mean(np.array([circular_mean(d11), circular_mean(d22)]))

        else:
            inds = mne.pick_types(evoked.info, meg=choose_ch[ch][0], eeg=choose_ch[ch][1])
            cov_ch = np.zeros((len(inds), len(inds)))

            phases_ch = phases[inds][:, msk_t]
            # loop across rows
            for r in range(len(inds)):
                for c in range(len(inds)):  # loop across column
                    d1 = phases_ch[r] - phases_ch[c]
                    d = (d1 + np.pi) % (2 * np.pi) - np.pi  # put between -pi and pi
                    cov_ch[r, c] = circular_mean(d)

        cov_phase.append(cov_ch)

    ###### CPLEX NBER ######
    # Compute a complex matrix combining amplitude and phase
    for ch, ch_type in enumerate(ch_types):
        if ch_type == 'grad':
            inds = mne.pick_types(evoked.info, meg='planar1', eeg=choose_ch[ch][1])
        else:
            inds = mne.pick_types(evoked.info, meg=choose_ch[ch][0], eeg=choose_ch[ch][1])

        # Complex matrix
        covX = cov_amp[ch] * np.exp(cov_phase[ch] * 1j)
        matrices[ch_type][0] = cov_amp[ch]
        matrices[ch_type][1] = cov_phase[ch]
        matrices[ch_type][2] = covX

    return phases, ampls, times,matrices    

def compare_meas_simu(entry, ev_proj):
    '''
    Compare the relationships between pairs of sensors in terms of amplitude, phase
    or complex values between measured and simulated signal.

    Parameters
    ----------
    entry : class
        Class for entry values
    ev_proj : STRING
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
    matrices_simu : DICT {channel type: amp/phase/cplx*chan*chan}
        Amplitude/phase/complexe matrices for simulated data and each channel pair.
    matrices_simu : DICT {channel type: amp/phase/cplx*chan*chan}
        Amplitude/phase/complexe matrices for measured data and each channel pair.
    '''

    # initiate outputs
    ch_types = ['mag', 'grad', 'eeg']
    zscores = np.zeros((len(ch_types), 3))  # zscore of the correlation ch_type*amp/phase/cplx
    R2_all = np.zeros((len(ch_types), 3))
    pval_all = np.zeros((len(ch_types), 3))

    # Read measured data evoked
    evoked = mne.read_evokeds(entry.measured)[0]
    evoked.crop(tmin=0, tmax=2)

    # Calculate RSA matrices
    phases_meas, ampls_meas, times, matrices_meas = create_RSA_matrices(entry, evoked)
    phases_simu, ampls_simu, times, matrices_simu = create_RSA_matrices(entry, ev_proj)
    phases = np.stack(phases_meas, phases_simu) # meas/simu
    ampls = np.stack(ampls_meas, ampls_simu)
    
    # Calculate correlations
    for ch, ch_type in enumerate(ch_types):
        mat_meas = matrices_meas[ch_type]
        mat_simu = matrices_simu[ch_type]
        n_ch = np.shape(mat_meas)[-1]
        msk_tri = np.triu(np.ones((n_ch, n_ch), bool), k=1)  # zero the diagonal and all values below
        
        # Correlate measured vs predicted matrix
        for i, data in enumerate(['amplitude', 'phase', 'complex']):
            meas = mat_meas[i][msk_tri]
            simu = mat_simu[i][msk_tri]
            
            if data == 'amplitude':
                R2, pval = scistats.spearmanr(meas, simu)
                zscores[ch, i] = 0.5 * np.log((1 + R2) / (1 - R2))  # fisher Z (ok for spearman when N>10)
            elif data == 'phase':
                R2, zscore, pval = circular_corr(meas, simu)
                zscores[ch, i] = zscore
            elif data == 'complex':
                R2 = np.abs(np.corrcoef(meas, simu))[1, 0]
                # fisher Z (ok for spearman when N>10)  Kanji 2006 as ref too
                zscores[ch,i] = 0.5 * np.log((1 + R2) / (1 - R2))  
                df = len(meas) - 2
                tscore = R2 * np.sqrt(df) / np.sqrt(1 - R2 * R2)  # Kanji 2006
                pval = scistats.t.sf(tscore, df)                
                              
            # Store statistics    
            R2_all[ch, i] = R2
            pval_all[ch, i] = pval    
        
    return phases, ampls, times, evoked.info, zscores, R2_all, pval_all, matrices_meas, matrices_simu
