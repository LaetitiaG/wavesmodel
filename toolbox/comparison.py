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
    matrices : DICT {channel type: amp/phase/cplx*meas/pred*chan*chan}
        Amplitude/phase/complexe matrices for each signal and each channel pair.

    '''
    fstim = entry.simulation_params.freq_temp
    # Initialize constants
    # Parameters for Morlet analysis for the SSVEP extraction
    freqs = np.arange(2., 50., 0.5)
    n_cycles = freqs / 2.
    ch_types = ['mag', 'grad', 'eeg']
    choose_ch = [['mag', False], ['grad', False], [False, True]]
    tmin_crop = 0.5

    # initiate outputs
    zscores = np.zeros((len(ch_types), 3))  # zscore of the correlation ch_type*amp/phase/cplx
    R2_all = np.zeros((len(ch_types), 3))
    pval_all = np.zeros((len(ch_types), 3))
    matrices = {'eeg': np.zeros((3, 2, 74, 74), dtype='complex_'),  # amp/phase/cplx *meas/pred*chan*chan
                'mag': np.zeros((3, 2, 102, 102), dtype='complex_'),
                'grad': np.zeros((3, 2, 102, 102),
                                 dtype='complex_')}  # complex connectivity matrices for each channel type

    # Read measured data evoked
    evoked = mne.read_evokeds(entry.measured)[0]
    evoked.crop(tmin=0, tmax=2)
    Nt = len(evoked.times)
    sf = evoked.info['sfreq']

    # Read projections
    evokeds = [evoked, ev_proj]  # 'measured', 'simulated'

    # Extract phase and amplitude at each time point at 5Hz
    phases = np.zeros((len(evokeds), np.shape(evoked.data)[0], Nt))  # train, test proj
    ampls = np.zeros((len(evokeds), np.shape(evoked.data)[0], Nt))
    for e, ev in enumerate(evokeds):
        # extract instantaneous phase and amplitude
        dat = ev.data[np.newaxis, :]
        tfr = mne.time_frequency.tfr_array_morlet(dat, sfreq=sf, freqs=freqs, n_cycles=n_cycles, output='complex')
        phase = np.angle(tfr)
        amp = np.abs(tfr)

        # extract instantaneous phase and amplitude at frequency of interest
        phases[e] = np.squeeze(phase[0, :, freqs == fstim])
        ampls[e] = np.squeeze(amp[0, :, freqs == fstim])

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
            cov_tmp = np.zeros((2, len(evokeds), len(inds1), len(inds2)))
            for j in range(len(ind_grad)):
                inds = ind_grad[j]
                for i in range(len(evokeds)):  # for measured and predicted data
                    ampls_ch = ampls[i, inds][:, msk_t]  # ampls[i,inds,msk_t]
                    # loop across rows
                    for r in range(len(inds)):
                        for c in range(len(inds)):  # loop across column
                            cov_tmp[j, i, r, c] = np.mean(np.log(ampls_ch[r] / ampls_ch[c]))
            cov_ch = cov_tmp.mean(0)
        else:
            inds = mne.pick_types(evoked.info, meg=choose_ch[ch][0], eeg=choose_ch[ch][1])
            cov_ch = np.zeros((len(evokeds), len(inds), len(inds)))

            for i in range(len(evokeds)):  # for measured and predicted data
                ampls_ch = ampls[i, inds][:, msk_t]
                # loop across rows
                for r in range(len(inds)):
                    for c in range(len(inds)):  # loop across column
                        # average across time and take the log to have a symmetrical matrix
                        cov_ch[i, r, c] = np.mean(np.log(ampls_ch[r] / ampls_ch[c]))  

        cov_amp.append(cov_ch)

        # Correlate measured vs predicted matrix
        msk_tri = np.triu(np.ones((len(inds), len(inds)), bool), k=1)  # zero the diagonal and all values below
        meas = cov_ch[0][msk_tri]
        simu = cov_ch[1][msk_tri]
        R2, pval = scistats.spearmanr(meas, simu)
        zscores[ch, 0] = 0.5 * np.log((1 + R2) / (1 - R2))  # fisher Z (ok for spearman when N>10)
        R2_all[ch, 0] = R2
        pval_all[ch, 0] = pval

        ###### PHASE ######
    # Compute a matrix of relative phase between sensors for each sensor type
    cov_phase = []
    for ch, ch_type in enumerate(ch_types):
        if ch_type == 'grad':
            # do the average of relative amplitude between grad1 and grad2
            inds1 = mne.pick_types(evoked.info, meg='planar1', eeg=False)
            inds2 = mne.pick_types(evoked.info, meg='planar2', eeg=False)
            inds = inds2
            cov_ch = np.zeros((len(evokeds), len(inds), len(inds)))
            for i in range(len(evokeds)):  # for measured and predicted data
                phases_1 = phases[i, inds1][:, msk_t]
                phases_2 = phases[i, inds2][:, msk_t]
                # loop across rows
                for r in range(len(inds)):
                    for c in range(len(inds)):  # loop across column
                        d1 = phases_1[r] - phases_1[c]
                        d11 = (d1 + np.pi) % (2 * np.pi) - np.pi  # put between -pi and pi

                        d2 = phases_2[r] - phases_2[c]
                        d22 = (d2 + np.pi) % (2 * np.pi) - np.pi  # put between -pi and pi

                        cov_ch[i, r, c] = circular_mean(np.array([circular_mean(d11), circular_mean(d22)]))

        else:
            inds = mne.pick_types(evoked.info, meg=choose_ch[ch][0], eeg=choose_ch[ch][1])
            cov_ch = np.zeros((len(evokeds), len(inds), len(inds)))

            for i in range(len(evokeds)):  # for measured and predicted data
                phases_ch = phases[i, inds][:, msk_t]
                # loop across rows
                for r in range(len(inds)):
                    for c in range(len(inds)):  # loop across column
                        d1 = phases_ch[r] - phases_ch[c]
                        d = (d1 + np.pi) % (2 * np.pi) - np.pi  # put between -pi and pi
                        cov_ch[i, r, c] = circular_mean(d)

        cov_phase.append(cov_ch)

        # Correlate measured vs predicted matrix
        msk_tri = np.triu(np.ones((len(inds), len(inds)), bool), k=1)  # zero the diagonal and all values below
        meas = cov_ch[0][msk_tri]
        simu = cov_ch[1][msk_tri]
        R2, zscore, pval = circular_corr(meas, simu)
        zscores[ch, 1] = zscore
        R2_all[ch, 1] = R2
        pval_all[ch, 1] = pval

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

        # Correlate matrix using conjugate
        msk_tri = np.triu(np.ones((len(inds), len(inds)), bool), k=1)  # zero the diagonal and all values below
        meas = covX[0][msk_tri]
        simu = covX[1][msk_tri]

        # amplitude of the complex correlation coefficient
        R2 = np.abs(np.corrcoef(meas, simu))[1, 0]
        zscores[ch, 2] = 0.5 * np.log(
            (1 + R2) / (1 - R2))  # fisher Z (ok for spearman when N>10)  Kanji 2006 as ref too
        df = len(meas) - 2
        tscore = R2 * np.sqrt(df) / np.sqrt(1 - R2 * R2)  # Kanji 2006
        pval = scistats.t.sf(tscore, df)
        pval_all[ch, 2] = pval
        R2_all[ch, 2] = R2

    return phases, ampls, times, evoked.info, zscores, R2_all, pval_all, matrices
