import mne
import numpy as np
import matplotlib.pyplot as plt

# topomap scale
ch_types = ['mag', 'grad', 'eeg']
scale = {'mag': 1200, 'grad': 400, 'eeg': 40}
chan_toplot = {'mag': ['MEG2111'], 'grad': ['MEG2112'], 'eeg': ['EEG070']}
col_cond = {'trav': 'crimson', 'stand': 'dodgerblue', 'trav_out': 'crimson', 'trav_in': 'darkorange',
            'fov_out': 'darkmagenta'}


def project_wave(measured, forward, c_space, c_wave, stc_gen):
    info = mne.io.read_info(measured)
    fwd = mne.read_forward_solution(forward)
    report = mne.Report(verbose=True)

    # Project into sensor space
    nave = np.inf  # number of averaged epochs - controlled the amount of noise
    evoked_gen = mne.simulation.simulate_evoked(fwd, stc_gen, info, cov=None, nave=nave,
                                                random_state=42)
    f = evoked_gen.plot()
    report.add_figs_to_section(f, captions='evoked_gen before rescaling', section=c_wave + ' ' + c_space)

    # evoked_gen.save(os.path.join(simupath, sname))

    # Plot evoked
    # mag and grad
    f = evoked_gen.plot_topo()
    report.add_figs_to_section(f, captions='topo meg ', section=c_wave + ' ' + c_space)

    # topo EEG
    tmp = evoked_gen.copy().pick_types(meg=False, eeg=True)
    f = tmp.plot_topo()
    report.add_figs_to_section(f, captions='topo eeg ', section=c_wave + ' ' + c_space)

    for chan in ch_types:
        # f = evoked_gen.plot_topomap(times = [0.05, 0.1, 0.15, 0.20, 0.25], ch_type = chan, vmax = scale[chan], vmin = -scale[chan])
        f = evoked_gen.plot_topomap(times=[0.0, 0.60, 0.120], ch_type=chan, vmax=scale[chan], vmin=-scale[chan])
        report.add_figs_to_section(f, captions=chan + ' ', section=c_wave + ' ' + c_space)

    # time course for a chosen sensor
    times = evoked_gen.times
    f, ax = plt.subplots(figsize=(10, 9))
    for ch, ch_type in enumerate(ch_types):
        ax = plt.subplot(3, 1, ch + 1)
        plt.plot(times, evoked_gen.copy().pick_channels(chan_toplot[ch_type]).data[0], col_cond[c_wave])
        plt.hlines(y=0, xmin=-0.2, xmax=2, color='k', linestyles='dotted')
        plt.axvline(x=0, color='k', linestyle='dotted')
        plt.xlim((-0.2, 2))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.ylabel(chan_toplot[ch_type][0] + '-' + c_space)
        if ch == 2: plt.legend(c_wave, frameon=False); plt.xlabel('time (s)')
    report.add_figs_to_section(f, captions='timecourse at selected sensor', section='bothcond ' + c_space)
