import matplotlib.pyplot as plt
from mne.report import Report


# topomap scale
ch_types = ['mag', 'grad', 'eeg']
scale = {'mag': 1200, 'grad': 400, 'eeg': 40}
chan_toplot = {'mag': ['MEG2111'], 'grad': ['MEG2112'], 'eeg': ['EEG070']}
col_cond = {'trav': 'crimson', 'STANDING': 'dodgerblue', 'TRAV_OUT': 'crimson', 'TRAV_IN': 'darkorange',
            'fov_out': 'darkmagenta'}


def plot_projection(entry, evoked_gen, report_path):
    c_wave = entry.stim
    c_space = entry.c_space

    report = Report(verbose=True)
    f = evoked_gen.plot()
    report.add_figure(f, title='evoked_gen before rescaling', section=c_wave + ' ' + c_space)

    # evoked_gen.save(os.path.join(simupath, sname))

    # Plot evoked
    # mag and grad
    f = evoked_gen.plot_topo()
    report.add_figure(f, title='topo meg ', section=c_wave + ' ' + c_space)

    # topo EEG
    tmp = evoked_gen.copy().pick_types(meg=False, eeg=True)
    f = tmp.plot_topo()
    report.add_figure(f, title='topo eeg ', section=c_wave + ' ' + c_space)

    for chan in ch_types:
        # f = evoked_gen.plot_topomap(times = [0.05, 0.1, 0.15, 0.20, 0.25], ch_type = chan, vmax = scale[chan], vmin = -scale[chan])
        f = evoked_gen.plot_topomap(times=[0.0, 0.60, 0.120], ch_type=chan, vmax=scale[chan], vmin=-scale[chan])
        report.add_figure(f, title=chan + ' ', section=c_wave + ' ' + c_space)

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
    report.add_figure(f, title='timecourse at selected sensor', section='bothcond ' + c_space)

    report.save(report_path, overwrite=True)
