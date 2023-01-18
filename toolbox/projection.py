import mne
import numpy as np


def project_wave(entry, stc_gen):
    measured = entry.measured
    forward = entry.fwd_model

    info = mne.io.read_info(measured)
    fwd = mne.read_forward_solution(forward)

    # Project into sensor space
    nave = np.inf  # number of averaged epochs - controlled the amount of noise
    evoked_gen = mne.simulation.simulate_evoked(fwd, stc_gen, info, cov=None, nave=nave,
                                                random_state=42)
    return evoked_gen

