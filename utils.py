from collections import namedtuple  # available in Python 2.6+
from tkinter import filedialog as fd


simulation_params = namedtuple("simulation_params",
                               ["freq_temp", "freq_spacial", "amplitude", "phase_offset"])


def select_file(dir):
    filetypes = (
        ('text files', '*.txt'),
        ('All files', '*.*')
    )
    return fd.askopenfilename(
        title='Open a config file',
        initialdir='/' if dir is None else dir,
        filetypes=filetypes)
