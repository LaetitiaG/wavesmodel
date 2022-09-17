from collections import namedtuple  # available in Python 2.6+
import tkinter as tk
from tkinter import filedialog as fd
from dataclasses import dataclass
from pathlib import Path

simulation_params = namedtuple("simulation_params",
                               ["freq_temp", "freq_spacial", "amplitude", "phase_offset"])


@dataclass
class Entry:
    """Data class corresponding to the entry structure"""
    measured: Path = Path('/')
    retino_map: Path = Path('/')
    params: simulation_params = None

    def set_params(self, param_list):
        self.params = simulation_params(*param_list)


def select_file(dir=None):
    filetypes = (
        ('text files', '*.txt'),
        ('All files', '*.*')
    )
    return fd.askopenfilename(
        title='Open a config file',
        initialdir='/' if dir is None else dir,
        filetypes=filetypes)

def save_file(dir=None):
    return fd.asksaveasfilename(
        title='Open a config file',
        initialdir='/' if dir is None else dir)

class Listbox(tk.Listbox):
    def __init__(self, master=None, cnf={}, **kw):
        super(Listbox, self).__init__(master, cnf, **kw)
        self.value = []

    def insert(self, index, *elements):
        super(Listbox, self).insert(index, elements)
        self.value += list(elements)

    def delete(self, first, last=None):
        super(Listbox, self).delete(first, last)
        if last:
            while first < last:
                self.value.pop(first)
                first += 1
        else:
            self.value.pop(first)

    def get_value(self, idx):
        return self.value[idx]


def add_file_input(mainFrame, txt, var, cmd):
    f = tk.Frame(mainFrame)
    lbl = tk.Label(f, text=txt)
    lbl.pack(side=tk.LEFT)
    lbl = tk.Label(f, textvariable=var, bg='white')
    lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
    load_button = tk.Button(f, text='Browse', command=cmd)
    load_button.pack(side=tk.LEFT)
    f.pack(side=tk.TOP, fill=tk.X, expand=True)
    return f
