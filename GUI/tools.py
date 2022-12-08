import tkinter as tk
from tkinter import filedialog as fd
from tkinter import ttk
from pathlib import Path

config_filetypes = (
    ('config files', '*.txt'),
    ('config files', '*.ini'),
    ('All files', '*.*')
)


def select_file_window(mainframe, dr=None):
    f = fd.askopenfilename(
        parent=mainframe,
        title='Open a config file',
        initialdir='/' if dr is None else dr,
        filetypes=config_filetypes)
    return Path(f)


def save_file_window(mainframe, dr=None):
    f = fd.asksaveasfilename(
        parent=mainframe,
        title='Save your file',
        initialdir='/' if dr is None else dr,
        filetypes=config_filetypes)
    return Path(f)


def show_file_path(mainFrame, txt, var):
    f = ttk.Frame(mainFrame)
    lbl = tk.Label(f, text=txt)
    lbl.pack(side=tk.LEFT)
    lbl = tk.Label(f, textvariable=var, bg='white')
    lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
    f.pack(side=tk.TOP, fill=tk.X, expand=True)
    return f


def add_file_input(mainFrame, txt, var, cmd):
    f = show_file_path(mainFrame, txt, var)
    load_button = tk.Button(f, text='Browse', command=lambda: var.set(cmd(mainFrame, txt)))
    load_button.pack(side=tk.LEFT)
    return f


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

    def get_value_list(self):
        return self.value

