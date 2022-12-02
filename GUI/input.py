import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mb
from tkinter import simpledialog
import tools
import utils
from utils import CONFIG_PATH, SIM_CONF, SCREEN_CONF
from pathlib import Path
from toolbox import configIO


class MainFrame(ttk.Frame):

    def __init__(self, container):
        super(MainFrame, self).__init__(container)
        self.listbox = None
        self.list_items = tk.Variable(value=[])
        self.save_file = tk.StringVar(self)
        self.config_file = tk.StringVar(self, Path('/'))
        self.main_window()

    def main_window(self):
        header = tk.Label(self, text='Add entries to generate simulations for given .fif file')
        header.pack(fill=tk.BOTH, expand=True)
        self.create_list_frame()
        self.save_config_frame()

    def create_list_frame(self):
        # create list for entries
        list_frame = ttk.Frame(self)
        list_frame['padding'] = (5, 10)
        list_frame.pack(expand=True, fill=tk.BOTH)
        self.listbox = tools.Listbox(list_frame, listvariable=self.list_items, height=10)
        self.listbox.bind('<Double-1>', self.edit_entry)
        self.listbox.pack(expand=True, side=tk.LEFT, fill=tk.BOTH)

        button_frame = ttk.Frame(list_frame)

        add_button = tk.Button(button_frame, text='ADD', command=self.entry_window)
        edit_button = tk.Button(button_frame, text='EDIT', command=self.edit_entry)
        remove_button = tk.Button(button_frame, text='REMOVE', command=self.remove_entry)
        loadconf_button = tk.Button(button_frame, text='LOAD CONFIG', command=self.load_config_file)
        for widget in button_frame.winfo_children():
            widget.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        run_button = tk.Button(button_frame, text='RUN', command=self.run_simulation)
        run_button.pack(side=tk.BOTTOM, fill=tk.X, padx=5)

        button_frame.pack(side=tk.LEFT, fill=tk.BOTH)

    def entry_window(self, entry=None):
        entryWin = EntryWindow(self, entry)
        entry = entryWin.get_entry()
        if entry is not None:
            self.listbox.insert(tk.END, entry)

    def edit_entry(self, event=None):
        idx = self.listbox.curselection()
        if not idx:
            mb.showerror('Error', 'You must select an entry to edit')
        else:
            idx = idx[0]
            entry = self.listbox.get_value(idx)
            self.entry_window(entry)
            self.listbox.delete(idx)

    def remove_entry(self):
        idx = self.listbox.curselection()
        if not idx:
            mb.showerror('Error', 'You must select an entry to delete')
        elif mb.askyesno('Verify', 'Are you sure you want to delete the selected entry?'):
            idx = idx[0]
            self.listbox.delete(idx)

    def load_config_file(self):
        filepath = tools.save_file(self, self.config_file)
        self.config_file.set(filepath)
        config_obj = configIO.get_config_object(filepath)
        sim_obj = configIO.get_config_object(SIM_CONF)
        screen_obj = configIO.get_config_object(SCREEN_CONF)
        entries_list = []
        for section in config_obj.sections():
            sim_section = config_obj[section]['simulation']
            screen_section = config_obj[section]['screen']
            sim_vals = sim_obj[sim_section].values()
            screen_vals = sim_obj[screen_section].values()
            entry = utils.Entry()
            entry.set_simulation_params(sim_vals)
            entry.set_screen_params(screen_vals)
            entries_list.append(entry)


    def save_config_frame(self):
        save_file = tk.StringVar()
        f = tools.add_file_input(self, 'Save location', save_file, tools.save_file)
        f['padding'] = (10, 5 )
        btn = tk.Button(f, text='SAVE', command=self.save_config)
        btn.pack(side=tk.LEFT)

    def save_config(self):
        return # TODO save the entries in a file

    def run_simulation(self):
        return


class ConfigFrame(ttk.Frame):
    def __init__(self, parent, param, config_path):
        super(ConfigFrame, self).__init__(parent)
        self.path = Path(config_path)
        self.config_obj = configIO.get_config_object(config_path)
        self.list_items = tk.Variable(value=self.config_obj.sections())
        self.param = param
        self.param_class = param.__class__
        self.txtInput = []
        self.create()

    def create(self):
        lbframe = ttk.Frame(self)
        listbox = tk.Listbox(lbframe, height=10, listvariable=self.list_items)
        listbox.pack(fill=tk.BOTH)
        listbox.bind('<Double-1>', self.load_configs)
        config_button = tk.Button(lbframe, text='Save parameters', command=self.save_config)
        config_button.pack(side=tk.BOTTOM)
        lbframe.pack(fill=tk.BOTH, side=tk.LEFT)
        self.add_text_inputs()
        self.load_params()
        self.pack(fill=tk.BOTH)

    def get_param(self):
        return self.param_class(*map(lambda x: x.get(), self.txtInput))

    def save_config(self):
        name = simpledialog.askstring("Config name", "Enter config name", parent=self)
        params = self.get_param()
        configIO.create_config_file(self.config_obj, params, name, self.path)
        self.list_items.set(self.config_obj.sections())

    def load_configs(self, event):
        widget = event.widget
        idx = widget.curselection()[0]
        sel = self.list_items.get()[idx]
        values = self.config_obj[sel].values()
        self.param = self.param_class(*values)
        self.load_params()

    def add_text_inputs(self):
        for field in self.param._fields:
            f = ttk.Frame(self)
            lbl = tk.Label(f, text=field)
            txt = ttk.Entry(f)
            self.txtInput.append(txt)
            lbl.pack(side=tk.LEFT)
            txt.pack(side=tk.LEFT, fill=tk.BOTH)
            f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def load_params(self):
        if self.param is None:
            return
        for i in range(len(self.param)):
            self.txtInput[i].delete(0, tk.END)
            self.txtInput[i].insert(0, self.param[i])


class EntryWindow(tk.Toplevel):
    def __init__(self, parent, entry):
        super(EntryWindow, self).__init__(parent)
        self.grab_set()
        self.new = entry is None
        self.entry = utils.Entry() if self.new else entry
        self.measuredStringVar = tk.StringVar(self, self.entry.measured)
        self.retinoStringVar = tk.StringVar(self, self.entry.retino_map)
        self.simulation_frame = None
        self.screen_frame = None

        self.txtInputs = {}
        self.saveButton = tk.Button(self, text='SAVE', command=self.save_entry)
        self.window_init()

    def window_init(self):
        title_base = 'Add new' if self.new else 'Edit'
        self.title(f'{title_base} entry')
        self.attributes('-topmost', 1)

        notebk = ttk.Notebook(self)
        notebk.pack(fill=tk.BOTH)

        f = ttk.Frame(notebk)
        f['padding'] = (5, 10)
        f.pack(fill=tk.BOTH)
        tools.add_file_input(f, 'Measured data', self.measuredStringVar, tools.select_file)

        self.simulation_frame = ConfigFrame(notebk, self.entry.simulation_params, SIM_CONF)
        self.screen_frame = ConfigFrame(notebk, self.entry.screen_params, SCREEN_CONF)

        mri_frame = ttk.Frame(notebk)
        tools.add_file_input(mri_frame, 'Retinotopic map MRI', self.retinoStringVar, tools.select_file)
        mri_frame.pack(fill=tk.BOTH)

        notebk.add(f, text='First version')
        notebk.add(self.simulation_frame, text='Simulation')
        notebk.add(self.screen_frame, text='Screen')
        notebk.add(mri_frame, text='MRI')

        self.saveButton.pack(side=tk.BOTTOM)

    def save_entry(self):
        self.entry.simulation_params = self.simulation_frame.get_param()
        self.entry.screen_params = self.screen_frame.get_param()
        self.entry.measured = Path(self.measuredStringVar.get())
        self.entry.retino_map = Path(self.retinoStringVar.get())
        self.destroy()

    def get_entry(self):
        self.close_window()
        return self.entry

    def close_window(self):
        self.deiconify()
        self.wm_protocol("WN_DELETE_WINDOW", self.destroy)
        self.wait_window(self)


class App(tk.Tk):
    def __init__(self):
        super(App, self).__init__()
        self.title('WAVES toolbox')
        self.geometry("700x350")

        gui = MainFrame(self)
        gui.pack(fill=tk.BOTH, expand=True)


def run():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    run()
