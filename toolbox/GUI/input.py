import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mb
from tkinter import simpledialog

from toolbox.GUI import tools
from toolbox.utils import CONFIG_PATH, SIM_CONF, SCREEN_CONF
from pathlib import Path
from toolbox import configIO, utils, main
from configparser import ConfigParser


class MainFrame(ttk.Frame):

    def __init__(self, container):
        super(MainFrame, self).__init__(container)
        self.listbox = None
        self.config_file = tk.StringVar(self, str(Path(CONFIG_PATH).resolve()))
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
        self.listbox = tools.Listbox(list_frame, listvariable=tk.Variable(value=[]), height=10)
        self.listbox.bind('<Double-1>', self.edit_entry)
        self.listbox.pack(expand=True, side=tk.LEFT, fill=tk.BOTH)

        button_frame = ttk.Frame(list_frame)

        add_button = tk.Button(button_frame, text='ADD', command=self.entry_window)
        edit_button = tk.Button(button_frame, text='EDIT', command=self.edit_entry)
        remove_button = tk.Button(button_frame, text='REMOVE', command=self.remove_entry)
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
        filepath = tools.select_file_window(self, self.config_file)
        self.config_file.set(filepath)
        for entry in configIO.read_entry_config(filepath):
            self.listbox.insert(tk.END, entry)
            # except KeyError:
            #     mb.showerror('Error', 'You must select a valid Entry configuration file')
            #     break

    def save_config(self):
        file = tools.save_file_window(self, self.config_file)
        self.config_file.set(file)
        config_obj = ConfigParser()
        section_idx = 1
        entry_list = self.listbox.get_value_list()
        for entry in entry_list:
            while config_obj.has_section('entry' + str(section_idx)):
                section_idx += 1
            section = 'entry' + str(section_idx)
            config_obj.add_section(section)
            config_obj[section] = entry.create_dictionary()
            section_idx += 1
        configIO.write_config(config_obj, file)

    def save_config_frame(self):
        f = tools.show_file_path(self, 'Config file', self.config_file)
        f['padding'] = (10, 5)
        load_btn = tk.Button(f, text='LOAD', command=self.load_config_file)
        save_btn = tk.Button(f, text='SAVE', command=self.save_config)
        load_btn.pack(side=tk.LEFT)
        save_btn.pack(side=tk.LEFT)

    def run_simulation(self):
        main.run_pipeline(self.listbox.get_value_list())


class ConfigFrame(ttk.Frame):
    def __init__(self, parent, param, config_path):
        super(ConfigFrame, self).__init__(parent)
        self.path = Path(config_path)
        self.config_obj = configIO.load_config(config_path)
        self.list_items = tk.Variable(value=self.config_obj.sections())
        self.param = param
        self.param_class = param.__class__
        self.txtInput = []
        self.config_name = 'None'
        self.create()

    def add_text_inputs(self):
        for field in self.param._fields:
            f, txt = tools.add_text_input(self, field)
            self.txtInput.append(txt)

    def create(self):
        lbframe = ttk.Frame(self)
        listbox = tk.Listbox(lbframe, height=10, listvariable=self.list_items)
        listbox.pack(fill=tk.BOTH)
        listbox.bind('<Double-1>', self.load_selected_config)
        config_button = tk.Button(lbframe, text='Save parameters', command=self.save_config)
        config_button.pack(side=tk.BOTTOM)
        lbframe.pack(fill=tk.BOTH, side=tk.LEFT)
        self.add_text_inputs()
        self.load_params()
        self.pack(fill=tk.BOTH)

    def get_param(self):
        return self.param_class(*map(lambda x: x.get(), self.txtInput))

    def get_config(self):
        return self.config_name, self.get_param()

    def load_selected_config(self, event):
        widget = event.widget
        idx = widget.curselection()[0]
        section = self.list_items.get()[idx]
        self.load_config(section)

    def load_config(self, section):
        values = self.config_obj[section].values()
        self.param = self.param_class(*values)
        self.load_params()
        self.config_name = section

    def save_config(self):
        section_name = simpledialog.askstring("Config name", "Enter config name", parent=self)
        params = self.get_param()
        configIO.create_config_section(self.config_obj, params, section_name, self.path)
        self.list_items.set(self.config_obj.sections())
        self.config_name = section_name

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
        self.measuredStringVar = tk.StringVar(self, str(self.entry.measured))
        self.freesurferStringVar = tk.StringVar(self, str(self.entry.freesurfer))
        self.forwardStringVar = tk.StringVar(self, str(self.entry.fwd_model))
        self.stim_list_box = None
        self.space_list_box = None
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

        main_frame = ttk.Frame(notebk)
        main_frame['padding'] = (5, 10)
        main_frame.pack(fill=tk.BOTH)
        tools.add_file_input(main_frame, 'Measured data', self.measuredStringVar)
        tools.add_file_input(main_frame, 'Freesurfer folder', self.freesurferStringVar)
        tools.add_file_input(main_frame, 'Forward model', self.forwardStringVar)
        self.stim_list_box, _ = tools.add_dropdown_input(main_frame, 'Stimulation', main.stim_list, self.entry.stim)
        self.space_list_box, _ =\
            tools.add_dropdown_input(main_frame, 'Spacial condition', main.c_space_list, self.entry.c_space)
        for widget in main_frame.winfo_children():
            widget.pack(side=tk.TOP)

        self.simulation_frame = ConfigFrame(notebk, self.entry.simulation_params, SIM_CONF)
        self.screen_frame = ConfigFrame(notebk, self.entry.screen_params, SCREEN_CONF)

        notebk.add(main_frame, text='Params')
        notebk.add(self.simulation_frame, text='Simulation')
        notebk.add(self.screen_frame, text='Screen')

        self.saveButton.pack(side=tk.BOTTOM)

    def save_entry(self):
        self.entry.measured = Path(self.measuredStringVar.get())
        self.entry.freesurfer = Path(self.freesurferStringVar.get())
        self.entry.fwd_model = Path(self.forwardStringVar.get())
        self.entry.stim = self.stim_list_box.get()
        self.entry.c_space = self.space_list_box.get()
        self.entry.simulation_config_section, self.entry.simulation_params = self.simulation_frame.get_config()
        self.entry.screen_config_section, self.entry.screen_params = self.screen_frame.get_config()
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


def run_gui():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    run_gui()
