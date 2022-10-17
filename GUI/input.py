import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mb
import tools
import utils
from pathlib import Path


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

    def save_config_frame(self):
        save_file = tk.StringVar()
        f = tools.add_file_input(self, 'Save location', save_file, tools.save_file)
        btn = tk.Button(f, text='SAVE', command=self.save_config)
        btn.pack(side=tk.LEFT)

    def save_config(self):
        return # TODO save the entries in a file

    def create_list_frame(self):
        # create list for entries
        list_frame = ttk.Frame(self)
        list_frame['padding'] = (5, 10)
        list_frame.pack(expand=True, fill=tk.BOTH)
        self.listbox = tools.Listbox(list_frame, listvariable=self.list_items, height=10)
        self.listbox.pack(expand=True, side=tk.LEFT, fill=tk.BOTH)

        button_frame = tk.Frame(list_frame)
        button_frame.pack(side=tk.LEFT, fill=tk.BOTH)

        add_button = tk.Button(button_frame, text='ADD', command=self.entry_window)
        add_button.pack(side=tk.TOP)
        edit_button = tk.Button(button_frame, text='EDIT', command=self.edit_entry)
        edit_button.pack(side=tk.TOP)
        remove_button = tk.Button(button_frame, text='REMOVE', command=self.remove_entry)
        remove_button.pack(side=tk.TOP)
        loadconf_button = tk.Button(button_frame, text='LOAD CONFIG', command=self.load_config_file)
        loadconf_button.pack(side=tk.TOP)

    def edit_entry(self):
        idx = self.listbox.curselection()
        if not idx:
            mb.showerror('Error', 'You must select an entry to edit')
        else:
            idx = idx[0]
            entry = self.listbox.get_value(idx)
            self.entry_window(entry)
            self.listbox.delete(idx)

    def entry_window(self, entry=None):
        entryWin = EntryWindow(self, entry)
        entry = entryWin.get_entry()
        if entry is not None:
            self.listbox.insert(tk.END, entry)

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


class EntryWindow(tk.Toplevel):

    def __init__(self, parent, entry):
        super(EntryWindow, self).__init__(parent)
        self.new = entry is None
        self.entry = utils.Entry() if self.new else entry
        self.measuredStringVar = tk.StringVar(self, self.entry.measured)
        self.retinoStringVar = tk.StringVar(self, self.entry.retino_map)

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

        simulation_frame = ttk.Frame(notebk)
        self.add_text_inputs(simulation_frame, utils.simulation_params)
        simulation_frame.pack(fill=tk.BOTH)

        screen_frame = ttk.Frame(notebk)
        self.add_text_inputs(screen_frame, utils.screen_params)
        screen_frame.pack(fill=tk.BOTH)

        mri_frame = ttk.Frame(notebk)
        tools.add_file_input(mri_frame, 'Retinotopic map MRI', self.retinoStringVar, tools.select_file)
        mri_frame.pack(fill=tk.BOTH)

        notebk.add(f, text='First version')
        notebk.add(simulation_frame, text='Simulation')
        notebk.add(screen_frame, text='Screen')
        notebk.add(mri_frame, text='MRI')

        self.saveButton.pack(side=tk.BOTTOM)
        if not self.new:
            self.load_entry()

    def add_text_inputs(self, mainFrame, params):
        ipt = []
        for field in params._fields:
            f = tk.Frame(mainFrame)
            lbl = tk.Label(f, text=field)
            txt = ttk.Entry(f)
            ipt.append(txt)
            lbl.pack(side=tk.LEFT)
            txt.pack(side=tk.LEFT, fill=tk.BOTH)
            f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.txtInputs[params.__name__] = ipt

    def load_entry(self):
        self.__load_entry_param(self.entry.simulation_params)
        self.__load_entry_param(self.entry.screen_params)

    def __load_entry_param(self, param):
        if param is None:
            return
        sim_input = self.txtInputs[param.__class__.__name__]
        for i in range(len(param)):
            sim_input[i].insert(0, param[i])

    def __get_param(self, param):
        return param(*map(lambda x: x.get(), self.txtInputs[param.__name__]))

    def save_entry(self):
        self.entry.simulation_params = self.__get_param(utils.simulation_params)
        self.entry.screen_params = self.__get_param(utils.screen_params)
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
