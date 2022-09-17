import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mb
import utils


## TO CHANGE : inherit class from toplevel/frame

class MainFrame(ttk.Frame):

    def __init__(self, container):
        super(MainFrame, self).__init__(container)
        self.listbox = None
        self.list_items = tk.Variable(value=[])
        self.config_file = None
        self.main_window()

    def main_window(self):
        header = tk.Label(self, text='Add entries to generate simulations for given .fif file')
        header.pack(fill=tk.BOTH, expand=True)
        self.create_list_frame()

    def create_list_frame(self):
        # create list for entries
        list_frame = ttk.Frame(self)
        list_frame['padding'] = (5, 10)
        list_frame.pack(expand=True, fill=tk.BOTH)
        self.listbox = utils.Listbox(list_frame, listvariable=self.list_items, height=10)
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
        print(idx)
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
        print(idx)
        if not idx:
            mb.showerror('Error', 'You must select an entry to delete')
        elif mb.askyesno('Verify', 'Are you sure you want to delete the selected entry?'):
            idx = idx[0]
            self.listbox.delete(idx)

    def load_config_file(self):
        self.config_file = utils.select_file(self.config_file)


class EntryWindow(tk.Toplevel):

    def __init__(self, parent, entry):
        super(EntryWindow, self).__init__(parent)
        self.new = entry is None
        self.entry = utils.Entry() if self.new else entry

        self.txtInputs = []
        self.saveButton = tk.Button(self, text='SAVE', command=self.save_entry)
        self.window_init()

    def window_init(self):
        title_base = 'Add new' if self.new else 'Edit'
        self.title(f'{title_base} entry')
        self.attributes('-topmost', 1)

        f = ttk.Frame(self)
        f['padding'] = (5, 10)
        f.pack(fill=tk.BOTH)
        self.add_file_input(f)
        self.add_text_inputs(f)
        self.saveButton.pack(side=tk.BOTTOM)

        if not self.new:
            self.load_entry(self.entry)

    def add_file_input(self, mainFrame):
        measuredFrame = tk.Frame(mainFrame)
        lbl = tk.Label(measuredFrame, text='Measured data')
        lbl.pack(side=tk.LEFT)
        lbl = tk.Label(measuredFrame, text=self.entry.measured, bg='white')
        lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        load_button = tk.Button(measuredFrame, text='Browse', command=self.select_measured_file)
        load_button.pack(side=tk.LEFT)
        measuredFrame.pack(side=tk.TOP, fill=tk.Y, expand=True)
        retinoFrame = tk.Frame(mainFrame)
        lbl = tk.Label(retinoFrame, text='Retinotopic map MRI')
        lbl.pack(side=tk.LEFT)
        lbl = tk.Label(retinoFrame, text=self.entry.retino_map, bg='white')
        lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        load_button = tk.Button(retinoFrame, text='Browse', command=self.select_retino_file)
        load_button.pack(side=tk.LEFT)
        retinoFrame.pack(side=tk.TOP, fill=tk.Y, expand=True)

    def select_measured_file(self):
        f = utils.select_file(self.entry.measured)
        self.entry.measured = utils.Path(f)

    def select_retino_file(self):
        f = utils.Path(utils.select_file(self.entry.retino_map))
        self.entry.retino_map = f

    def add_text_inputs(self, mainFrame):
        for field in utils.simulation_params._fields:
            f = tk.Frame(mainFrame)
            lbl = tk.Label(f, text=field)
            txt = ttk.Entry(f)
            self.txtInputs.append(txt)
            lbl.pack(side=tk.LEFT)
            txt.pack(side=tk.LEFT, fill=tk.BOTH)
            f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def load_entry(self, entry):
        params = entry.params
        for i in range(len(params)):
            self.txtInputs[i].insert(0, params[i])

    def save_entry(self):
        self.entry.params = utils.simulation_params._make(list(map(lambda x: x.get(), self.txtInputs)))
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


if __name__ == "__main__":
    app = App()
    app.mainloop()