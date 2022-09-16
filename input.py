import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mb
import utils


## TO CHANGE : inherit class from toplevel/frame

class MainFrame(ttk.Frame):

    def __init__(self, container):
        super(MainFrame, self).__init__(container)
        self.listbox = None
        self.config_file = None
        self.counter = 0
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
        list_items = tk.Variable(value=[])
        self.listbox = tk.Listbox(list_frame, listvariable=list_items, height=10)
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
        self.entry_window('entry')

    def entry_window(self, entry=None):
        entryWin = EntryWindow(self, entry is None, entry)
        entry = entryWin.get_entry()
        if entry is not None:
            self.listbox.insert(tk.END, f'{entry} {self.counter}')
            self.counter += 1

    def remove_entry(self):
        idx = self.listbox.curselection()
        if not idx:
            mb.showerror('Error', 'You must select an entry to delete')
        elif mb.askyesno('Verify', 'Are you sure you want to delete the selected entry?'):
            self.listbox.delete(idx)

    def load_config_file(self):
        self.config_file = utils.select_file(self.config_file)


class EntryWindow(tk.Toplevel):

    def __init__(self, parent, new, entry):
        super(EntryWindow, self).__init__(parent)
        self.new = new
        self.entry = entry

        self.txtInputs = []
        self.saveButton = tk.Button(self, text='SAVE', command=self.save_entry)
        self.window_init()

    def window_init(self):
        title_base = 'Add new' if self.new else 'Edit'
        self.title(f'{title_base} entry')
        self.attributes('-topmost', 1)

        if not self.new:
            self.load_entry()

        f = ttk.Frame(self)
        f['padding'] = (5, 10)
        f.pack(fill=tk.BOTH)
        self.add_text_inputs(f)
        self.saveButton.pack(side=tk.BOTTOM)

    def add_text_inputs(self, mainFrame):
        for field in utils.simulation_params._fields:
            f = tk.Frame(mainFrame)
            lbl = tk.Label(f, text=field)
            txt = ttk.Entry(f)
            self.txtInputs.append(txt)
            lbl.pack(side=tk.LEFT)
            txt.pack(side=tk.LEFT, fill=tk.BOTH)
            f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def load_entry(self):
        return  # TODO: function to load the entry if editing

    def save_entry(self):
        self.entry = ''
        for ipt in self.txtInputs:
            self.entry += ipt.get()
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