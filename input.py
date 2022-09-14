import tkinter as tk
from tkinter import messagebox as mb

## TO CHANGE : inherit class from toplevel/frame

class MainWindow(tk.Frame):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.listbox = None
        self.counter = 0
        self.main_window()

    def main_window(self):
        self.master.title('WAVES toolbox')
        self.pack(fill=tk.BOTH, expand=True)

        header = tk.Label(text='Add entries to generate simulations for given .fif file')
        header.pack(side=tk.TOP)
        self.create_list_frame()

    def create_list_frame(self):
        # create list for entries
        list_frame = tk.Frame(self)
        list_frame.pack(expand=True)
        list_items = tk.Variable(value=[])
        self.listbox = tk.Listbox(list_frame, listvariable=list_items, height=10)
        self.listbox.pack(expand=True, side=tk.LEFT)

        button_frame = tk.Frame(list_frame)
        button_frame.pack(side=tk.LEFT)

        add_button = tk.Button(button_frame, text='ADD', command=self.entry_window)
        add_button.pack(side=tk.TOP)
        edit_button = tk.Button(button_frame, text='EDIT', command=self.edit_entry)
        edit_button.pack(side=tk.TOP)
        remove_button = tk.Button(button_frame, text='REMOVE', command=self.remove_entry)
        remove_button.pack(side=tk.TOP)

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


class EntryWindow(tk.Toplevel):

    def __init__(self, parent, new, entry):
        super(EntryWindow, self).__init__(parent)
        self.new = new
        self.entry = entry

        self.txtInput = tk.Text(self)
        self.saveButton = tk.Button(self, text='SAVE', command=self.save_entry)
        self.window_init()

    def window_init(self):
        title_base = 'Add new' if self.new else 'Edit'
        self.title(f'{title_base} entry')
        self.attributes('-topmost', 1)

        if not self.new:
            self.load_entry()

        self.txtInput.pack()
        self.saveButton.pack()

    def load_entry(self):
        return  # TODO: function to load the entry if editing

    def save_entry(self):
        self.entry = self.txtInput.get(1.0, "end-1c")
        self.destroy()

    def get_entry(self):
        self.close_window()
        return self.entry

    def close_window(self):
        self.deiconify()
        self.wm_protocol("WN_DELETE_WINDOW", self.destroy)
        self.wait_window(self)


r = tk.Tk()
r.geometry("700x350")
gui = MainWindow()
r.mainloop()
