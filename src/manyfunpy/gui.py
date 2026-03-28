#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GUI utility functions for SEMSR

This module contains functions for creating GUI dialogs and interfaces.
"""

import tkinter as tk
from tkinter import ttk

def create_selection_dialog(title, items, multiple=True):
    """Create a dialog with listbox for selection
    
    Parameters
    ----------
    title : str
        Title of the dialog window
    items : list
        List of items to display in the listbox
    multiple : bool, optional
        Whether to allow multiple selection, by default True
    
    Returns
    -------
    list
        List of selected items
    """
    root = tk.Tk()
    root.title(title)
    
    # Create and pack a frame
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Create listbox with scrollbar
    scrollbar = ttk.Scrollbar(frame)
    scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, selectmode='multiple' if multiple else 'single')
    listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    scrollbar.config(command=listbox.yview)
    
    # Add items to listbox
    for item in items:
        listbox.insert(tk.END, item)
    
    # Add selection button
    selected_items = []
    def on_select():
        selected_items.extend([items[int(i)] for i in listbox.curselection()])
        root.quit()
        
    ttk.Button(frame, text="Select", command=on_select).grid(row=1, column=0, columnspan=2)
    
    root.mainloop()
    root.destroy()
    return selected_items 