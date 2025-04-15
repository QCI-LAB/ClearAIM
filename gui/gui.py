import tkinter as tk
from tkinter import ttk, filedialog

def browse_folder(entry):
    folder_selected = filedialog.askdirectory(initialdir=entry.get())
    if folder_selected:
        entry.delete(0, tk.END)
        entry.insert(0, folder_selected)


def run_gui(main_script: callable):
    
    def get_params_from_gui() -> dict:
        params = {
            "is_display": bool(var_is_display.get()),
            "downscale_factor": float(entry_downscale_factor.get()),
            "folderpath_source": entry_folderpath_source.get(),
            "folderpath_save": entry_folderpath_save.get(),
            "num_positive_points": int(entry_num_positive_points.get()),
            "num_negative_points": int(entry_num_negative_points.get()),
            "is_roi": bool(var_is_roi.get())
        }
        return params
    
    def start_main_script():
        root.withdraw()  # Hide the GUI
        print("Starting main script...")
        main_script(get_params_from_gui())
        print("Main script finished...")
        root.quit()  # Stop the mainloop
    
    root = tk.Tk()
    root.title("Program Settings")

    root.resizable(False, False)
    root.configure(padx=10, pady=10)

    style = ttk.Style()
    style.configure("TButton", padding=5)
    style.configure("TCheckbutton", padding=5)

    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0)

    # Display results
    var_is_display = tk.BooleanVar()
    ttk.Label(frame, text="Display results:").grid(row=0, column=0, sticky="w")
    ttk.Checkbutton(frame, variable=var_is_display).grid(row=0, column=1, sticky="w")

    # Downscale Factor
    entry_downscale_factor = ttk.Entry(frame)
    entry_downscale_factor.insert(0, "1.0")
    ttk.Label(frame, text="Downscale Factor:").grid(row=1, column=0, sticky="w")
    entry_downscale_factor.grid(row=1, column=1)

    # Source folder
    entry_folderpath_source = ttk.Entry(frame, width=40)
    entry_folderpath_source.insert(0, r".\Materials")
    ttk.Label(frame, text="Source folder:").grid(row=2, column=0, sticky="w")
    entry_folderpath_source.grid(row=2, column=1)
    ttk.Button(frame, text="Browse", command=lambda: browse_folder(entry_folderpath_source)).grid(row=2, column=2)

    # Save folder
    entry_folderpath_save = ttk.Entry(frame, width=40)
    entry_folderpath_save.insert(0, r".\Results")
    ttk.Label(frame, text="Save folder:").grid(row=3, column=0, sticky="w")
    entry_folderpath_save.grid(row=3, column=1)
    ttk.Button(frame, text="Browse", command=lambda: browse_folder(entry_folderpath_save)).grid(row=3, column=2)

    # Number of positive points
    entry_num_positive_points = ttk.Entry(frame)
    entry_num_positive_points.insert(0, "2")
    ttk.Label(frame, text="Number of positive points:").grid(row=4, column=0, sticky="w")
    entry_num_positive_points.grid(row=4, column=1)

    # Number of negative points
    entry_num_negative_points = ttk.Entry(frame)
    entry_num_negative_points.insert(0, "12")
    ttk.Label(frame, text="Number of negative points:").grid(row=5, column=0, sticky="w")
    entry_num_negative_points.grid(row=5, column=1)

    # Use ROI
    var_is_roi = tk.BooleanVar()
    ttk.Label(frame, text="Use ROI:").grid(row=6, column=0, sticky="w")
    ttk.Checkbutton(frame, variable=var_is_roi).grid(row=6, column=1, sticky="w")

    # Next button
    ttk.Button(frame, text="Next", command=start_main_script).grid(row=7, column=0, columnspan=3, pady=10)

    root.mainloop()

