import ctypes  # for High DPI support on Windows
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog

# Add project directory to sys.path for imports to work from anywhere
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
sys.path.insert(0, str(project_root))

from src.mask_detector import MaskDetectorBuilder  # main builder for detection

# Set High DPI awareness on Windows (2 = PER_MONITOR_DPI_AWARE)
if sys.platform.startswith('win'):
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        # If DPI setting fails, just continue
        pass

# Dynamic interface scaling based on system DPI
try:
    _tmp = tk.Tk()  # temporary window to get DPI
    dpi = _tmp.winfo_fpixels('1i')  # number of pixels per inch
    _tmp.tk.call('tk', 'scaling', dpi/72)  # scale relative to 72 DPI
    _tmp.destroy()
except Exception:
    pass


def browse_folder(entry):
    """
    Opens a folder selection dialog and inserts the selected path into the given Entry field.
    """
    folder_selected = filedialog.askdirectory(initialdir=entry.get() or ".")
    if folder_selected:
        entry.delete(0, tk.END)
        entry.insert(0, folder_selected)


def run_gui():
    """
    Creates the main GUI window, collects parameters, and starts mask detection.
    """
    # Callback to collect data from GUI
    def get_params_from_gui() -> dict:
        return {
            "is_display": bool(var_is_display.get()),
            "downscale_factor": float(entry_downscale_factor.get()),
            "folderpath_source": entry_folderpath_source.get(),
            "folderpath_save": entry_folderpath_save.get(),
            "num_positive_points": int(entry_num_positive_points.get()),
            "num_negative_points": int(entry_num_negative_points.get()),
            "is_roi": bool(var_is_roi.get())
        }
    
    # Callback to start processing
    def start_processing():
        root.withdraw()  # hide GUI during processing
        params = get_params_from_gui()
        print("Starting mask detection with parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")

        # Configure builder based on parameters
        builder = (
            MaskDetectorBuilder()
            .set_source(params["folderpath_source"])
            .set_save(params["folderpath_save"])
            .set_positive_points(params["num_positive_points"])
            .set_negative_points(params["num_negative_points"])
            .set_display(params["is_display"])
            .set_roi(params["is_roi"])
            .set_downscale(params["downscale_factor"])
            )
            
        # Create detector and process images
        detector = builder.build()
        detector.process_images()

        print("Mask detection complete.")
        root.quit()  # close GUI application

    # Build main window
    root = tk.Tk()
    root.title("Mask Detector Settings")
    root.resizable(False, False)
    root.configure(padx=10, pady=10)

    style = ttk.Style()
    style.configure("TButton", padding=5)
    style.configure("TCheckbutton", padding=5)

    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0)

    # Checkbox to display results on screen
    var_is_display = tk.BooleanVar()
    ttk.Label(frame, text="Display results:").grid(row=0, column=0, sticky="w")
    ttk.Checkbutton(frame, variable=var_is_display).grid(row=0, column=1, sticky="w")

    # Image downscale factor
    entry_downscale_factor = ttk.Entry(frame)
    entry_downscale_factor.insert(0, "1.0")  # default value
    ttk.Label(frame, text="Downscale Factor:").grid(row=1, column=0, sticky="w")
    entry_downscale_factor.grid(row=1, column=1)

    # Source directory of images
    entry_folderpath_source = ttk.Entry(frame, width=40)
    entry_folderpath_source.insert(0, str(project_root / "Materials"))
    ttk.Label(frame, text="Source folder:").grid(row=2, column=0, sticky="w")
    entry_folderpath_source.grid(row=2, column=1)
    ttk.Button(frame, text="Browse", command=lambda: browse_folder(entry_folderpath_source)).grid(row=2, column=2)

    # Destination directory for results
    entry_folderpath_save = ttk.Entry(frame, width=40)
    entry_folderpath_save.insert(0, str(project_root / "Results"))
    ttk.Label(frame, text="Save folder:").grid(row=3, column=0, sticky="w")
    entry_folderpath_save.grid(row=3, column=1)
    ttk.Button(frame, text="Browse", command=lambda: browse_folder(entry_folderpath_save)).grid(row=3, column=2)

    # Number of positive points for initialization
    entry_num_positive_points = ttk.Entry(frame)
    entry_num_positive_points.insert(0, "2")
    ttk.Label(frame, text="Number of positive points:").grid(row=4, column=0, sticky="w")
    entry_num_positive_points.grid(row=4, column=1)

    # Number of negative points
    entry_num_negative_points = ttk.Entry(frame)
    entry_num_negative_points.insert(0, "12")
    ttk.Label(frame, text="Number of negative points:").grid(row=5, column=0, sticky="w")
    entry_num_negative_points.grid(row=5, column=1)

    # Whether to use ROI
    var_is_roi = tk.BooleanVar()
    ttk.Label(frame, text="Use ROI:").grid(row=6, column=0, sticky="w")
    ttk.Checkbutton(frame, variable=var_is_roi).grid(row=6, column=1, sticky="w")

    # Button to start processing
    ttk.Button(frame, text="Run", command=start_processing).grid(row=7, column=0, columnspan=3, pady=10)

    root.mainloop()


if __name__ == "__main__":
    run_gui()