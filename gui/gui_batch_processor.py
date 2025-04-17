import ctypes
import tkinter as tk
from tkinter import ttk, filedialog
import sys
from pathlib import Path

# Ensure batch_processor module is importable (go up one directory)
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
sys.path.insert(0, str(project_root))

from bin.batch_processor import BatchProcessor

# Enable high DPI awareness on Windows
if sys.platform.startswith('win'):
    try:
        # 2 for PER_MONITOR_DPI_AWARE
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

def browse_folder(entry):
    folder_selected = filedialog.askdirectory(initialdir=entry.get() or ".")
    if folder_selected:
        entry.delete(0, tk.END)
        entry.insert(0, folder_selected)


def browse_csv(entry):
    file_selected = filedialog.asksaveasfilename(
        initialfile="folder_processing_status.csv",
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*")],
        initialdir=entry.get() or "."
    )
    if file_selected:
        entry.delete(0, tk.END)
        entry.insert(0, file_selected)


def run_gui():
    def get_params_from_gui():
        # Split ignore list by commas and strip whitespace
        ignore_text = entry_ignore.get()
        ignore_list = [item.strip() for item in ignore_text.split(",") if item.strip()]
        return {
            "source_folder": entry_source.get(),
            "ignore_list": ignore_list,
            "csv_path": entry_csv.get()
        }

    def start_main_script():
        root.withdraw()  # Hide the GUI
        params = get_params_from_gui()
        print("Starting BatchProcessor with settings:")
        print(f"  Source folder: {params['source_folder']}")
        print(f"  Ignore list: {params['ignore_list']}")
        print(f"  CSV path: {params['csv_path']}")

        # Run the batch processor
        processor = BatchProcessor(
            source_folder=params['source_folder'],
            ignore=params['ignore_list'],
            csv_path=params['csv_path']
        )
        processor.run()

        print("Batch processing finished.")
        root.quit()

    root = tk.Tk()
    root.title("Batch Processor GUI")
    root.resizable(False, False)
    root.configure(padx=10, pady=10)

    # Adjust Tk scaling based on system DPI
    try:
        dpi = root.winfo_fpixels('1i')
        root.tk.call('tk', 'scaling', dpi/72)
    except Exception:
        pass

    style = ttk.Style()
    style.configure("TButton", padding=5)
    style.configure("TLabel", padding=5)

    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0)

    # Source folder
    ttk.Label(frame, text="Source folder:").grid(row=0, column=0, sticky="w")
    entry_source = ttk.Entry(frame, width=40)
    entry_source.insert(0, str(project_root / "Materials"))
    entry_source.grid(row=0, column=1)
    ttk.Button(frame, text="Browse", command=lambda: browse_folder(entry_source)).grid(row=0, column=2)

    # Ignore folders
    ttk.Label(frame, text="Ignore folders (comma-separated):").grid(row=1, column=0, sticky="w")
    entry_ignore = ttk.Entry(frame, width=40)
    entry_ignore.insert(0, "odrzucone,temp")
    entry_ignore.grid(row=1, column=1, columnspan=2, sticky="w")

    # CSV path
    ttk.Label(frame, text="CSV path:").grid(row=2, column=0, sticky="w")
    entry_csv = ttk.Entry(frame, width=40)
    entry_csv.insert(0, str(project_root / "folder_processing_status.csv"))
    entry_csv.grid(row=2, column=1)
    ttk.Button(frame, text="Browse", command=lambda: browse_csv(entry_csv)).grid(row=2, column=2)

    # Run button
    ttk.Button(frame, text="Run", command=start_main_script).grid(row=3, column=0, columnspan=3, pady=10)

    root.mainloop()


if __name__ == "__main__":
    run_gui()
