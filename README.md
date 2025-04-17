# ClearAIM

**Clear Artificial Intelligence Monitoring** – a tool for biological image analysis using the Segment Anything Model (SAM) in Python, with optional time-series analysis in MATLAB.

---

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download the SAM model**

   ```bash
   python setup/download_sam.py
   ```

3. **Build standalone executables (optional)**

   Run the dedicated scripts in `setup/`:

   ```bash
   python setup/build_exe_batch_processor.py
   python setup/build_exe_mask_detection.py
   ```

   This uses PyInstaller to create single-file executables for the GUIs. The output appears in `dist/`.

   **Alternatively, download ready-to-use `.exe` files here:**
   - [gui_batch_processing.exe](http://gofile.me/67C1K/l6sFj2rqx)
      - 👉 [Documentation for `gui_batch_processor.exe`](docs/gui_batch_processor.md)

   - [gui_gui_mask_detector.exe](http://gofile.me/67C1K/yN2szAxD2)
      - 👉 [Documentation for `gui_mask_detector.exe`](docs/gui_mask_detector.md)


--- 

## Requirements

- Python 3.8+
- Windows (recommended for full GUI support)
- **MATLAB with Computer Vision Toolbox** – *only required for result visualization and analysis in `main_analize.m`*

--- 

## Project structure

```
models/      # SAM model weight files (.pth)
bin/         # Main scripts (mask detection and batch processing GUIs)
gui/         # GUI modules (interface definitions)
setup/       # setup scripts: download_sam.py and build_exe_file.py
src/         # Core mask detection logic (MaskDetectorBuilder, MaskDetector)
```

---

## Running the GUIs

### Mask Detection GUI

```bash
python bin/gui_gui_mask_detector.py
```

Configure parameters:
- **Display results**: show live output.
- **Downscale Factor**: scaling factor before processing.
- **Source folder**: directory with input images.
- **Save folder**: directory for output masks.
- **Number of positive points**: number of positive initialization points.
- **Number of negative points**: number of negative points.
- **Use ROI**: crop to a region of interest before selecting points.

Click **Run**. The GUI will hide and begin processing images.


### Batch Processing GUI

```bash
python bin/gui_batch_processing.py
```

---

## Analysis and visualization (MATLAB)

The `main_analize.m` script:

1. Loads image–mask pairs  
2. Computes Weber contrast, object area, and transmittance  
3. Reads timestamps from EXIF metadata (if available)  
4. Plots metrics over time  
5. Generates a video animation

### How to run

1. Open `main_analize.m` in MATLAB  
2. Set `path_images` and `path_masks`  
3. Run the script

