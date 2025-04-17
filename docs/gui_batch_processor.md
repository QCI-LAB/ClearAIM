# Documentation: gui_batch_processor.exe

`gui_batch_processor.exe` is a graphical desktop application for **batch image processing** using the **Segment Anything Model (SAM)**. It enables users to easily select data folders and automatically process biological images without running any terminal commands.

The app is written in Python and compiled into a standalone `.exe` using **PyInstaller**.

---

## Key Features

- Processes images from multiple nested subfolders.
- Automatically detects the **deepest subdirectories** with images.
- Skips empty or user-defined **ignored folders** (e.g., `rejected`, `temp`).
- Interactive **ROI (Region of Interest)** selection for each batch.
- Saves segmentation masks in a **mirrored folder structure**.
- Tracks processing status using a **CSV file**.
- Simple and intuitive **graphical user interface (GUI)**.

---

## Requirements

- **System**: Windows (recommended)
- **Python**: not required when using the `.exe`
- **MATLAB**: not required (only needed for separate analysis via `main_analize.m`)

---

## How to Use the `.exe`

1. Launch `gui_batch_processor.exe`
2. Provide the following inputs:
   - **Source folder** – the main directory with your images (e.g., `Materials/`)
   - **Ignore folders** – comma-separated list of folder names to skip (e.g., `rejected,temp`)
   - **CSV path** – path to the status-tracking file (e.g., `folder_processing_status.csv`)
3. Click **Run**

The GUI will hide, and processing will begin in the background. It will close automatically when finished.

---

## How Automation Works

### Example Folder Structure (Supported)

```
Materials/
├── sample_1/
│   └── series_A/
│       ├── img1.png
│       └── img2.png
├── sample_2/
│   └── series_B/
│       ├── img1.jpg
│       └── ...
├── rejected/              <-- will be skipped (if added to the ignore list)
│   └── test/
└── temp/                  <-- can also be ignored
```

The program will:

1. **Automatically find** the deepest subfolders containing images (`series_A`, `series_B`)
2. **Ignore** specified folders like `rejected` and `temp`
3. **Create output folders** inside `results_mask/`, preserving structure:
   ```
   results_mask/
   ├── sample_1/
   │   └── series_A/
   └── sample_2/
       └── series_B/
   ```
4. **Process all images** in each folder:
   - You select ROI and positive points on the first image
   - Remaining images are processed automatically
   - Output masks are saved like:
     ```
     results_mask/sample_1/series_A/img1_mask.png
     ```

5. **Update the CSV file** to prevent duplicate processing on future runs.

---

## CSV File Format

| Folder Name | Processed | Date                |
|-------------|-----------|---------------------|
| series_A    | Yes       | 2025-04-17 13:45:22 |
| series_B    | No        | 2025-04-17 13:45:22 |

---

## Benefits

- Batch-process multiple image folders in a single run
- No reprocessing – already completed folders are tracked
- GUI replaces manual scripting
- Output is automatically organized and mirrored

