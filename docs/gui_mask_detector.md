# Documentation: gui_mask_detector.exe

`gui_mask_detector.exe` is a graphical desktop application for **single-folder image segmentation** using the **Segment Anything Model (SAM)**. It allows users to quickly test mask generation on a selected image folder with customizable parameters through an interactive GUI.

This application is written in Python and compiled into a standalone `.exe` using **PyInstaller**.

---

## Key Features

- Segments images from a single selected folder
- Interactive **ROI (Region of Interest)** selection
- Manual selection of **positive and negative points**
- Customizable downscale factor for faster processing
- Optional live display of mask overlay
- Easy folder selection for both input and output paths
- Lightweight and intuitive GUI for quick testing

---

## Requirements

- **System**: Windows (recommended)
- **Python**: not required when using the `.exe`
- **SAM model checkpoint**: must be available in `models/sam_vit_h.pth` if using from source

---

## How to Use the `.exe`

1. Launch `gui_mask_detector.exe`
2. Configure the following parameters:
   - **Display results** – whether to show live visualization (checkbox)
   - **Downscale factor** – e.g. `1.0` for full res, `2.0` for half size
   - **Source folder** – directory containing input images
   - **Save folder** – where to store the resulting masks
   - **Number of positive points** – number of clicks to initialize segmentation
   - **Number of negative points** – number of background reference points
   - **Use ROI** – whether to crop to a region before selecting points
3. Click **Run**

The GUI will hide, and processing will start. After completion, the GUI will close automatically.

---

## How It Works

- The GUI collects all parameters and uses them to configure a `MaskDetectorBuilder`
- The first image is shown for ROI and point selection
- All images in the source folder are processed in order using the SAM model
- Masks are saved next to the original filenames with `_mask` appended, e.g.:

```
Results/
├── img1_mask.png
├── img2_mask.png
```

---

## Example Folder Setup

```
Materials/
├── test_dataset/
│   ├── img1.png
│   ├── img2.jpg
│   └── img3.png
```

You can select `Materials/test_dataset/` as the **source folder** and any output folder (e.g., `Results/`) as the **save folder**.

---

## Benefits

- Great for testing segmentation before running full batch jobs
- Fine-tuned control over parameters per run
- Standalone executable – no Python setup needed
- User-friendly GUI interface with minimal configuration required

