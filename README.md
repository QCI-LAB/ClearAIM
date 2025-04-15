# ClearAIM

**Clear Artificial Intelligence Monitoring** – a tool for biological image analysis using the Segment Anything Model (SAM) and MATLAB. It enables object mask detection and time-based analysis of image properties.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the SAM model

```bash
python download_sam.py
```

### 3. (Optional) Build a standalone `.exe` file

```bash
python build_exe_file.py
```

> You can download a sample executable for Windows [here](https://drive.google.com/file/d/1DxwpeNgjtG6KxP-4_RjzXf9UCjP-uhOU/view?usp=sharing).  
> It demonstrates how the program works without requiring any setup.

---

## Mask detection (Python)

The file `main_mask_detection.py` is an example usage of the `MaskDetector` class.  
It is **not** intended to be run directly from the terminal – you should modify the parameters in code.

```python
from src.mask_detector import MaskDetectorBuilder

def transform_source_path_to_save_path(path_source: str) -> str:
    return path_source.replace("Materials", "Result")

if __name__ == "__main__":
    builder = MaskDetectorBuilder()
    builder.folderpath_source = r".\Materials\500um brain\skrawek 3"
    builder.folderpath_save = transform_source_path_to_save_path(builder.folderpath_source)
    builder.num_negative_points = 20
    builder.is_display = True
    builder.is_roi = True

    detector = builder.build()
    detector.process_images()
```

---

## Analysis and visualization (MATLAB)

The `main_analize.m` script:

- loads image–mask pairs,
- calculates Weber contrast, object area, and transmittance,
- reads timestamps from EXIF metadata (if available),
- plots metrics over time,
- generates a video animation of the results.

### How to run:

1. Open the script in MATLAB.
2. Set `path_images` and `path_masks` in the code.
3. Run `main_analize`.

---

## Folder structure

```
Materials/          # Input images
Results/            # Output masks
models/             # SAM model (.pth file)
src/                # Mask detection logic
```

---

## Main files

| File                  | Description                                             |
|-----------------------|---------------------------------------------------------|
| `main_mask_detection.py` | Example usage of the mask detector                     |
| `main_analize.m`          | Analysis of results and animation generation          |
| `download_sam.py`         | Downloads the required SAM model                      |
| `build_exe_file.py`       | Creates a standalone `.exe` file (Windows only)       |
| `requirements.txt`        | Python dependencies list                              |

---
