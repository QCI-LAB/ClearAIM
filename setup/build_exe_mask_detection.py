import PyInstaller.__main__
import os

# Determine project root directory (one level up from this script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Paths to your GUI script and model file
script_path = os.path.join(project_root, "gui", "gui_mask_detector.py")
model_path = os.path.join(project_root, "models", "sam_vit_h.pth")

# Verify model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Build the one-file executable, include the model and your project on the import path
PyInstaller.__main__.run([
    script_path,
    "--onefile",
    f"--add-data={model_path};models",
    f"--paths={project_root}",               # so PyInstaller can find your bin/ package
    "--hidden-import=bin.mask_detection",   # force-include any modules under bin/
    "--noconfirm"
])
