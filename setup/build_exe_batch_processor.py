import PyInstaller.__main__
import os

# project directory (not setup)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
script_name = os.path.join(project_root, "gui", "gui_batch_processor.py")
model_path  = os.path.join(project_root, "models", "sam_vit_h.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

PyInstaller.__main__.run([
    script_name,
    "--onefile",
    f"--add-data={model_path};models",
    f"--paths={project_root}",                # so that PyInstaller can see bin/
    "--hidden-import=bin.batch_processor",    # include bin module
    "--noconfirm"
])
