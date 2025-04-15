import PyInstaller.__main__
import os

script_name = "main_detection_GUI.py"
model_relative_path = os.path.join("models", "sam_vit_h.pth")

if not os.path.exists(model_relative_path):
    raise FileNotFoundError(f"Model file not found: {model_relative_path}")

PyInstaller.__main__.run([
    script_name,
    "--onefile",
    f"--add-data={model_relative_path};models",
    "--noconfirm"
])
