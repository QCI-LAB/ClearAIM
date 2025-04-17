'''Script for running the mask detection on a folder of images.'''
import sys
from pathlib import Path
# Add project directory to sys.path for imports to work from anywhere
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
sys.path.insert(0, str(project_root))

from src.mask_detector import MaskDetectorBuilder

def transform_source_path_to_save_path(path_source: str) -> str:
    """
    Transforms the source file path to a save file path
    by replacing the directory name "Materials" with "Results".
    Args:
        path_source (str): The source file path to be transformed.
    Returns:
        str: The transformed file path with "Materials" replaced by "Results".
    """
    return path_source.replace("Materials", "Results")

if __name__ == "__main__":
    builder = MaskDetectorBuilder()
    builder.folderpath_source = r".\Materials\250um brain\skrawek 2"
    builder.folderpath_save = transform_source_path_to_save_path(builder.folderpath_source)
    builder.num_negative_points = 20
    builder.is_display = True
    builder.is_roi = True

    detector = builder.build()
    detector.process_images()