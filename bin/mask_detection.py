'''Script for running the mask detection on a folder of images.'''
import sys
from pathlib import Path
# Add project directory to sys.path for imports to work from anywhere
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
sys.path.insert(0, str(project_root))

from src.mask_detector import MaskDetectorConfig, MaskDetector
from src.utility import ImagePathUtility, get_roi_box

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
    source_path = r".\Materials\250um brain\skrawek 1"
    save_path = transform_source_path_to_save_path(source_path)

    input_paths = ImagePathUtility.get_image_paths(source_path)
    save_paths = [Path(save_path) / (Path(path).stem + "_mask.png") for path in input_paths]

    config = MaskDetectorConfig()
    config.input_paths = input_paths
    config.output_paths = save_paths
    config.num_negative_points = 20
    config.num_positive_points = 2
    config.is_display = False
    config.downscale_factor = 2.0
    
    config.box_roi = get_roi_box(config.input_paths[0], config.downscale_factor)

    detector = MaskDetector(config)
    detector.process_images()