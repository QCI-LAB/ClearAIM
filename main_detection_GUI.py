from gui.gui import run_gui
from src.mask_detector import MaskDetectorBuilder

def main_mask_detection(params: dict):
    builder = MaskDetectorBuilder()
    builder.folderpath_source = params["folderpath_source"]
    builder.folderpath_save = params["folderpath_save"]
    builder.num_negative_points = params["num_negative_points"]
    builder.is_display = params["is_display"]
    builder.is_roi = params["is_roi"]

    detector = builder.build()
    detector.process_images()

run_gui(main_mask_detection)