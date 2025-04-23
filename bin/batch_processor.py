import os
from pathlib import Path
import datetime
import pandas as pd
import cv2

# Determine the project root directory and add it to sys.path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.mask_detector import MaskDetectorConfig, ImageProcessor, get_click_coordinates
from src.mask_detector import MaskDetector

class BatchProcessor:
    def __init__(self, source_folder, ignore, csv_path):
        self.source_folder = Path(source_folder)
        self.ignore = ignore
        self.csv_path = Path(csv_path)
        self.results_base = self.source_folder.parent / 'results_mask'

    def find_deepest_subfolders(self):
        """
        Searches the directory tree starting from source_folder,
        ignoring folders with names on the ignore list,
        and returns a list of full paths to the deepest folders.
        """
        deepest_folders = []
        for current, dirs, _ in os.walk(self.source_folder):
            dirs[:] = [d for d in dirs if d not in self.ignore]
            if not dirs:
                deepest_folders.append(current)
        return deepest_folders

    def create_results_folders(self, deepest_folders):
        """ 
        For each folder in the deepest_folders list, create an analogous
        folder in the 'results_mask' directory inside the parent directory of source_folder.
        """
        for folder in deepest_folders:
            folder = Path(folder)
            relative_folder = folder.relative_to(self.source_folder)
            new_path = self.results_base / relative_folder
            new_path.mkdir(parents=True, exist_ok=True)
            print(f"Created folder: {new_path}")

    def update_processing_csv(self, deepest_folders):
        """
        Creates or updates a CSV file containing the processing status of folders,
        avoiding duplicate records.
        """
        # If it does not exist or the CSV file is empty, create a CSV file
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(columns=["Folder Name", "Processed", "Date"])
            df.to_csv(self.csv_path, index=False)
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "Folder Name": [Path(path).name for path in deepest_folders],
            "Processed": ["No"] * len(deepest_folders),
            "Date": [current_time] * len(deepest_folders)
        }
        new_df = pd.DataFrame(data)

        # Try to load CSV, if the file is empty, create a DataFrame with constant columns
        try:
            df = pd.read_csv(self.csv_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["Folder Name", "Processed", "Date"])

        # Remove records that already exist in CSV
        new_df = new_df[~new_df['Folder Name'].isin(df['Folder Name'])]

        if not new_df.empty:
            df = pd.concat([df, new_df], ignore_index=True)
            action = "updated"
        else:
            action = "saved"

        df.to_csv(self.csv_path, index=False)
        print(f"CSV file {action} as {self.csv_path}")

    def process_images(self, deepest_folders):
        """
        Processes images for each non-empty folder in the deepest_folders list.
        Uses configuration and functions imported from the 'src.mask_detector' module
        """
        df = pd.read_csv(self.csv_path)
        configs = []
        for folder in deepest_folders:
            
            if df[df['Folder Name'] == Path(folder).name]['Processed'].values[0] == "Yes":
                print(f"Folder {folder} already processed. Skipping.")
                continue
    
            folder = Path(folder)
            if not any(folder.iterdir()):
                print(f"Folder {folder} is empty. Skipping.")
                continue

            paths_image = list(folder.rglob('*.jpg')) + list(folder.rglob('*.png'))
            paths_image = [str(path) for path in paths_image]

            config = MaskDetectorConfig()
            config.folderpath_source = str(folder)
            # Create a result folder analogous to the structure from source_folder
            relative_path = Path(folder).relative_to(self.source_folder)
            config.folderpath_save = str(self.results_base / relative_path)
            config.num_negative_points = 20
            config.is_display = False
            config.is_roi = False
            config.downscale_factor = 3.0

            # Initialization from the first image
            first_image = ImageProcessor.load_image(paths_image[0])
            first_image = ImageProcessor.rescale(first_image, 1/config.downscale_factor)

            # ROI selection
            cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Select ROI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Select ROI", first_image)
            cv2.waitKey(1)
            box_roi = cv2.selectROI("Select ROI", first_image, False, False)
            cv2.destroyWindow("Select ROI")
            config.box_roi = box_roi

            first_image = ImageProcessor.crop_image(first_image, config.box_roi)

            print(f"Select {config.num_positive_points} positive points on the image")
            init_points_positive = get_click_coordinates(
                cv2.cvtColor(first_image, cv2.COLOR_RGB2BGR),
                config.num_positive_points
            )
            config.init_points_positive = init_points_positive
            configs.append(config)

        # Load data from CSV
        df = pd.read_csv(self.csv_path)
        for config in configs:
            folder_name = Path(config.folderpath_source).name
            if df[df['Folder Name'] == folder_name]['Processed'].values[0] == "Yes":
                print(f"Folder {config.folderpath_source} already processed.")
                continue

            detector = MaskDetector(cfg=config)
            detector.process_images()
            print(f"Folder {config.folderpath_source} processed.")

            # Update status in CSV file
            df.loc[df['Folder Name'] == folder_name, 'Processed'] = "Yes"
            df.to_csv(self.csv_path, index=False)
            print(f"Processing status of folder {config.folderpath_source} updated in CSV file.")

    def run(self):
        deepest = self.find_deepest_subfolders()
        print("Deepest folders (full paths):")
        for path in deepest:
            print("    " + path)
        self.create_results_folders(deepest)
        print("Result folders created in the 'results_mask' directory.")
        self.update_processing_csv(deepest)
        self.process_images(deepest)

if __name__ == "__main__":
    source_folder = r'C:\Praca\QCI Lab repozytoria\ClearAIM\Materials'
    ignore = ['odrzucone', 'temp']
    csv_path = r'C:\Praca\QCI Lab repozytoria\ClearAIM\Materials\folder_processing_status.csv'
    
    processor = BatchProcessor(source_folder, ignore, csv_path)
    processor.run()
