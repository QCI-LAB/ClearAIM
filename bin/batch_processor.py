import os
from pathlib import Path
import datetime
import pandas as pd
import cv2

# Ustalenie katalogu głównego projektu i dodanie go do sys.path
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
        Przeszukuje drzewo katalogów począwszy od source_folder,
        ignorując foldery o nazwach znajdujących się na liście ignore,
        i zwraca listę pełnych ścieżek do najgłębszych folderów.
        """
        deepest_folders = []
        for current, dirs, _ in os.walk(self.source_folder):
            dirs[:] = [d for d in dirs if d not in self.ignore]
            if not dirs:
                deepest_folders.append(current)
        return deepest_folders

    def create_results_folders(self, deepest_folders):
        """
        Dla każdego folderu z listy deepest_folders utwórz analogiczny
        folder w katalogu 'results_mask' wewnątrz katalogu nadrzędnego source_folder.
        """
        for folder in deepest_folders:
            folder = Path(folder)
            relative_folder = folder.relative_to(self.source_folder)
            new_path = self.results_base / relative_folder
            new_path.mkdir(parents=True, exist_ok=True)
            print(f"Utworzono folder: {new_path}")

    def update_processing_csv(self, deepest_folders):
        """
        Tworzy lub aktualizuje plik CSV zawierający status przetwarzania folderów.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "Folder Name": [Path(path).name for path in deepest_folders],
            "Processed": ["No"] * len(deepest_folders),
            "Date": [current_time] * len(deepest_folders)
        }
        new_df = pd.DataFrame(data)
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            df = pd.concat([df, new_df], ignore_index=True)
            action = "zaktualizowany"
        else:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            df = new_df
            action = "zapisany"

        df.to_csv(self.csv_path, index=False)
        print(f"Plik CSV został {action} jako {self.csv_path}")

    def process_images(self, deepest_folders):
        """
        Przetwarza obrazy dla każdego niepustego folderu znajdującego się na liście deepest_folders.
        Wykorzystuje konfigurację i funkcje importowane z modułu 'src.mask_detector'
        """
        configs = []
        for folder in deepest_folders:
            folder = Path(folder)
            if not any(folder.iterdir()):
                print(f"Folder {folder} jest pusty. Pomijam.")
                continue

            paths_image = list(folder.rglob('*.jpg')) + list(folder.rglob('*.png'))
            paths_image = [str(path) for path in paths_image]

            config = MaskDetectorConfig()
            config.folderpath_source = str(folder)
            # Tworzymy folder wynikowy analogicznie do struktury z source_folder
            config.folderpath_save = str(Path(str(folder)).as_posix().replace(str(self.source_folder), str(self.results_base)))
            config.num_negative_points = 20
            config.is_display = False
            config.is_roi = False
            config.downscale_factor = 1.0

            # Inicjalizacja od pierwszego obrazu
            first_image = ImageProcessor.load_image(paths_image[0])
            first_image = ImageProcessor.rescale(first_image, 1/config.downscale_factor)

            # Wybór ROI
            cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Select ROI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Select ROI", first_image)
            cv2.waitKey(1)
            roi = cv2.selectROI("Select ROI", first_image, False, False)
            cv2.destroyWindow("Select ROI")
            config.box_roi = roi

            first_image = ImageProcessor.crop_image(first_image, config.box_roi)

            print(f"Wybierz {config.num_positive_points} pozytywne punkty na obrazie")
            init_points_positive = get_click_coordinates(
                cv2.cvtColor(first_image, cv2.COLOR_RGB2BGR),
                config.num_positive_points
            )
            config.init_points_positive = init_points_positive
            configs.append(config)

        # Wczytanie danych z CSV
        df = pd.read_csv(self.csv_path)
        for config in configs:
            folder_name = Path(config.folderpath_source).name
            if df[df['Folder Name'] == folder_name]['Processed'].values[0] == "Yes":
                print(f"Folder {config.folderpath_source} już przetworzony.")
                continue

            detector = MaskDetector(config=config)
            detector.process_images()
            print(f"Folder {config.folderpath_source} przetworzony.")

            # Aktualizacja statusu w pliku CSV
            df.loc[df['Folder Name'] == folder_name, 'Processed'] = "Yes"
            df.to_csv(self.csv_path, index=False)
            print(f"Status przetwarzania folderu {config.folderpath_source} zaktualizowany w pliku CSV.")

    def run(self):
        deepest = self.find_deepest_subfolders()
        print("Najgłębsze foldery (ścieżki pełne):")
        for path in deepest:
            print(path)
        self.create_results_folders(deepest)
        print("Foldery wynikowe utworzone w katalogu 'results_mask'.")
        self.update_processing_csv(deepest)
        self.process_images(deepest)

if __name__ == "__main__":
    source_folder = r'C:\Praca\QCI Lab repozytoria\ClearAIM\Materials'
    ignore = ['odrzucone']
    csv_path = r'C:\Praca\QCI Lab repozytoria\ClearAIM\Materials\folder_processing_status.csv'
    
    processor = BatchProcessor(source_folder, ignore, csv_path)
    processor.run()
