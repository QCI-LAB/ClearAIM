import os

def find_deepest_subfolders(root_folder, ignore_names):
    """
    Przeszukuje drzewo katalogów począwszy od `root_folder`,
    ignorując foldery o nazwach znajdujących się w `ignore_names`,
    i zwraca listę pełnych ścieżek do najgłębszych (bez podfolderów) folderów.
    
    Parametry:
      root_folder: Ścieżka do katalogu głównego.
      ignore_names: Lista nazw folderów, które mają być ignorowane.
    """
    deepest_folders = []
    for current, dirs, _ in os.walk(root_folder, topdown=True):
        # Usuń foldery, które znajdują się na liście ignorowanych
        dirs[:] = [d for d in dirs if d not in ignore_names]
        # Jeśli po przefiltrowaniu nie pozostały podfoldery, katalog jest najgłębszy
        if not dirs:
            deepest_folders.append(current)
    return deepest_folders

def create_results_folders(source_folder, deepest_folders):
    """
    Dla każdego folderu z listy `deepest_folders` utwórz analogiczny folder w 
    katalogu 'results' umieszczonym wewnątrz `source_folder`.
    
    Parametry:
        source_folder: Ścieżka do głównego folderu źródłowego.
        deepest_folders: Lista najgłębszych folderów znalezionych w source_folder.
    """
    results_base = os.path.join(source_folder, 'results')
    for path in deepest_folders:
        new_path = path.replace(source_folder, results_base)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print(f"Utworzono folder: {new_path}")
        else:
            print(f"Folder już istnieje: {new_path}")

# Przykład użycia:
if __name__ == "__main__":
    folder = r'C:\Praca\QCI Lab repozytoria\ClearAIM\Materials'
    ignore = ['odrzucone']
    path_to_csv = r'C:\Praca\QCI Lab repozytoria\ClearAIM\Materials\folder_processing_status.csv'
    wynik = find_deepest_subfolders(folder, ignore)
    print("Najgłębsze foldery (ścieżki względne):")
    for path in wynik:
        print(path)
        
    # Stwórz analogiczne foldery ale w folderze o nazwie results
    create_results_folders(folder, wynik)
    print("Foldery wynikowe utworzone w katalogu 'results'.")
    
    # Make list in CSV that says if folder was processed or not
    # Create a CSV file with the folder names and processed status
    import pandas as pd
    import datetime
    from pathlib import Path
    import os
    
    if not os.path.exists(path_to_csv):
        # Create the directory if it doesn't exist
        Path(path_to_csv).parent.mkdir(parents=True, exist_ok=True)
        # Create a DataFrame with the folder names and processed status
        data = {
            "Folder Name": [os.path.basename(path) for path in wynik],
            "Processed": ["No" for _ in wynik],
            "Date": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") for _ in wynik]
        }
        df = pd.DataFrame(data)
        
        #Save the DataFrame to a CSV file
        df.to_csv(path_to_csv, index=False)
        print(f"Plik CSV z nazwami folderów i statusem przetwarzania został zapisany jako {path_to_csv}.")
    else:
        # Load the existing CSV file
        df = pd.read_csv(path_to_csv)
        
        # Update the DataFrame with the new folder names and processed status
        new_data = {
            "Folder Name": [os.path.basename(path) for path in wynik],
            "Processed": ["No" for _ in wynik],
            "Date": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") for _ in wynik]
        }
        new_df = pd.DataFrame(new_data)
        
        # Append the new data to the existing DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Save the updated DataFrame to the CSV file
        df.to_csv(path_to_csv, index=False)
        print(f"Plik CSV zaktualizowany i zapisany jako {path_to_csv}.")
    
    
    
    
    

