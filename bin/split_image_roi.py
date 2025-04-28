import os
import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog

def resize_to_screen(img, max_fraction=0.9):
    # automatyczne wykrywanie rozdzielczości
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    h, w = img.shape[:2]
    scale = min(screen_width * max_fraction / w, screen_height * max_fraction / h)
    if scale >= 1.0:
        return img.copy(), 1.0  # nie trzeba skalować

    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale

def select_multiple_rois(img, num_rois):
    img_resized, scale = resize_to_screen(img)
    clone = img_resized.copy()
    selected_rois = []

    for i in range(num_rois):
        temp = clone.copy()
        for (x, y, w, h) in selected_rois:
            x_resized = int(x * scale)
            y_resized = int(y * scale)
            w_resized = int(w * scale)
            h_resized = int(h * scale)
            cv2.rectangle(temp, (x_resized, y_resized), (x_resized + w_resized, y_resized + h_resized), (0, 0, 255), 2)

        r = cv2.selectROI(f"Wybierz ROI {i+1} (ESC = pomiń)", temp, fromCenter=False, showCrosshair=True)

        if r == (0, 0, 0, 0):
            print(f"ROI {i+1} pominięte.")
            continue
        
        x, y, w, h = r
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

        selected_rois.append((x, y, w, h))

    cv2.destroyAllWindows()
    return selected_rois

def main():
    # Wybór folderu
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Wybierz folder z obrazami")
    if not folder_path:
        print("Nie wybrano folderu.")
        return

    # Pobranie obrazów
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    if not image_files:
        print("Brak obrazów w folderze.")
        return

    # Zapytanie o liczbę ROI
    num_rois = simpledialog.askinteger("Liczba ROI", "Ile ROI chcesz zaznaczyć?", minvalue=1, maxvalue=100)
    if not num_rois:
        print("Nie podano liczby ROI.")
        return

    # Wczytanie pierwszego obrazu
    first_image_path = os.path.join(folder_path, image_files[0])
    img = cv2.imread(first_image_path)
    if img is None:
        print("Nie udało się wczytać obrazu.")
        return

    # Zaznaczanie ROI
    rois = select_multiple_rois(img, num_rois)
    if not rois:
        print("Nie zaznaczono żadnych ROI.")
        return

    # Tworzenie podfolderów
    output_folders = []
    for idx in range(len(rois)):
        out_folder = os.path.join(folder_path, f"roi_{idx+1}")
        os.makedirs(out_folder, exist_ok=True)
        output_folders.append(out_folder)

    # Przetwarzanie wszystkich obrazów
    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        for idx, (x, y, w, h) in enumerate(rois):
            roi_crop = img[y:y+h, x:x+w]
            out_path = os.path.join(output_folders[idx], filename)
            cv2.imwrite(out_path, roi_crop)

    print("Gotowe!")

if __name__ == "__main__":
    main()
