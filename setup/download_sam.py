import os
from urllib.request import urlopen, Request
from tqdm import tqdm

def download_model():
    os.makedirs("models", exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    target_path = os.path.join("models", "sam_vit_h.pth")

    if os.path.exists(target_path):
        print("Model already exists at models/sam_vit_h.pth.")
        return

    print("Downloading SAM model...")

    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(req) as response:
        total = int(response.getheader('Content-Length').strip())
        with open(target_path, 'wb') as f, tqdm(
            total=total, unit='B', unit_scale=True, desc="Downloading", ncols=80
        ) as pbar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

    print("Download completed and saved as sam_vit_h.pth.")

if __name__ == "__main__":
    download_model()
