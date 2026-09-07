from pathlib import Path
import zipfile

import kagglehub
import py7zr


RAW_DIR = Path("data/raw")
DOWNLOAD_DIR = RAW_DIR / "_download"
COMPETITION = "favorita-grocery-sales-forecasting"


RAW_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

path = Path(kagglehub.competition_download(COMPETITION, output_dir=str(DOWNLOAD_DIR), force_download=True))
print("путь к файлам соревнования:", path)

for archive_path in DOWNLOAD_DIR.glob("*.zip"):
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(RAW_DIR)

for archive_path in DOWNLOAD_DIR.glob("*.7z"):
    with py7zr.SevenZipFile(archive_path) as archive:
        archive.extractall(RAW_DIR)

print("файлы в data/raw:")
for file_path in sorted(RAW_DIR.iterdir()):
    print("-", file_path.name)
