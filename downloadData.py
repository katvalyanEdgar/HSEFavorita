from pathlib import Path
import zipfile

import kagglehub


RAW_DIR = Path("data/raw")
COMPETITION = "favorita-grocery-sales-forecasting"


RAW_DIR.mkdir(parents=True, exist_ok=True)

path = Path(kagglehub.competition_download(COMPETITION, output_dir=str(RAW_DIR)))
print("путь к файлам соревнования:", path)

for archive_path in RAW_DIR.glob("*.zip"):
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(RAW_DIR)

print("файлы в data/raw:")
for file_path in sorted(RAW_DIR.iterdir()):
    print("-", file_path.name)
