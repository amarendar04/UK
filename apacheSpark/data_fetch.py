from pathlib import Path
import shutil

import kagglehub


# Download latest version
download_path = Path(kagglehub.dataset_download("blastchar/telco-customer-churn"))

# Keep a copy in the same folder as this script.
script_dir = Path(__file__).resolve().parent
csv_files = sorted(download_path.glob("*.csv"))

if not csv_files:
	raise FileNotFoundError(f"No CSV file found in {download_path}")

source_file = csv_files[0]
target_file = script_dir / source_file.name
shutil.copy2(source_file, target_file)

print("Downloaded dataset folder:", download_path)
print("Local CSV copy:", target_file)