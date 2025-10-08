import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from osgeo import gdal
import requests
import platform
import ee
from google.oauth2 import service_account
import re
from collections import defaultdict
import csv


# ------------------------------
# CONFIGURATION
# ------------------------------
PLANET_PBF_URL = "https://planet.openstreetmap.org/pbf"
OUTPUT_BASE_DIR = Path("flii_outputs")
CONFIG_PATH = Path("osm_config.json")

# ------------------------------
# GOOGLE CLOUD / EARTH ENGINE CONFIGURATION
# ------------------------------

def load_env(filepath=".env"):
    """Simple parser for .env files."""
    if not os.path.exists(filepath):
        print(f"[WARN] .env file not found: {filepath}")
        return

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

# Load environment variables
load_env()

# Access them
GCP_PROJECT = os.getenv("GCP_PROJECT")
GCS_BUCKET = os.getenv("GCS_BUCKET")
SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT")
SERVICE_KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
EE_ASSET_ROOT = os.getenv("EE_ASSET_ROOT")

# ------------------------------
# UTILITIES
# ------------------------------

def activate_gcloud_service_account():
    """Activate GCP service account for gcloud/gsutil commands."""
    print(f"[GCP AUTH] Activating service account for gcloud: {SERVICE_ACCOUNT}")

    # Look for gcloud executable
    gcloud_path = shutil.which("gcloud") or shutil.which("gcloud.cmd") or shutil.which("gcloud.CMD")

    if gcloud_path is None:
        print("[GCP AUTH ERROR] 'gcloud' not found in PATH.")
        print("Install it from: https://cloud.google.com/sdk/docs/install")
        return

    try:
        # Activate service account
        subprocess.run([
            gcloud_path,
            "auth", "activate-service-account", SERVICE_ACCOUNT,
            "--key-file", SERVICE_KEY_PATH
        ], check=True)

        # Set the project
        subprocess.run([
            gcloud_path,
            "config", "set", "project", GCP_PROJECT
        ], check=True)

        # Verify authentication
        subprocess.run([gcloud_path, "auth", "list"], check=True)

        print("[GCP AUTH] Service account activated and project set successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[GCP AUTH ERROR] Failed to activate service account: {e}")


def init_google_earth_engine():
    """Authenticate and initialize Earth Engine using a service account."""
    activate_gcloud_service_account()  # 🔹 ensure gsutil & gcloud use the right account

    print(f"[AUTH] Authenticating with service account: {SERVICE_ACCOUNT}")
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_KEY_PATH, scopes=["https://www.googleapis.com/auth/earthengine"]
    )
    ee.Initialize(credentials, project=GCP_PROJECT)
    print(f"[AUTH] Earth Engine initialized for project: {GCP_PROJECT}")

def _find_osm_pbf_url(task_year: int, max_age_days: int = 60) -> Tuple[str, str]:
    """
    Search for the closest available OSM .pbf snapshot around {year}-12-31.
    Tries multiple repositories automatically.
    Returns (url, date_str) if found, else raises ValueError.
    """
    taskdate = datetime(task_year, 12, 31)
    candidates = [
        ("https://ftp.fau.de/osm-planet/pbf/planet-{planetdate}.osm.pbf", "fau.de"),
        ("https://planet.openstreetmap.org/pbf/planet-{planetdate}.osm.pbf", "planet.osm.org"),
    ]

    for days_back in range(max_age_days + 1):
        check_date = taskdate - timedelta(days=days_back)
        date_str = check_date.strftime("%y%m%d")
        for base_url, source in candidates:
            url = base_url.format(planetdate=date_str)
            print(f"[CHECK] Trying {url} ...", end="")
            try:
                r = requests.head(url, allow_redirects=True, timeout=10)
                if r.status_code == 200:
                    print(" found ✅")
                    return url, check_date.strftime("%Y-%m-%d")
            except requests.RequestException:
                pass
            print(" not found")

    # fallback to latest
    latest_url = "https://ftp.fau.de/osm-planet/pbf/planet-latest.osm.pbf"
    print(f"[FALLBACK] Using latest available: {latest_url}")
    return latest_url, "latest"


def download_pbf(year: int, dest_path: Path) -> Path:
    """
    Download or reuse cached OSM PBF file for a given year (~Dec 31 snapshot).
    Logs metadata about the selected file and source.
    """
    metadata_path = dest_path.parent / "osm_download_metadata.json"

    # If cached file exists, skip download but refresh metadata
    if dest_path.exists():
        size_mb = round(os.path.getsize(dest_path) / (1024 * 1024), 2)
        print(f"[CACHE] Using cached PBF for {year}: {dest_path} ({size_mb} MB)")
        url, found_date = _find_osm_pbf_url(year)  # refresh metadata info
        metadata = {
            "year_requested": year,
            "download_date": date.today().isoformat(),
            "pbf_url": url,
            "pbf_date_found": found_date,
            "size_mb": size_mb,
            "cached": True
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[CACHE] Metadata refreshed: {metadata_path}")
        return dest_path

    # Otherwise, search and download fresh
    url, found_date = _find_osm_pbf_url(year)
    print(f"[DOWNLOAD] Using PBF file for {year}: {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(response.raw, f)

    size_mb = round(os.path.getsize(dest_path) / (1024 * 1024), 2)
    metadata = {
        "year_requested": year,
        "download_date": date.today().isoformat(),
        "pbf_url": url,
        "pbf_date_found": found_date,
        "size_mb": size_mb,
        "cached": False
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Downloaded and logged metadata to {metadata_path}")
    return dest_path

def osmium_to_text(pbf_path: Path, txt_path: Path) -> Path:
    """Convert .pbf file to osmium text format."""
    cmd = [
        "osmium", "export",
        "-f", "text",
        "-o", str(txt_path),
        str(pbf_path)
    ]
    print(f"[OSMIUM] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return txt_path

def split_text_to_csv(txt_file, csv_dir, config):

    os.makedirs(csv_dir, exist_ok=True)

    target_tags = set(config.keys())
    print(f"[CONFIG] Loaded {len(target_tags)} tag=value weights from config")

    tag_geoms = defaultdict(list)

    with open(txt_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.startswith("POINT("):
                continue

            # Extract coordinates
            coord_match = re.match(r'^POINT\(([-\d\.]+) ([-\d\.]+)\)', line)
            if not coord_match:
                continue

            lon, lat = coord_match.groups()

            # Extract tags
            tags = dict(re.findall(r'@?([\w\:\-]+)=([\w\-\.\%\:/]+)', line))

            # if i < 100:
            #     print(f"[DEBUG] Line {i+1}: tags = {tags}")

            for tag_val in target_tags:
                if "=" not in tag_val:
                    print(f"[WARNING] Skipping malformed config key: {tag_val}")
                    continue
                tag, val = tag_val.split("=", 1)
                if tag in tags and tags[tag] == val:
                    tag_geoms[tag_val].append((lon, lat))

    total = sum(len(v) for v in tag_geoms.values())
    print(f"[DONE] Processed {i+1:,} lines, matched {total:,} features")

    csv_files = []
    for tag_val, points in tag_geoms.items():
        tag_safe = tag_val.replace(":", "_").replace("/", "_")
        out_path = os.path.join(csv_dir, f"{tag_safe}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as out_csv:
            writer = csv.writer(out_csv)
            writer.writerow(["WKT", "BURN"])
            for lon, lat in points:
                writer.writerow([f"POINT({lon} {lat})", 1])
        csv_files.append(out_path)

    return csv_files

def rasterize_all(csvs: List[Path], out_dir: Path, bounds: Tuple[float,float,float,float], res: float) -> List[Path]:
    """
    Rasterize each CSV into a GeoTIFF using gdal_rasterize.
    Each CSV has columns lon,lat,BURN.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rasters = []

    xmin, ymin, xmax, ymax = bounds

    for csv in csvs:
        safe_name = csv.stem.replace("=", "_") + ".tif"
        out_tif = out_dir / safe_name
        args = [
            "gdal_rasterize",
            "-a", "BURN",
            "-tr", str(res), str(res),
            "-te", str(xmin), str(ymin), str(xmax), str(ymax),
            "-a_srs", "EPSG:4326",
            "-ot", "Int16",              # força saída Int16
            "-co", "COMPRESS=LZW",       # compressão
            str(csv), str(out_tif)
        ]

        print(f"[RASTERIZE] {csv} -> {out_tif}")
        subprocess.run(args, check=True)
        rasters.append(out_tif)

    return rasters

def run_gdal_calc(rasters: List[Path], categories: Dict, out_path: Path):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    args = [
        sys.executable, "-m", "osgeo_utils.gdal_calc",
        "--overwrite",
        "--outfile", str(out_path),
        "--co=COMPRESS=LZW",
        "--type=Int16"   # evitar overflow
    ]

    expr_terms = []
    for i, raster in enumerate(rasters):
        if i >= len(letters):
            raise ValueError("Too many rasters for one gdal_calc call (max 26)")
        letter = letters[i]        # maiúscula
        args.append(f"-{letter}")
        args.append(str(raster))

        weight = 1
        for tagval, w in categories.items():
            tag_safe = tagval.replace(":", "_").replace("/", "_")
            if tag_safe in raster.name:
                weight = w
                break

        expr_terms.append(f"{weight}*{letter}")

    args.append(f"--calc={' + '.join(expr_terms)}")
    print("[DEBUG] gdal_calc expression:", args[-1])   # debug útil
    subprocess.run(args, check=True)

def apply_weights_merge(rasters: List[Path], categories: Dict, merged_path: Path):
    """
    Merge rasters with weights applied, handling >26 rasters by chunking.
    Cleans up intermediate files after the final merge.
    """
    if not rasters:
        print("No rasters to merge.")
        return

    print(f"[MERGE] Found {len(rasters)} rasters...")

    # Case 1: small number of rasters
    if len(rasters) <= 26:
        run_gdal_calc(rasters, categories, merged_path)
        print(f"Done. Final merged raster: {merged_path}")
        return

    # Case 2: too many rasters → chunk into groups
    temp_files = []
    chunk_size = 20  # safe margin under 26
    for i in range(0, len(rasters), chunk_size):
        chunk = rasters[i:i+chunk_size]
        tmp_out = merged_path.parent / f"merge_part{i//chunk_size+1}.tif"
        run_gdal_calc(chunk, categories, tmp_out)
        temp_files.append(tmp_out)

    # Final merge from intermediate results
    run_gdal_calc(temp_files, categories, merged_path)
    print(f"Done. Final merged raster: {merged_path}")

    # Cleanup intermediate files
    for tmp in temp_files:
        try:
            os.remove(tmp)
            print(f"[CLEANUP] Removed {tmp}")
        except Exception as e:
            print(f"[CLEANUP] Could not remove {tmp}: {e}")

import shutil
import subprocess

def upload_to_gcs(local_path: Path, gcs_uri: str):
    """Upload a local raster to Google Cloud Storage using gsutil."""
    print(f"[UPLOAD] {local_path.name} → {gcs_uri}")

    # Find gsutil executable (handle .cmd and .CMD on Windows)
    gsutil_path = shutil.which("gsutil") or shutil.which("gsutil.cmd") or shutil.which("gsutil.CMD")

    if gsutil_path is None:
        print("[ERROR] gsutil not found. Make sure Google Cloud SDK is installed and in your PATH.")
        raise FileNotFoundError("gsutil executable not found.")

    try:
        cmd = [gsutil_path, "cp", str(local_path), gcs_uri]
        subprocess.run(cmd, check=True)
        print("[UPLOAD] Uploaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Upload failed for {local_path.name}: {e}")
        raise

def import_to_earth_engine(asset_id: str, gcs_uri: str):
    """Import a GeoTIFF from GCS to Earth Engine as an asset using the EE CLI."""
    print(f"[EE IMPORT] {gcs_uri} → {asset_id}")

    cmd = [
        "earthengine", "upload", "image",
        f"--asset_id={asset_id}",
        "--pyramiding_policy=MEAN",
        gcs_uri
    ]

    try:
        subprocess.run(cmd, check=True)
        print("[EE IMPORT] Upload task started.")
    except FileNotFoundError:
        print("[ERROR] Earth Engine CLI not found. Install via `pip install earthengine-api`.")
        raise
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Earth Engine import failed: {e}")
        raise

def export_rasters_to_gee(raster_dir: Path, year: int):
    """
    Upload all rasters (individual + merged) to Cloud Storage,
    and import them to Earth Engine.
    """
    print(f"[EXPORT] Uploading rasters for {year} to GCS and importing to Earth Engine...")
    tifs = list(raster_dir.glob("*.tif"))
    if not tifs:
        print("No rasters found to export.")
        return

    exported_assets = []

    for tif in tifs:
        gcs_uri = f"gs://{GCS_BUCKET}/{year}/{tif.name}"
        asset_id = f"{EE_ASSET_ROOT}/{year}/{tif.stem}"

        try:
            # 1. Upload to bucket
            upload_to_gcs(tif, gcs_uri)

            # 2. Import to Earth Engine
            import_to_earth_engine(asset_id, gcs_uri)

            exported_assets.append({
                "filename": tif.name,
                "asset_id": asset_id,
                "gcs_uri": gcs_uri,
                "status": "imported"
            })

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed for {tif.name}: {e}")
            exported_assets.append({
                "filename": tif.name,
                "asset_id": asset_id,
                "gcs_uri": gcs_uri,
                "status": f"failed - {e}"
            })

    # Save summary metadata locally and to GCS
    metadata_path = raster_dir / f"gee_export_metadata_{year}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(exported_assets, f, indent=2)

    gcs_metadata_uri = f"gs://{GCS_BUCKET}/{year}/gee_export_metadata_{year}.json"
    try:
        subprocess.run(["gsutil", "cp", str(metadata_path), gcs_metadata_uri], check=True)
        print(f"[METADATA] Export summary uploaded: {gcs_metadata_uri}")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Could not upload metadata to GCS: {e}")

    print(f"Export summary saved: {gcs_metadata_uri}")


# ------------------------------
# MAIN LOGIC
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=date.today().year)
    parser.add_argument("--resolution", type=float, default=0.00269)
    parser.add_argument("--bounds", nargs=4, type=float, metavar=("xmin", "ymin", "xmax", "ymax"),
                        default=[-180, -90, 180, 90])
    args = parser.parse_args()

    year = args.year
    res = args.resolution
    bounds = tuple(args.bounds)

    year_dir = OUTPUT_BASE_DIR / f"{year}"
    txt_file = year_dir / f"osm_{year}.txt"
    csv_dir = year_dir / "csvs"
    raster_dir = year_dir / "rasters"
    merged_path = year_dir / f"flii_osm_{year}_merged.tif"
    pbf_path = year_dir / f"osm_{year}.osm.pbf"

    for d in [csv_dir, raster_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Access the nested 'weights' dictionary
    categories = config.get("osm_categories", {}).get("weights", {})

    if not pbf_path.exists():
        download_pbf(year, pbf_path)

    if not txt_file.exists():
        osmium_to_text(pbf_path, txt_file)

    print("[1] Splitting text into CSVs...")
    csvs = split_text_to_csv(txt_file, csv_dir, categories)
    csvs = [Path(f) for f in csvs]

    print("[2] Rasterizing CSVs...")
    rasters = rasterize_all(csvs, raster_dir, bounds, res)

    print("[3] Applying weights + merging...")
    apply_weights_merge(rasters, categories, merged_path)

    print("[4] Exporting rasters to Google Earth Engine...")
    init_google_earth_engine()
    export_rasters_to_gee(raster_dir, year)


if __name__ == "__main__":
    main()
