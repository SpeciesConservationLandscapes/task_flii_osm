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
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------
# CONFIGURATION
# ------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE_DIR = PROJECT_ROOT / "flii_outputs"
CONFIG_PATH = PROJECT_ROOT / "src" / "osm_config.json"
PLANET_PBF_URL = "https://planet.openstreetmap.org/pbf"

# ------------------------------
# GOOGLE CLOUD / EARTH ENGINE CONFIGURATION
# ------------------------------

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

    gcloud_path = shutil.which("gcloud") or shutil.which("gcloud.cmd") or shutil.which("gcloud.CMD")

    if gcloud_path is None:
        print("[GCP AUTH ERROR] 'gcloud' not found in PATH.")
        print("Install it from: https://cloud.google.com/sdk/docs/install")
        return

    key_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_env:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set in the environment.")

    # Handle both inline JSON and file path
    if key_env.strip().startswith("{"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write(key_env)
            tmp_path = tmp.name
        key_file = tmp_path
        print(f"[GCP AUTH] Wrote inlined service account JSON to temporary file: {key_file}")
    else:
        key_file = key_env

    try:
        # Activate service account
        subprocess.run([
            gcloud_path,
            "auth", "activate-service-account", SERVICE_ACCOUNT,
            "--key-file", key_file
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
    finally:
        # Clean up temporary file if created
        if key_env.strip().startswith("{") and os.path.exists(key_file):
            os.remove(key_file)
            print(f"[GCP AUTH] Removed temporary key file: {key_file}")


def init_google_earth_engine():
    """Authenticate and initialize Earth Engine using a service account."""
    activate_gcloud_service_account()  # ensure gsutil & gcloud use the right account

    print(f"[AUTH] Authenticating with service account: {SERVICE_ACCOUNT}")
    if SERVICE_KEY_PATH is None:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set in the environment.")

    # Detect whether it's a JSON string or a file path
    if SERVICE_KEY_PATH.strip().startswith("{"):
        # JSON string or dict passed directly
        print("[AUTH] Using inlined service account JSON.")
        key_info = json.loads(SERVICE_KEY_PATH)
        credentials = service_account.Credentials.from_service_account_info(
            key_info, scopes=["https://www.googleapis.com/auth/earthengine"]
        )
    else:
        # File path passed
        print(f"[AUTH] Using service account file: {SERVICE_KEY_PATH}")
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
    Each CSV has columns WKT,BURN.
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

def run_gdal_calc(rasters: List[Path], categories: Dict, out_path: Path, weighted: bool = True):
    """
    Combine multiple rasters into one using gdal_calc.py.
    If weighted=True, apply category weights.
    If weighted=False, simply sum rasters.
    """
    if not rasters:
        raise ValueError("No rasters provided to run_gdal_calc")

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(rasters) > len(letters):
        raise ValueError(f"Too many rasters for one gdal_calc call ({len(rasters)} > 26)")

    args = [
        "gdal_calc.py",
        "--overwrite",
        "--outfile", str(out_path),
        "--co=COMPRESS=LZW",
        "--type=Int16"
    ]

    expr_terms = []
    for i, raster in enumerate(rasters):
        letter = letters[i]
        args.append(f"-{letter}")
        args.append(str(raster))

        weight = 1
        if weighted:
            for tagval, w in categories.items():
                tag_safe = tagval.replace(":", "_").replace("/", "_")
                if tag_safe in raster.name:
                    weight = w
                    break
        expr_terms.append(f"{weight}*{letter}")

    args.append(f"--calc={' + '.join(expr_terms)}")
    print(f"[DEBUG] gdal_calc expression: {args[-1]}")
    subprocess.run(args, check=True)

def apply_weights_merge(rasters: List[Path], categories: Dict, merged_path: Path):
    """
    Merge rasters with weights applied, handling >26 rasters by chunking and parallelizing.
    - Weighted merges for first-level chunks
    - Simple sum for higher-level merges
    - Automatically recurses if >26 intermediates
    """
    if not rasters:
        print("No rasters to merge.")
        return

    print(f"[MERGE] Found {len(rasters)} rasters...")

    # Helper for chunk merging
    def merge_chunk(i, chunk):
        tmp_out = merged_path.parent / f"merge_part{i+1}.tif"
        print(f"[CHUNK {i+1}] Merging {len(chunk)} rasters → {tmp_out}")
        run_gdal_calc(chunk, categories, tmp_out, weighted=True)
        return tmp_out

    # Base case: 26 or fewer rasters
    if len(rasters) <= 26:
        run_gdal_calc(rasters, categories, merged_path, weighted=True)
        print(f"Final merged raster: {merged_path}")
        return

    # Chunking setup
    chunk_size = 20
    chunks = [rasters[i:i+chunk_size] for i in range(0, len(rasters), chunk_size)]
    temp_files = []

    # Run merges in parallel (I/O-bound → threads are fine)
    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
        futures = {executor.submit(merge_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            try:
                temp_files.append(future.result())
            except Exception as e:
                print(f"[ERROR] Chunk {futures[future]+1} failed: {e}")

    print(f"[MERGE] {len(temp_files)} intermediate rasters created.")

    # If the intermediate rasters still exceed 26, recurse
    if len(temp_files) > 26:
        print("[MERGE] Too many intermediates, merging recursively...")
        apply_weights_merge(temp_files, categories, merged_path)
        return

    # Final simple merge (unweighted)
    print("[MERGE] Combining intermediate rasters (unweighted)...")
    run_gdal_calc(temp_files, categories, merged_path, weighted=False)
    print(f"✅ Done. Final merged raster: {merged_path}")

    # Cleanup
    for tmp in temp_files:
        try:
            os.remove(tmp)
            print(f"[CLEANUP] Removed {tmp}")
        except Exception as e:
            print(f"[CLEANUP] Could not remove {tmp}: {e}")

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

def ensure_ee_folder_exists(folder_path: str):
    """
    Ensure that an Earth Engine asset folder exists.
    If it doesn't, create it recursively.
    """
    try:
        ee.data.getAsset(folder_path)
        print(f"[EE FOLDER] Exists: {folder_path}")
    except ee.ee_exception.EEException:
        parent = "/".join(folder_path.split("/")[:-1])
        if parent and not parent.startswith("projects/"):
            parent = f"projects/{GCP_PROJECT}/assets/{parent}"
        # Recursively ensure parent exists first
        if parent and parent != folder_path:
            ensure_ee_folder_exists(parent)
        print(f"[EE FOLDER] Creating: {folder_path}")
        ee.data.createAsset({'type': 'Folder'}, folder_path)

def import_to_earth_engine(asset_id: str, gcs_uri: str):
    """Import a GeoTIFF from GCS to Earth Engine using the Python API (service account safe)."""
    print(f"[EE IMPORT] {gcs_uri} → {asset_id}")

    try:
        # Ensure parent folder exists
        folder_path = "/".join(asset_id.split("/")[:-1])
        ensure_ee_folder_exists(folder_path)

        # Check if asset already exists
        try:
            ee.data.getAsset(asset_id)
            print(f"[EE IMPORT] Asset already exists: {asset_id} — skipping upload.")
            return
        except ee.ee_exception.EEException:
            pass

        params = {
            'id': asset_id,
            'tilesets': [{'sources': [{'uris': [gcs_uri]}]}],
            'bands': [{'id': 'b1'}],
            'pyramidingPolicy': 'MEAN'
        }

        task = ee.data.startIngestion(str(uuid.uuid4()), params)
        print(f"[EE IMPORT] Ingestion task started: {task['id']}")

    except Exception as e:
        print(f"[EE IMPORT ERROR] Failed to import {asset_id}: {e}")
        raise

def export_rasters_to_gee(raster_dir: Path, year: int):
    """
    Upload all rasters (individual + merged) to Cloud Storage,
    and import them to Earth Engine.
    """
    print(f"[EXPORT] Uploading rasters for {year} to GCS and importing to Earth Engine...")

    # Collect both per-feature rasters and the final merged one
    tifs = list(raster_dir.glob("*.tif"))

    # Also include the final merged raster at year_dir level if it exists
    merged_candidate = raster_dir.parent / f"flii_infra_{year}.tif"
    if merged_candidate.exists():
        tifs.append(merged_candidate)
        print(f"[EXPORT] Including merged raster: {merged_candidate.name}")

    if not tifs:
        print("[EXPORT] No rasters found to export.")
        return

    exported_assets = []

    for tif in tifs:
        # Normalize name and remote paths
        gcs_uri = f"gs://{GCS_BUCKET}/{year}/{tif.name}"
        asset_id = f"{EE_ASSET_ROOT}/{year}/{tif.stem}"

        try:
            # Upload to Cloud Storage
            upload_to_gcs(tif, gcs_uri)

            # Import to Earth Engine
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

    # Save summary metadata locally and upload to GCS
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
    merged_path = year_dir / f"flii_infra_{year}.tif"
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
