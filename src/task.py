import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from osgeo import gdal, gdalconst
import requests
import ee
from google.oauth2 import service_account
import re
from collections import defaultdict
import csv
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
from contextlib import contextmanager
import math
from tqdm import tqdm
import itertools

# ------------
# GDAL CONFIG 
# ------------
gdal.UseExceptions()
gdal.SetConfigOption("GDAL_CACHEMAX", "512")
gdal.SetConfigOption("GDAL_SWATH_SIZE", "512")
gdal.SetConfigOption("GDAL_MAX_DATASET_POOL_SIZE", "400")
os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"
os.environ["NUMEXPR_MAX_THREADS"] = str(os.cpu_count())

# ----------------------
# DIRECTORY / OSM CONFIG
# ----------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE_DIR = PROJECT_ROOT / "flii_outputs"
CONFIG_PATH = PROJECT_ROOT / "src" / "osm_config.json"
PLANET_PBF_URL = "https://planet.openstreetmap.org/pbf"

# ----------------------------------
# GOOGLE CLOUD / EARTH ENGINE CONFIG
# ----------------------------------

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCS_BUCKET = os.getenv("GCS_BUCKET")
SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT")
SERVICE_KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
EE_ASSET_ROOT = os.getenv("EE_ASSET_ROOT")

# ------------------------------
# UTILITIES
# ------------------------------

@contextmanager
def log_time(label: str):
    """
    Measures how long a code block takes to run.
    It prints the elapsed time when the block ends.
    Used in the 'main' function.
    """
    start = time.time()
    print(f"[TIMER] Starting: {label}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"[TIMER] Finished {label} — {elapsed/60:.2f} min ({elapsed:.1f} sec)")

def degrees_to_meters(res_deg: float) -> int:
    """
    Approximate degrees to meters conversion at the equator.
    Used to append resolution suffixes like '_300m' to raster filenames.
    """

    meters = res_deg * 111_320
    return int(math.ceil(meters / 10.0) * 10)

def activate_gcloud_service_account():
    """
    Activates the gcloud service account using the provided key file or inlined JSON. 
    """

    print(f"[GCP AUTH] Activating service account for gcloud: {SERVICE_ACCOUNT}")

    # Find gcloud in PATH.
    gcloud_path = shutil.which("gcloud") or shutil.which("gcloud.cmd") or shutil.which("gcloud.CMD")
    if gcloud_path is None:
        print("[GCP AUTH ERROR] 'gcloud' not found in PATH.")
        print("Install it from: https://cloud.google.com/sdk/docs/install")
        return
    
    # Get credentials from environment variable.
    key_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_env:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set in the environment.")

    if key_env.strip().startswith("{"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write(key_env)
            tmp_path = tmp.name
        key_file = tmp_path
        print(f"[GCP AUTH] Wrote inlined service account JSON to temporary file: {key_file}")
    else:
        key_file = key_env

    # Authenticate service account and set cloud project.
    try:
        subprocess.run([gcloud_path, "auth", "activate-service-account", SERVICE_ACCOUNT, "--key-file", key_file], check=True)
        subprocess.run([gcloud_path, "config", "set", "project", GCP_PROJECT], check=True)
        subprocess.run([gcloud_path, "auth", "list"], check=True)
        print("[GCP AUTH] Service account activated and project set successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[GCP AUTH ERROR] Failed to activate service account: {e}")
    finally:
        if key_env.strip().startswith("{") and os.path.exists(key_file):
            os.remove(key_file)
            print(f"[GCP AUTH] Removed temporary key file: {key_file}")

def init_google_earth_engine():
    """
    Authenticates with Google Earth Engine using the same service account.
    Initializes the Earth Engine API.
    Sets cloud project.
    """

    activate_gcloud_service_account()
    print(f"[AUTH] Authenticating with service account: {SERVICE_ACCOUNT}")

    if SERVICE_KEY_PATH is None:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set in the environment.")

    if SERVICE_KEY_PATH.strip().startswith("{"):
        print("[AUTH] Using inlined service account JSON.")
        key_info = json.loads(SERVICE_KEY_PATH)
        credentials = service_account.Credentials.from_service_account_info(
            key_info, scopes=["https://www.googleapis.com/auth/earthengine"]
        )
    else:
        print(f"[AUTH] Using service account file: {SERVICE_KEY_PATH}")
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_KEY_PATH, scopes=["https://www.googleapis.com/auth/earthengine"]
        )

    ee.Initialize(credentials, project=GCP_PROJECT)
    print(f"[AUTH] Earth Engine initialized for project: {GCP_PROJECT}")

def _find_osm_pbf_url(task_year: int, max_age_days: int = 60) -> Tuple[str, str]:
    """
    Finds the closest OSM file (.pbf or .bz2) for the year before the given year, 
    checking up to max_age_days.
    For example, if task_year=2024, it searches around 2023-12-31.
    It checks multiple known mirror URLs.
    """

    import requests
    from datetime import datetime, timedelta

    taskdate = datetime(task_year - 1, 12, 31)

    def _try_osm_urls(urlbase: str, maxage: int) -> Optional[Tuple[str, str]]:
        """
        Checks the given URL base for the OSM file, 
        trying dates from 'taskdate' back to 'taskdate - maxage days'.
        """
        for days_back in range(maxage + 1):
            check_date = taskdate - timedelta(days=days_back)
            params = {
                "year": check_date.strftime("%Y"),
                "planetdate": check_date.strftime("%y%m%d"),
            }
            url = urlbase.format(**params)
            print(f"[CHECK] Trying {url} ...", end="")
            try:
                r = requests.head(url, allow_redirects=False, timeout=10)
                if r.status_code == requests.codes.ok:
                    print(" found")
                    return url, check_date.strftime("%Y-%m-%d")
                if r.status_code == requests.codes.found and "Location" in r.headers:
                    redirected_url = r.headers["Location"]
                    new_response = requests.head(redirected_url, timeout=10)
                    if new_response.status_code == requests.codes.ok:
                        print(f" redirected ({redirected_url})")
                        return redirected_url, check_date.strftime("%Y-%m-%d")
            except requests.RequestException as e:
                print(f" error ({e.__class__.__name__})", end="")
            print(" not found")
        return None

    faude_pbf = "https://ftp.fau.de/osm-planet/pbf/planet-{planetdate}.osm.pbf"
    planet_osm_archive = "https://planet.openstreetmap.org/planet/{year}/planet-{planetdate}.osm.bz2"

    # Try Faude.de for 2020+ (more frequent updates).
    if task_year >= 2020:
        result = _try_osm_urls(faude_pbf, maxage=6)
        if result:
            return result

    # Try Planet OSM archive for older years.
    result = _try_osm_urls(planet_osm_archive, maxage=max_age_days)
    if result:
        return result

    raise ValueError(
        f"No OSM .pbf or .bz2 snapshot found within {max_age_days} days before {task_year}-12-31.\n"
        f"Tried:\n  - {faude_pbf}\n  - {planet_osm_archive}\n"
        f"Check the planet archive at https://planet.openstreetmap.org/planet/{task_year}/"
    )

def download_pbf(year: int, dest_path: Path) -> Path:
    """
    Downloads the OSM file (.pbf or .bz2) for the specified year to dest_path.
    Supports resuming partial downloads if the server allows it.
    Stores metadata (size, URL, date) about the download in a JSON file alongside the data.
    """

    metadata_path = dest_path.parent / "osm_download_metadata.json"
    url, found_date = _find_osm_pbf_url(year)
    print(f"[DOWNLOAD] Preparing to fetch {url}")

    dest_stem = dest_path.stem
    while any(dest_stem.endswith(ext) for ext in [".osm", ".pbf", ".bz2"]):
        dest_stem = Path(dest_stem).stem
    if url.endswith(".bz2"):
        dest_path = dest_path.with_name(dest_stem + ".osm.bz2")
    elif url.endswith(".pbf"):
        dest_path = dest_path.with_name(dest_stem + ".osm.pbf")
    else:
        raise ValueError(f"Unsupported OSM file type in URL: {url}")

    existing_size = dest_path.stat().st_size if dest_path.exists() else 0
    head = requests.head(url, allow_redirects=True, timeout=10)
    total_size = int(head.headers.get("Content-Length", 0))
    supports_range = head.headers.get("Accept-Ranges", "none") != "none"

    if existing_size > 0 and existing_size < total_size:
        if supports_range:
            print(f"[RESUME] Partial file detected ({existing_size / 1e6:.2f} MB). Resuming...")
            headers = {"Range": f"bytes={existing_size}-"}
            mode = "ab"
        else:
            print("[RESUME] Server does not support partial downloads. Restarting from scratch.")
            existing_size = 0
            headers = {}
            mode = "wb"
    else:
        headers = {}
        mode = "wb"

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        chunk_size = 1024 * 1024
        with open(dest_path, mode) as f, tqdm(
            total=math.ceil(total_size / chunk_size) if total_size else None,
            initial=math.ceil(existing_size / chunk_size) if existing_size else 0,
            unit="MB",
            desc=f"Downloading OSM {year}",
            ascii=True,
            ncols=80,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    if total_size:
                        pbar.update(1)

    final_size = dest_path.stat().st_size
    if total_size > 0 and final_size != total_size:
        raise IOError(
            f"[ERROR] Download incomplete: expected {total_size} bytes, got {final_size} bytes"
        )

    print(f"[DOWNLOAD] Completed: {dest_path} ({final_size / 1e6:.2f} MB)")

    metadata = {
        "year_requested": year,
        "download_date": date.today().isoformat(),
        "pbf_url": url,
        "pbf_date_found": found_date,
        "size_mb": round(final_size / (1024 * 1024), 2),
        "cached": False,
        "resumed": existing_size > 0,
        "compressed": url.endswith(".bz2"),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[METADATA] Saved to: {metadata_path}")

    return dest_path

def osmium_to_text(osm_path: Path, txt_dir: Path, config: Dict) -> Path:
    """
    Convert an OSM PBF/BZ2 file directly to text using osmium export.
    Optimized for global runs: filters by relevant keys only, no intermediate PBF.
    Compatible with Osmium versions that expect JSON configs.
    """
    
    if not osm_path.exists():
        raise FileNotFoundError(f"[ERROR] OSM input file not found: {osm_path}")

    txt_dir.mkdir(parents=True, exist_ok=True)
    out_path = txt_dir / f"{osm_path.stem}.txt"

    # Extract unique OSM keys from config (e.g., highway, railway, man_made).
    keys = sorted({k.split("=")[0] for k in config.keys()})

    # Build JSON configuration.
    cfg = {"filters": [{"key": k} for k in keys]}
    cfg_path = txt_dir / "osmium_keys.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"[OSMIUM] Exporting {osm_path.name} → {out_path}")
    print(f"[OSMIUM] Generated JSON filter config for {len(keys)} keys:")
    print(", ".join(keys[:20]) + ("..." if len(keys) > 20 else ""))

    cmd = [
        "osmium", "export",
        "--progress",
        "--overwrite",
        "-c", str(cfg_path),
        "-f", "text",
        "-o", str(out_path),
        str(osm_path)
    ]

    # Use stdbuf for unbuffered output if available
    if shutil.which("stdbuf"):
        cmd = ["stdbuf", "-oL", "-eL"] + cmd

    # Force progress even in non-TTY Docker environments
    env = os.environ.copy()
    env["OSMIUM_SHOW_PROGRESS"] = "1"

    # Run and stream live progress (handles both \r and \n)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
        env=env
    )

    while True:
        chunk = process.stderr.read(1)
        if not chunk:
            break
        sys.stdout.write(chunk)
        sys.stdout.flush()

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

    print(f"[DONE] Export complete → {out_path}")

    try:
        cfg_path.unlink()
    except Exception:
        pass

    return out_path

def split_text_to_csv_streaming(txt_file: Union[str, Path], csv_dir: Union[str, Path], config: Dict, max_rows:int = 1_000_000) -> List[Path]:
    """
    Stream the OSM text and write per-tag CSVs in shards up to max_rows each.
    For each text line (geometry + tags):
    - Extracts matching tags from the osm_config.json
    - Writes each tag's features to its own CSV shard 
    Each CSV has columns: "WKT" (geometry) and "BURN" (the tag's numeric weight).
    This allows flexible per-tag rasterization later.
    Shards are named like highway_residential_001.csv, highway_residential_002.csv, etc.
    """
    
    os.makedirs(csv_dir, exist_ok=True)
    target_tags = set(config.keys())  # e.g., {"highway=primary": 9, ...}
    tag_files: Dict[str, csv.writer] = {}
    tag_fhandles: Dict[str, any] = {}
    tag_shard_counts: Dict[str, int] = defaultdict(int)
    tag_row_counts: Dict[str, int] = defaultdict(int)

    out_paths: List[Path] = []

    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("POINT(") and not line.startswith("LINESTRING(") and not line.startswith("POLYGON("):
                continue

            geom_match = re.match(r'^(POINT|LINESTRING|POLYGON)\((.+)\)', line)
            if not geom_match:
                continue
            wkt_end = line.index(")") + 1
            wkt = line[:wkt_end]

            # Tags after geometry
            tags_str = line[wkt_end:]
            tags = dict(re.findall(r'@?([\w\:\-]+)=([\w\-\.\%\:/]+)', tags_str))

            for tag_val in target_tags:
                if "=" not in tag_val:
                    continue
                tag, val = tag_val.split("=", 1)
                if tags.get(tag) == val:
                    # open/rollover shard if needed
                    if (tag_val not in tag_files) or (tag_row_counts[tag_val] >= max_rows):
                        if tag_val in tag_fhandles:
                            tag_fhandles[tag_val].close()
                        safe = tag_val.replace(":", "_").replace("/", "_").replace("=", "_")
                        shard_idx = tag_shard_counts[tag_val] + 1
                        out_path = Path(csv_dir) / f"{safe}_{shard_idx:03d}.csv"
                        fh = open(out_path, "w", newline="", encoding="utf-8")
                        writer = csv.writer(fh)
                        writer.writerow(["WKT", "BURN"])
                        tag_files[tag_val] = writer
                        tag_fhandles[tag_val] = fh
                        tag_shard_counts[tag_val] = shard_idx
                        tag_row_counts[tag_val] = 0
                        out_paths.append(out_path)

                    # Bake the weight here.
                    burn = int(config[tag_val])
                    tag_files[tag_val].writerow([wkt, burn])
                    tag_row_counts[tag_val] += 1

    for fh in tag_fhandles.values():
        try:
            fh.close()
        except:
            pass

    print(f"[SPLIT] {txt_file} → {len(out_paths)} CSV shards")
    return out_paths

def _rasterize_single(in_csv: Union[str, Path], out_tif: Union[str, Path],
                      bounds: Tuple[float,float,float,float], res: float):
    """
    Rasterize a 2-column CSV (WKT,BURN) to a tiled, compressed Int16 GeoTIFF.
    """

    res_m = degrees_to_meters(res)
    out_tif = Path(out_tif)
    out_tif = out_tif.with_name(f"{out_tif.stem}_{res_m}m.tif")

    opts = gdal.RasterizeOptions(
        format="GTiff",
        outputType=gdalconst.GDT_Int16,
        initValues=0,
        burnValues=None, # read from attribute
        attribute="BURN",
        xRes=res,
        yRes=res,
        targetAlignedPixels=True,
        outputSRS="EPSG:4326",
        outputBounds=bounds,
        creationOptions=[
            "TILED=YES",
            "BLOCKXSIZE=1024",
            "BLOCKYSIZE=1024",
            "COMPRESS=LZW",
        ],
    )
    # Ensure directory
    Path(out_tif).parent.mkdir(parents=True, exist_ok=True)
    # Rasterize
    gdal.Rasterize(str(out_tif), str(in_csv), options=opts)
    return Path(out_tif)

def rasterize_all(csvs: List[Path], out_dir: Path, bounds: Tuple[float,float,float,float], res: float) -> List[Path]:
    """
    Run _rasterize_single in parallel for all CSVs using ProcessPoolExecutor.
    Produce one GeoTIFF per CSV shard in out_dir.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    out_tifs = [out_dir / (Path(c).stem + ".tif") for c in csvs]
    num_cpus = max(1, (os.cpu_count() or 2) - 1)
    print(f"[RASTERIZE] {len(csvs)} CSVs with {num_cpus} workers...")

    with ProcessPoolExecutor(max_workers=num_cpus) as exe:
        list(exe.map(_rasterize_single, csvs, out_tifs, itertools.repeat(bounds), itertools.repeat(res)))

    return out_tifs

def merge_tag_shards(raster_dir: Path) -> List[Path]:
    """
    Merge all tag-level shard rasters (e.g., highway_residential_001, _002)
    into a single raster per tag using gdal_calc.
    Returns a list of merged tag rasters.
    """

    rasters = sorted(raster_dir.glob("*.tif"))
    if not rasters:
        print("[TAG MERGE] No rasters found to merge by tag.")
        return []

    tag_groups: Dict[str, List[Path]] = defaultdict(list)
    for r in rasters:
        base = re.sub(r"_[0-9]{3}$", "", r.stem)  # strip trailing _001 etc.
        tag_groups[base].append(r)

    merged_paths: List[Path] = []
    for base, group in tag_groups.items():
        if len(group) == 1:
            merged_paths.append(group[0])
            continue
        print(f"[TAG MERGE] Merging {len(group)} shards for tag: {base}")
        out_path = raster_dir / f"{base}.tif"
        _calc_sum(group, out_path)
        merged_paths.append(out_path)
        # optional cleanup
        for g in group:
            if g != out_path:
                try:
                    g.unlink()
                except:
                    pass
    print(f"[TAG MERGE] Produced {len(merged_paths)} tag-level rasters.")
    return merged_paths

def _calc_sum(inputs: List[Path], out_path: Path):
    """
    Sum ≤26 (limit) rasters safely (A+B+...).
    Unsets NoData on all inputs and output to avoid propagation.
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" # 26.
    if len(inputs) == 0:
        raise ValueError("No rasters provided to _calc_sum")
    if len(inputs) > 26:
        raise ValueError("Too many inputs for a single gdal_calc call (max=26)")

    # Unset NoData before summing
    for r in inputs:
        subprocess.run(["gdal_edit.py", "-unsetnodata", str(r)], check=True)

    calc_expr = " + ".join([f"nan_to_num({chr(65+i)})" for i in range(len(inputs))])

    args = [
        "gdal_calc.py", "--quiet", "--overwrite",
        f"--outfile={str(out_path)}",
        "--type=Int16",
        "--co=COMPRESS=LZW",
        "--calc", calc_expr
    ]
    for i, r in enumerate(inputs):
        args.extend([f"-{chr(65+i)}", str(r)])
    subprocess.run(args, check=True)
    subprocess.run(["gdal_edit.py", "-unsetnodata", str(out_path)], check=True)


def _calc_sum_safe(inputs: List[Path], out_path: Path, chunk_size: int = 20):
    """
    Handles cases with more than 26 rasters by merging them in chunks in parallel.
    Runs several _calc_sum() calls concurrently and merges intermediate results
    until one final raster remains.
    """

    if len(inputs) == 0:
        raise ValueError("No rasters provided to _calc_sum_safe")

    # If only one input, just copy
    if len(inputs) == 1:
        shutil.copy(inputs[0], out_path)
        subprocess.run(["gdal_edit.py", "-unsetnodata", str(out_path)], check=True)
        return

    tmp_dir = out_path.parent / "_tmp_merge_chunks"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    current = inputs
    round_idx = 0
    num_threads = max(1, min(os.cpu_count() or 2, 16))  # limit to avoid I/O thrash

    while len(current) > 1:
        round_idx += 1
        next_round: List[Path] = []
        chunks = [current[i:i + chunk_size] for i in range(0, len(current), chunk_size)]
        print(f"[MERGE] Round {round_idx}: {len(current)} rasters → {len(chunks)} chunks (parallel {num_threads})")

        with ThreadPoolExecutor(max_workers=num_threads) as exe:
            futures = {}
            for ci, chunk in enumerate(chunks):
                out = tmp_dir / f"sum_{round_idx:02d}_{ci:03d}.tif"
                fut = exe.submit(_calc_sum, chunk, out)
                futures[fut] = out

            for fut in as_completed(futures):
                fut.result()
                next_round.append(futures[fut])

        # Clean up older intermediates from previous round
        if round_idx > 1:
            for p in current:
                if p.parent == tmp_dir and p.exists():
                    try:
                        p.unlink()
                    except:
                        pass

        current = next_round

    shutil.move(str(current[0]), str(out_path))
    subprocess.run(["gdal_edit.py", "-unsetnodata", str(out_path)], check=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[MERGE] Final merged raster: {out_path}")


def sum_all_rasters(inputs: List[Path], out_path: Path):
    """
    Wrapper that merges all tag-level rasters into the final infrastructure raster.
    """

    if not inputs:
        print("[MERGE] No rasters to sum.")
        return
    print(f"[MERGE] Summing {len(inputs)} rasters into {out_path.name} (multi-threaded) ...")
    _calc_sum_safe(inputs, out_path)

def upload_to_gcs(local_path: Path, gcs_uri: str):
    """
    Uses the 'gsutil' CLI to copy rasters to a GCS bucket.
    """

    print(f"[UPLOAD] {local_path.name} → {gcs_uri}")
    gsutil_path = shutil.which("gsutil") or shutil.which("gsutil.cmd") or shutil.which("gsutil.CMD")
    if gsutil_path is None:
        print("[ERROR] gsutil not found. Make sure Google Cloud SDK is installed and in your PATH.")
        raise FileNotFoundError("gsutil executable not found.")
    try:
        subprocess.run([gsutil_path, "cp", str(local_path), gcs_uri], check=True)
        print("[UPLOAD] Uploaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Upload failed for {local_path.name}: {e}")
        raise

def ensure_ee_folder_exists(folder_path: str):
    """
    Ensure a GEE folder exists before uploading assets into it.
    """

    try:
        ee.data.getAsset(folder_path)
        print(f"[EE FOLDER] Exists: {folder_path}")
    except ee.ee_exception.EEException:
        parent = "/".join(folder_path.split("/")[:-1])
        if parent and not parent.startswith("projects/"):
            parent = f"projects/{GCP_PROJECT}/assets/{parent}"
        if parent and parent != folder_path:
            ensure_ee_folder_exists(parent)
        print(f"[EE FOLDER] Creating: {folder_path}")
        ee.data.createAsset({'type': 'Folder'}, folder_path)

def import_to_earth_engine(asset_id: str, gcs_uri: str):
    """
    Given a GCS URI and desired asset ID, tell Earth Engine
    to ingest the raster as an asset.
    Uses the GEE REST ingestion API (ee.data.startIngestion).
    """

    print(f"[EE IMPORT] {gcs_uri} → {asset_id}")
    try:
        folder_path = "/".join(asset_id.split("/")[:-1])
        ensure_ee_folder_exists(folder_path)
        try:
            ee.data.getAsset(asset_id)
            print(f"[EE IMPORT] Asset already exists: {asset_id} — skipping upload.")
            return
        except ee.ee_exception.EEException:
            pass

        # Extract resolution if encoded in filename (e.g., *_300m.tif)
        res_match = re.search(r"_(\d{2,4})m", asset_id)
        res_m = int(res_match.group(1)) if res_match else None

        params = {
            'id': asset_id,
            'tilesets': [{'sources': [{'uris': [gcs_uri]}]}],
            'bands': [{'id': 'b1'}],
            'pyramidingPolicy': 'MEAN',
            'properties': {
                'year': int(re.search(r"/(\d{4})/", asset_id).group(1)) if re.search(r"/(\d{4})/", asset_id) else None,
                'resolution_m': res_m
            }
        }
        task = ee.data.startIngestion(str(uuid.uuid4()), params)
        print(f"[EE IMPORT] Ingestion task started: {task['id']}")
    except Exception as e:
        print(f"[EE IMPORT ERROR] Failed to import {asset_id}: {e}")
        raise

def export_rasters_to_gee(raster_dir: Path, year: int, res: float, upload_merged_only: bool = False):
    """
    Upload rasters to GCS and GEE.
    If upload_merged_only=True, uploads only the merged final raster (infrastucture layer).
    Otherwise, upload. all per-tag rasters plus the merged one.
    Each uploaded raster is registered as a new GEE asset.
    The OSM download metadata file is also uploaded to GCS.
    """

    print(f"[EXPORT] Uploading rasters for {year} to GCS and importing to Earth Engine...")

    res_m = degrees_to_meters(res)

    if upload_merged_only:
        merged_candidate = raster_dir.parent / f"flii_infra_{year}_{res_m}m.tif"
        if not merged_candidate.exists():
            print("[EXPORT] Merged raster not found — cannot upload merged-only.")
            return
        tifs = [merged_candidate]
        print(f"[EXPORT] Uploading only merged raster: {merged_candidate.name}")
    else:
        tifs = list(raster_dir.glob("*.tif"))
        merged_candidate = raster_dir.parent / f"flii_infra_{year}_{res_m}m.tif"
        if merged_candidate.exists():
            tifs.append(merged_candidate)
            print(f"[EXPORT] Including merged raster: {merged_candidate.name}")

    if not tifs:
        print("[EXPORT] No rasters found to export.")
        return

    exported_assets = []
    for tif in tifs:
        gcs_uri = f"gs://{GCS_BUCKET}/{year}/{tif.name}"
        safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", tif.stem)
        asset_id = f"{EE_ASSET_ROOT}/{year}/{safe_stem}"

        try:
            upload_to_gcs(tif, gcs_uri)
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

    # OSM metadata file.
    osm_meta_path = raster_dir.parent / "osm_download_metadata.json"
    if osm_meta_path.exists():
        osm_gcs_uri = f"gs://{GCS_BUCKET}/{year}/osm_download_metadata.json"
        try:
            subprocess.run(["gsutil", "cp", str(osm_meta_path), osm_gcs_uri], check=True)
            print(f"[METADATA] Uploaded OSM metadata to GCS: {osm_gcs_uri}")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Could not upload OSM metadata to GCS: {e}")
    else:
        print("[WARN] No OSM metadata file found to upload.")

def cleanup_intermediates(year_dir: Path):
    """
    After a successful run, this removes large intermediate files (CSV shards,
    temporary merge rasters, etc.) to save disk space.
    """

    print(f"[CLEANUP] Removing intermediates in {year_dir}")
    for sub in ["csvs", "rasters/_tmp_merge"]:
        p = year_dir / sub
        shutil.rmtree(p, ignore_errors=True)

def main():
    """
    Runs all stages from OSM download to Earth Engine export.
    """

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(prog="python src/task.py",
        description=(
            "Run the FLII OSM-based infrastructure pipeline.\n\n"
            "This pipeline downloads OpenStreetMap global data for the specified year, "
            "filters features based on configured tags and weights (in osm_config.json), "
            "converts them to CSV and rasters, merges them into an infrastructure layer, "
            "and uploads both individual layers and the final raster to Google Earth Engine."
        ),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--year", type=int, default=date.today().year,
        help="Target year for OSM data (e.g., 2024). \n" \
        "Default = current (today's) year.\n" \
        "Uses previous year's planet snapshot (looks for a file from/closest to 12/31/year-1).")
    
    parser.add_argument("--resolution", type=float, default=0.00269,
        help="Output raster resolution in degrees (default = 0.00269 ≈ 300m at the equator).")
    
    parser.add_argument("--bounds", nargs=4, type=float, metavar=("xmin", "ymin", "xmax", "ymax"),
                        default=[-180, -90, 180, 90],
        help=(
            "Geographic bounds for rasterization in EPSG:4326.\n"
            "Format: xmin ymin xmax ymax.\n"
            "Default: global (-180 -90 180 90).\n"
            "Example: --bounds -10 -56 -35 5"
        ))
    parser.add_argument("--max_csv_rows", type=int, default=1_000_000, 
        help="Max rows per tag CSV shard (streaming split). Default: 1_000_000")
    
    parser.add_argument("--cleanup", action="store_true", 
        help="Remove intermediate CSVs and raster files after successful processing.")
    
    parser.add_argument("--upload_merged_only", action="store_true", 
        help="If set, upload only the merged raster to GCS and GEE instead of all tag rasters.")

    args = parser.parse_args()

    year = args.year
    res = args.resolution
    bounds = tuple(args.bounds)
    max_rows = args.max_csv_rows
    do_cleanup = args.cleanup
    upload_merged_only = args.upload_merged_only
    res_m = degrees_to_meters(res)
    
    # Define paths and names.
    year_dir = OUTPUT_BASE_DIR / f"{year}"
    txt_dir = year_dir
    csv_dir = year_dir / "csvs"
    raster_dir = year_dir / "rasters"
    merged_path = year_dir / f"flii_infra_{year}_{res_m}m.tif"
    pbf_path = year_dir / f"osm_{year}.osm.pbf"

    for d in [csv_dir, raster_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    categories = config.get("osm_categories", {}).get("weights", {})

    # Run all steps with timing logs.
    with log_time("[1] Download OSM PBF"):
        if not pbf_path.exists() and not any(year_dir.glob("osm_*.osm.bz2")):
            download_pbf(year, pbf_path)
        else:
            print("[SKIP] PBF/BZ2 already present.")

    with log_time("[2] Convert PBF to text"):
        txt_path = next(txt_dir.glob("*.txt"), None)
        if txt_path is None:
            osm_source = next(iter(year_dir.glob("osm_*.osm.*")))
            osmium_to_text(osm_source, txt_dir, categories)
        else:
            print("[SKIP] Text file already exists.")

    with log_time("[3] Split OSM text to CSV (streaming shards)"):
        txt_files = list(txt_dir.glob("*.txt"))
        if not txt_files:
            print("[WARN] No OSM text file found.")
        else:
            split_text_to_csv_streaming(txt_files[0], csv_dir, categories, max_rows=max_rows)

    with log_time("[4] Rasterize CSVs (parallel)"):
        csv_files = list(csv_dir.glob("*.csv"))
        if not csv_files:
            print("[WARN] No CSV files to rasterize.")
        else:
            rasters = rasterize_all(csv_files, raster_dir, bounds, res)

    with log_time("[5] Merge all rasters"):
        tag_rasters = merge_tag_shards(raster_dir)
        sum_all_rasters(tag_rasters, merged_path)

    with log_time("[6] Initialize Earth Engine + export rasters"):
        init_google_earth_engine()
        export_rasters_to_gee(raster_dir, year, res, upload_merged_only)

    # Optionally clean up intermediates.
    if do_cleanup:
        with log_time("[7] Cleanup intermediates"):
            cleanup_intermediates(year_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Pipeline failed: {e}")
        sys.exit(1)
