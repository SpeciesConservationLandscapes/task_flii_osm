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
import multiprocessing

# ------------
# GDAL CONFIG 
# ------------

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_CACHEMAX", "4096")  
gdal.SetConfigOption("GDAL_SWATH_SIZE", "1048576")  
gdal.SetConfigOption("GDAL_MAX_DATASET_POOL_SIZE", "400")
os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"
os.environ["VSI_CACHE"] = "FALSE" 

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
    txt_file = Path(txt_file)
    csv_dir = Path(csv_dir)

    # Build config lookup for fast membership check
    config_index = defaultdict(set)
    for kv in config.keys():
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        config_index[k].add(v)

    tag_files: Dict[str, csv.writer] = {}
    tag_fhandles: Dict[str, any] = {}
    tag_shard_counts: Dict[str, int] = defaultdict(int)
    tag_row_counts: Dict[str, int] = defaultdict(int)
    out_paths: List[Path] = []

    geom_prefixes = (
        "POINT(", "MULTIPOINT(", "LINESTRING(", "MULTILINESTRING(",
        "POLYGON(", "MULTIPOLYGON(", "GEOMETRYCOLLECTION("
    )

    # Super-tolerant regex for key=value pairs
    tag_pattern = re.compile(
        r'@?"?([\w:\-]+)"?\s*=\s*"?([\w\.\-:/%]+)"?',
        flags=re.UNICODE
    )

    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith(geom_prefixes):
                continue

            try:
                wkt_end = line.index(")") + 1
            except ValueError:
                continue

            wkt = line[:wkt_end]
            tags_str = line[wkt_end:].strip()

            # Extract all key=value pairs (robust regex)
            tags = {}
            for k, v in tag_pattern.findall(tags_str):
                tags[k.strip("@")] = v

            # If nothing found, skip
            if not tags:
                continue

            # Process valid matches
            for tag, val in tags.items():
                if tag not in config_index or val not in config_index[tag]:
                    continue
                tag_val = f"{tag}={val}"
                burn = config[tag_val]

                # Rotate shard if full
                if (tag_val not in tag_files) or (tag_row_counts[tag_val] >= max_rows):
                    if tag_val in tag_fhandles:
                        tag_fhandles[tag_val].close()
                    safe = re.sub(r"[=:/@]", "_", tag_val)
                    shard_idx = tag_shard_counts[tag_val] + 1
                    out_path = csv_dir / f"{safe}_{shard_idx:03d}.csv"
                    fh = open(out_path, "w", newline="", encoding="utf-8")
                    writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(["WKT", "BURN"])
                    tag_files[tag_val] = writer
                    tag_fhandles[tag_val] = fh
                    tag_shard_counts[tag_val] = shard_idx
                    tag_row_counts[tag_val] = 0
                    out_paths.append(out_path)

                tag_files[tag_val].writerow([wkt, burn])
                tag_row_counts[tag_val] += 1

    for fh in tag_fhandles.values():
        try:
            fh.close()
        except:
            pass

    print(f"[SPLIT] {txt_file.name} → {len(out_paths)} CSV shards written to {csv_dir}")
    return out_paths

def generate_global_tiles(bounds=(-180, -90, 180, 90), step=60):
    """
    Generate a list of (xmin, ymin, xmax, ymax) tiles covering the global extent.
    The 'step' defines the tile size in degrees.
    Each tile will be aligned on whole degree boundaries for consistency.
    """
    xmin, ymin, xmax, ymax = bounds
    tiles = []
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            tiles.append((
                x,
                y,
                min(x + step, xmax),
                min(y + step, ymax)
            ))
            y += step
        x += step
    return tiles

def _rasterize_tag(tag, csv_group, tile_dir, tile_bounds, res):
    safe_tag = re.sub(r"[=:/@]", "_", tag)
    out_tif = tile_dir / f"{safe_tag}.tif"
    if out_tif.exists():
        return out_tif
    subprocess.run([
        "gdal_rasterize",
        "-a", "BURN",
        "-a_srs", "EPSG:4326",
        "-ot", "Byte",
        "-te", *map(str, tile_bounds),
        "-tr", str(res), str(res),
        "-a_nodata", "0",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=YES",
        "-co", "COMPRESS=NONE",
        f"CSV:{csv_group[0]}",
        str(out_tif)
    ], check=True)
    return out_tif

def _rasterize_tag_group(
    tag: str,
    shard_csvs: List[Path],
    out_dir: Path,
    bounds: Tuple[float, float, float, float],
    res: float
) -> Optional[Path]:
    """
    Rasterize all CSV shards for a tag into one .tif for the given tile.

    Each CSV contains 'WKT' and 'BURN' fields. The BURN value is rasterized as pixel value.
    This function merges multiple CSV shards for the same tag into one raster per tile.
    """
    try:
        safe_tag = re.sub(r"[=:/@]", "_", tag)
        out_tif = out_dir / f"{safe_tag}.tif"

        # Temporary intermediate rasters (one per shard)
        shard_tifs = []

        for csv_file in shard_csvs:
            shard_tif = out_dir / f"{csv_file.stem}.tif"

            subprocess.run([
                "gdal_rasterize",
                "-a", "BURN",
                "-a_srs", "EPSG:4326",
                "-ot", "Byte",
                "-te", *map(str, bounds),
                "-tr", str(res), str(res),
                "-a_nodata", "0",
                "-co", "TILED=YES",
                "-co", "COMPRESS=NONE",
                "-co", "BIGTIFF=YES",
                f"CSV:{csv_file}",
                str(shard_tif)
            ], check=True)

            shard_tifs.append(shard_tif)

        # Merge shards for this tag into one final .tif
        if len(shard_tifs) > 1:
            args = [
                "gdal_merge.py", "-o", str(out_tif),
                "-of", "GTiff",
                "-co", "TILED=YES",
                "-co", "BIGTIFF=YES",
                "-n", "0", "-a_nodata", "0",
            ] + [str(p) for p in shard_tifs]
            subprocess.run(args, check=True)
            for t in shard_tifs:
                t.unlink(missing_ok=True)
        else:
            shutil.move(str(shard_tifs[0]), str(out_tif))

        return out_tif

    except Exception as e:
        print(f"[RASTERIZE] Failed for {tag}: {e}")
        return None


def rasterize_all_fast(
    csvs: List[Path],
    out_dir: Path,
    bounds: Tuple[float, float, float, float],
    res: float,
    config: dict
) -> Optional[Path]:
    """
    Rasterize all tag CSVs into a single multi-band GeoTIFF for this tile.
    Each tag becomes one band. Also returns the list of tag names for reference.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    num_cpus = max(1, multiprocessing.cpu_count() - 1)
    print(f"[RASTERIZE] {len(csvs)} CSV shards detected — using {num_cpus} workers.")

    # --- Group CSV shards by tag name (strip shard suffix) ---
    tag_groups = defaultdict(list)
    for csv_path in csvs:
        tag_base = re.sub(r"(_[0-9]{3})+$", "", csv_path.stem)
        tag_groups[tag_base].append(csv_path)

    tag_list = sorted(tag_groups.keys())
    print(f"[RASTERIZE] {len(tag_list)} unique tags found.")

    # --- Rasterize each tag into temporary single-band TIFF ---
    tmp_rasters = []
    with ProcessPoolExecutor(max_workers=num_cpus) as exe:
        futures = {
            exe.submit(_rasterize_tag_group, tag, group, out_dir, bounds, res): tag
            for tag, group in tag_groups.items()
        }

        for i, f in enumerate(as_completed(futures), 1):
            tag = futures[f]
            result = f.result()
            if result:
                tmp_rasters.append(result)
            if i % 10 == 0 or i == len(futures):
                print(f"[RASTERIZE] Completed {i}/{len(futures)} tags")

    if not tmp_rasters:
        print("[RASTERIZE] No rasters produced for this tile.")
        return None

    # --- Combine all tag rasters into one multi-band TIFF ---
    tile_multiband = out_dir / "tags_tile.tif"
    print(f"[MERGE] Creating multi-band raster for {len(tmp_rasters)} tags → {tile_multiband.name}")

    merge_cmd = [
        "gdal_merge.py",
        "-separate",
        "-o", str(tile_multiband),
        "-of", "GTiff",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=YES",
        "-co", "COMPRESS=LZW",
        "-a_nodata", "0",
    ] + [str(r) for r in tmp_rasters]

    subprocess.run(merge_cmd, check=True)

    # Optionally clean up temporary single-band rasters
    for r in tmp_rasters:
        try:
            r.unlink()
        except:
            pass

    print(f"[RASTERIZE] Multi-band raster created: {tile_multiband}")
    return tile_multiband

def _calc_sum(inputs: List[Path], out_path: Path):
    """
    Sum ≤26 (limit) rasters safely (A+B+...).
    Unsets NoData on all inputs and output to avoid propagation.
    """
    
    if len(inputs) == 0:
        raise ValueError("No rasters provided to _calc_sum")
    if len(inputs) > 26:
        raise ValueError("Too many inputs for a single gdal_calc call (max=26)")

    # Unset NoData before summing
    for r in inputs:
        subprocess.run(["gdal_edit.py", "-unsetnodata", str(r)], check=True)

    calc_expr = " + ".join([f"nan_to_num({chr(65+i)})" for i in range(len(inputs))])

    if out_path.exists():
        out_path.unlink()

    args = [
        "gdal_calc.py", "--quiet", "--overwrite",
        f"--outfile={str(out_path)}",
        "--type=Byte",
        "--calc", calc_expr,
        "--co=COMPRESS=NONE",
        "--co=BIGTIFF=YES",
    ]
    for i, r in enumerate(inputs):
        args.extend([f"-{chr(65+i)}", str(r)])

    subprocess.run(args, check=True)

    # Remove any NoData flag on output to ensure valid numeric sum
    subprocess.run(["gdal_edit.py", "-unsetnodata", str(out_path)], check=True)


def _calc_sum_safe(inputs: List[Path], out_path: Path, chunk_size: int = 20):
    """
    Handles cases with more than 26 rasters by merging them in parallel chunks.
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


def sum_all_rasters(multiband_path: Path, out_path: Path):
    """
    Sum all bands of a multi-band raster into a single-band TIFF.
    Produces the final infra_tile_xxx.tif file.
    """
    if not multiband_path.exists():
        print(f"[MERGE] Multi-band raster not found: {multiband_path}")
        return

    print(f"[MERGE] Summing all bands in {multiband_path.name} → {out_path.name}")

    # Determine actual band count dynamically
    ds = gdal.Open(str(multiband_path))
    if ds is None:
        print(f"[ERROR] Could not open raster: {multiband_path}")
        return
    band_count = ds.RasterCount
    ds = None

    calc_expr = " + ".join([f"A{i+1}" for i in range(band_count)])

    args = [
        "gdal_calc.py",
        "--overwrite",
        f"--outfile={out_path}",
        "--type=Byte",
        "--calc", calc_expr,
        f"-A={multiband_path}",
        "--allBands=A",
        "--co=COMPRESS=LZW",
        "--co=TILED=YES",
        "--co=BIGTIFF=YES",
    ]
    subprocess.run(args, check=True)
    subprocess.run(["gdal_edit.py", "-unsetnodata", str(out_path)], check=True)
    print(f"[MERGE] Infra raster created: {out_path}")


def merge_tag_across_tiles_vrt(tag, tiles_dir, output_dir):
    """Merge all tile rasters for a tag into a global GeoTIFF using VRT + gdal_translate."""
    input_tifs = sorted(Path(tiles_dir).rglob(f"*/{tag}.tif"))
    if not input_tifs:
        print(f"[SKIP] No rasters found for tag {tag}")
        return tag, None

    vrt_path = Path(output_dir) / f"{tag}.vrt"
    out_tif = Path(output_dir) / f"{tag}_global.tif"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already created (resume-safe)
    if out_tif.exists():
        print(f"[SKIP] Already merged {tag}")
        return tag, out_tif

    buildvrt_cmd = ["gdalbuildvrt", str(vrt_path)] + [str(f) for f in input_tifs]
    translate_cmd = [
        "gdal_translate", str(vrt_path), str(out_tif),
        "-co", "COMPRESS=LZW",
        "-co", "BIGTIFF=YES",
        "-co", "TILED=YES",
        "-a_nodata", "0"
    ]

    try:
        subprocess.run(buildvrt_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(translate_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[MERGE OK] {tag} → {out_tif}")
        return tag, out_tif
    except subprocess.CalledProcessError:
        print(f"[MERGE FAIL] {tag}")
        return tag, None


def merge_all_tags_parallel(tags, tiles_dir, output_dir, max_workers=None):
    """Run VRT-based merges for all tags in parallel."""
    if max_workers is None:
        max_workers = min(12, multiprocessing.cpu_count())

    tiles_dir = Path(tiles_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MERGE] Starting parallel global merges for {len(tags)} tags using {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(merge_tag_across_tiles_vrt, tag, tiles_dir, output_dir): tag for tag in tags}

        for i, f in enumerate(as_completed(futures), 1):
            tag = futures[f]
            try:
                _, result = f.result()
                if result:
                    print(f"[{i}/{len(tags)}] Done: {tag}")
            except Exception as e:
                print(f"[ERROR] {tag}: {e}")

    print(f"[MERGE] All tag merges complete. Outputs in {output_dir}")


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

        # Build properties safely
        props = {}
        year_match = re.search(r"/(\d{4})/", asset_id)
        if year_match:
            props["year"] = int(year_match.group(1))
        if res_m is not None:
            props["resolution_m"] = int(res_m)

        params = {
            "id": asset_id,
            "tilesets": [{"sources": [{"uris": [gcs_uri]}]}],
            "bands": [{"id": "b1"}],
            "pyramidingPolicy": "MEAN",
            "properties": props
        }

        task = ee.data.startIngestion(str(uuid.uuid4()), params)
        print(f"[EE IMPORT] Ingestion task started: {task['id']}")
    except Exception as e:
        print(f"[EE IMPORT ERROR] Failed to import {asset_id}: {e}")
        raise


def export_rasters_to_gee(raster_dir: Path, year: int, res: float, upload_merged_only: bool = False):
    """
    Upload rasters to GCS and GEE.
    If upload_merged_only=True, uploads only the merged final raster (infrastructure layer).
    Otherwise, uploads all per-tag rasters plus the merged one.
    Each uploaded raster is registered as a new GEE asset.
    The OSM download metadata file is also uploaded to GCS.
    """

    print(f"[EXPORT] Uploading rasters for {year} to GCS and importing to Earth Engine...")

    res_m = degrees_to_meters(res)

    # --- Key expected rasters ---
    merged_infra = raster_dir.parent / f"flii_infra_{year}_{res_m}m.tif"
    multi_band = raster_dir / f"flii_tags_global_{year}_{res_m}m.tif"

    if not merged_infra.exists():
        print("[WARN] Global infrastructure raster not found.")
    else:
        print(f"[EXPORT] Found global infra raster: {merged_infra.name}")

    if multi_band.exists():
        print(f"[EXPORT] Found global multi-band tag raster: {multi_band.name}")

    # --- Gather per-tag rasters (only used if not upload_merged_only) ---
    tag_rasters = []
    if not upload_merged_only:
        for tif in raster_dir.glob("tile_*/[!infra]*.tif"):
            tag_rasters.append(tif)

    # --- Build list of rasters to upload ---
    to_upload = []

    if multi_band.exists():
        to_upload.append(multi_band)
    if merged_infra.exists():
        to_upload.append(merged_infra)

    if not upload_merged_only and tag_rasters:
        to_upload.extend(tag_rasters)

    if not to_upload:
        print("[EXPORT] No rasters found to export.")
        return

    # --- Upload & register each raster ---
    exported_assets = []
    for tif in to_upload:
        # Destination paths
        gcs_uri = f"gs://{GCS_BUCKET}/{year}/{tif.name}"
        safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", tif.stem)
        asset_id = f"{EE_ASSET_ROOT}/{year}/{safe_stem}"

        try:
            print(f"[UPLOAD] Uploading {tif.name} to {gcs_uri}")
            upload_to_gcs(tif, gcs_uri)

            # --- Handle multi-band specially (set band count) ---
            bands = []
            try:
                ds = gdal.Open(str(tif))
                band_count = ds.RasterCount if ds else 1
                bands = [{"id": f"b{i+1}"} for i in range(band_count)]
                ds = None
            except Exception:
                bands = [{"id": "b1"}]

            print(f"[EE IMPORT] Importing {tif.name} ({len(bands)} bands) → {asset_id}")

            ensure_ee_folder_exists("/".join(asset_id.split("/")[:-1]))

            try:
                ee.data.getAsset(asset_id)
                print(f"[EE IMPORT] Asset already exists: {asset_id} — skipping.")
                continue
            except ee.ee_exception.EEException:
                pass

            props = {"year": year, "resolution_m": res_m}
            params = {
                "id": asset_id,
                "tilesets": [{"sources": [{"uris": [gcs_uri]}]}],
                "bands": bands,
                "pyramidingPolicy": "MEAN",
                "properties": props,
            }

            task = ee.data.startIngestion(str(uuid.uuid4()), params)
            print(f"[EE IMPORT] Ingestion started: {task['id']}")

            exported_assets.append({
                "filename": tif.name,
                "asset_id": asset_id,
                "gcs_uri": gcs_uri,
                "bands": len(bands),
                "status": "imported",
            })

        except Exception as e:
            print(f"[ERROR] Failed to upload/import {tif.name}: {e}")
            exported_assets.append({
                "filename": tif.name,
                "asset_id": asset_id,
                "gcs_uri": gcs_uri,
                "status": f"failed - {e}",
            })

    # --- Save metadata locally & to GCS ---
    metadata_path = raster_dir / f"gee_export_metadata_{year}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(exported_assets, f, indent=2)

    gcs_metadata_uri = f"gs://{GCS_BUCKET}/{year}/gee_export_metadata_{year}.json"
    try:
        subprocess.run(["gsutil", "cp", str(metadata_path), gcs_metadata_uri], check=True)
        print(f"[METADATA] Export summary uploaded: {gcs_metadata_uri}")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Could not upload metadata to GCS: {e}")

    # --- Upload OSM metadata if available ---
    osm_meta_path = raster_dir.parent / "osm_download_metadata.json"
    if osm_meta_path.exists():
        osm_gcs_uri = f"gs://{GCS_BUCKET}/{year}/osm_download_metadata.json"
        try:
            subprocess.run(["gsutil", "cp", str(osm_meta_path), osm_gcs_uri], check=True)
            print(f"[METADATA] Uploaded OSM metadata to GCS: {osm_gcs_uri}")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Could not upload OSM metadata to GCS: {e}")

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
            existing_csvs = list(csv_dir.glob("*.csv"))
            if existing_csvs:
                print(f"[SKIP] Found {len(existing_csvs)} CSV files — skipping text-to-CSV conversion.")
            else:
                print("[SPLIT] No CSVs found — generating from text...")
                split_text_to_csv_streaming(txt_files[0], csv_dir, categories, max_rows=max_rows)


    with log_time("[4] Rasterize CSVs by tile (single-band rasters per tag)"):

        csv_files = list(csv_dir.glob("*.csv"))
        if not csv_files:
            print("[WARN] No CSV files to rasterize.")
        else:
            tile_size = 60  # ~18 global tiles
            tiles = generate_global_tiles(bounds, step=tile_size)
            print(f"[RASTERIZE] Splitting world into {len(tiles)} tiles of {tile_size}°×{tile_size}°")

            tile_outputs = []
            for i, tile_bounds in enumerate(tiles, 1):
                tile_dir = raster_dir / f"tile_{i:03d}"
                tile_dir.mkdir(parents=True, exist_ok=True)
                tile_out = tile_dir / f"infra_tile_{i:03d}.tif"

                # --- Skip if tile already exists ---
                if tile_out.exists():
                    print(f"[SKIP] Tile {i}/{len(tiles)} already rasterized ({tile_out.name}) — skipping.")
                    tile_outputs.append(tile_out)
                    continue

                print(f"[TILE {i}/{len(tiles)}] Processing bounds {tile_bounds}")

                try:
                    # Group CSV shards by tag (strip shard suffix)
                    tag_groups = defaultdict(list)
                    for csv_path in csv_files:
                        tag_base = re.sub(r"(_[0-9]{3})+$", "", csv_path.stem)
                        tag_groups[tag_base].append(csv_path)

                    # Parallel rasterization (each tag → single-band raster)
                    num_cpus = max(1, multiprocessing.cpu_count() - 1)
                    print(f"[RASTERIZE] {len(tag_groups)} unique tags found — using {num_cpus} workers.")

                    with ThreadPoolExecutor(max_workers=num_cpus) as exe:
                        futures = [
                            exe.submit(_rasterize_tag, t, g, tile_dir, tile_bounds, res)
                            for t, g in tag_groups.items()
                        ]
                        for j, f in enumerate(as_completed(futures), 1):
                            try:
                                f.result()
                            except Exception as e:
                                print(f"[TILE {i}] Tag rasterization failed: {e}")
                            if j % 10 == 0 or j == len(futures):
                                print(f"[TILE {i}] {j}/{len(futures)} tags rasterized.")

                    # --- Merge all tag rasters for this tile into a summed infrastructure layer ---
                    tag_tifs = list(tile_dir.glob("*.tif"))
                    if not tag_tifs:
                        print(f"[TILE {i}] No rasters produced — skipping.")
                        continue

                    _calc_sum_safe(tag_tifs, tile_out)
                    tile_outputs.append(tile_out)
                    print(f"[TILE {i}] Completed and saved {tile_out.name}")

                except Exception as e:
                    print(f"[TILE {i}] Failed: {e}")
                    continue

            print(f"[RASTERIZE] Completed {len(tile_outputs)}/{len(tiles)} tiles.")

    # (a) Merge per-tile infrastructure rasters into global infra layer
    with log_time("[5a] Merge tile infrastructure rasters into global layer"):
        tile_outputs = sorted(raster_dir.glob("tile_*/infra_tile_*.tif"))
        if not tile_outputs:
            print("[WARN] No tile rasters found for merging.")
        else:
            print(f"[MERGE] Combining {len(tile_outputs)} tile rasters → {merged_path.name}")
            subprocess.run([
                "gdal_merge.py", "-o", str(merged_path),
                "-of", "GTiff",
                "-co", "TILED=YES",
                "-co", "BLOCKXSIZE=1024",
                "-co", "BLOCKYSIZE=1024",
                "-co", "COMPRESS=LZW",
                "-co", "BIGTIFF=YES"
            ] + [str(t) for t in tile_outputs], check=True)
            print(f"[MERGE] Global raster created: {merged_path}")

    # (b) Stack all tag rasters (after merging per-tag across tiles)
    with log_time("[5b] Stack all tag rasters into global multi-band raster"):
        tag_rasters = sorted(raster_dir.glob("*_global.tif"))
        if not tag_rasters:
            print("[MERGE] No tag rasters found to stack.")
        else:
            global_multiband = raster_dir / f"flii_tags_global_{year}_{res_m}m.tif"
            if global_multiband.exists():
                print(f"[SKIP] Global multi-band raster already exists: {global_multiband.name}")
            else:
                vrt_path = raster_dir / "flii_tags_global_tmp.vrt"
                print(f"[STACK] Stacking {len(tag_rasters)} rasters → {global_multiband.name}")
                subprocess.run(
                    ["gdalbuildvrt", "-separate", str(vrt_path)] + [str(r) for r in tag_rasters],
                    check=True
                )
                subprocess.run([
                    "gdal_translate", str(vrt_path), str(global_multiband),
                    "-co", "TILED=YES",
                    "-co", "BIGTIFF=YES",
                    "-co", "COMPRESS=LZW",
                    "-a_nodata", "0"
                ], check=True)
                vrt_path.unlink(missing_ok=True)
                print(f"[STACK] Created global multi-band raster: {global_multiband}")


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
