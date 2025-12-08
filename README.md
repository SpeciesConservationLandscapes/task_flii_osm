# FLII OSM Pipeline
---

## Overview

The **FLII OSM** task automates the processing of OpenStreetMap (OSM) data to generate infrastructure layers used in the *Forest Landscape Integrity Index (FLII)*.

This script fetches, processes, and exports OSM features into GeoTIFFs, and publishes them as Google Earth Engine (EE) assets for use in FLII generation.

It is recommended to run the FLII OSM pipeline inside a Docker container to ensure a consistent and reproducible environment.

---

## What does this task do?

1. **Fetch OSM PBF file**  
   Downloads the `.osm.pbf` snapshot from (or closest to) Dec 31 of the chosen year from official OSM mirrors. Stores metadata (size, URL, date) in a JSON file (`osm_download_metadata.json`) alongside the data.

2. **Convert PBF file → text**  
   Uses `osmium` to convert the OSM file into a `.txt` format.

3. **Split text into filtered CSVs with tag-specific weights**  
   - Extracts features (e.g. lines, points) matching the OSM tags and subcategories defined in [`osm_config.json`](osm_config.json).  
   - Writes one or more CSVs per tag-value combination (e.g., `highway_footway_001.csv`, `highway_footway_002.csv`) with the geographic WKT information, and subcategory-specific weights defined in `osm_config.json`.

4. **Rasterize each CSV file**  
   Converts each CSV into a GeoTIFF using `gdal_rasterize` (e.g., `highway_footway_001.tif`, `highway_footway_002.tif`). This process is done by tiles.

5. **Mosaics rasters into global layers and sums them into the final infrastructure layer**  
   Mosaics tiled rasters into global tag-specific layers and then calculates one weighted raster representing the final infrastructure layer.

6. **Export results to Google Cloud Storage (GCS)**  
   Uploads each GeoTIFF (and metadata) to the configured GCS bucket.

7. **Import to Google Earth Engine (EE)**  
   Publishes each raster as an EE asset. The code creates a `<year>` folder to allocate the rasters. The infrastructure layer is uploaded as an image and the tag rasters as images inside an `ImageCollection` called `tags_<year>_<res_m>m`.

---

## Environment Variables

All credentials and project configuration are loaded from a `.env` file.  
Refer to [`.env.example`](/.env.example) to setup your env locally.

## Structure

```text
task_flii_osm/
│
├── src/
│     ├── task.py
│     └── osm_config.json
│
├── flii_outputs/
│     ├── <years>
│     ├── 2021/
│     ├── 2022/
│     ├── 2023/
│     └── 2024/
│           ├── csvs/
│           │     ├── <category_subcategory>_001.csv
|           |     └── <category_subcategory>_002.csv
│           ├── rasters/
│           │     ├── tile_001/
|           |     |     ├── <category_subcategory>_001.tif
|           |     |     └── <category_subcategory>_002.tif
|           |     ├── tile_002/
|           |     ├── <category_subcategory>_global_2024_300m.tif
│           │     └── gee_export_metadata_2024.json
│           ├── flii_infra_2024_300m.tif
│           ├── osm_2024.osm.pbf
│           ├── osm_2024.txt
│           └── osm_download_metadata.json
├── .env
├── Dockerfile
├── Makefile
└── README.md
```

## Example usage

Run `python src/task.py --help` for a list of variables:

```
usage: python src/task.py [-h] [--year YEAR] [--resolution RESOLUTION] [--bounds xmin ymin xmax ymax]
                          [--max_csv_rows MAX_CSV_ROWS] [--tile TILE] [--cleanup] [--upload_merged_only]

Run the FLII OSM-based infrastructure pipeline.

This pipeline downloads OpenStreetMap global data for the specified year, filters features based on configured tags and weights (in osm_config.json), converts them to CSV and rasters, merges them into an infrastructure layer, and uploads both individual layers and the final raster to Google Earth Engine.

options:
  -h, --help            show this help message and exit
  --year YEAR           Target year for OSM data (e.g., 2024).
                        Default = current (today's) year.
                        Uses year's planet snapshot (looks for a file from/closest to 12/31/year).        
  --resolution RESOLUTION
                        Output raster resolution in degrees (default = 0.0026949458523585646 ≈ 300m at the equator).
  --bounds xmin ymin xmax ymax
                        Geographic bounds for rasterization in EPSG:4326.
                        Format: xmin ymin xmax ymax.
                        Default: global (-180.000823370733258 -88.000761862916562 180.000823370733258 88.000761862916562).
                        Example: --bounds -10 -56 -35 5
  --max_csv_rows MAX_CSV_ROWS
                        Max rows per tag CSV shard (streaming split). Default: 1_000_000
  --tile TILE           Tile size in degrees for global tiling (default = 60). Smaller values create more tiles.     
  --cleanup             Remove intermediate CSVs and raster files after successful processing.
  --upload_merged_only  If set, upload only the merged raster to GCS and GEE instead of all tag rasters.
```

By default, the code runs for the year to date, outputs `0.0026949458523585646 degrees` resolution rasters in `EPSG:4326` (`300 meters` near the Equator), and for global bounds (e.g. `-180.000823370733258, -88.000761862916562, 180.000823370733258, 88.000761862916562`). By default, all rasters (~ 135 OSM tag-level/subcategory layers + final infrastructure layer) are uploaded to GCP and GEE. These are customizable settings.

To run the task:

```
python src/task.py
```

Set the  `--year` flag:
```
python src/task.py --year 2024
```

If you want to upload only the final infrastructure layer to Cloud Storage and as an GEE asset, add the    `--upload_merged_only` flag.
```
python src/task.py --year 2024 --upload_merged_only
```

If you want to change the raster resolution, change `--resolution` flag. E.g. 100 m (near the Equator). Always in EPSG:4326:
```
python src/task.py --year 2024 --resolution 0.0008983152841195215
```

*Note*: For resolutions lower than 300 m (0.0026949458523585646), you might need to change the `--tile` flag to avoid tiles with too many pixels (e.g, `--tile 30`).

If you want to clean up intermediate files from the container/VM right away (e.g. CSV files, temporary merge rasters, etc), you can use the `--cleanup` flag.
```
python src/task.py --year 2024 --cleanup
```

Summary of flags:
| Flag                           | Description                                                |
| ------------------------------ | ---------------------------------------------              |
| `--tile`                       | Tile size in degrees (default 60)                          |
| `--bounds xmin ymin xmax ymax` | Rasterization extent (default -180.000823370733258 -88.000761862916562 180.000823370733258 88.000761862916562)|
| `--resolution`                 | Pixel size in degrees (default 0.0026949458523585646)      |
| `--upload_merged_only`         | Upload only final infra raster                             |
| `--max_csv_rows`               | Limit for shard splitting (default 1,000,000)              |
| `--cleanup`                    | Delete CSVs and temp rasters after export                  |

## License
Copyright (C) 2025 Wildlife Conservation Society.

The files in this repository are part of the task framework for calculating 
the Forest Landscape Integrity Index [(https://www.forestintegrity.com/)](https://www.forestintegrity.com/) and are released under the GPL license: [https://www.gnu.org/licenses/#GPL](https://www.gnu.org/licenses/#GPL). See [LICENSE](./LICENSE) for details.
