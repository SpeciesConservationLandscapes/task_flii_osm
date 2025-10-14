# FLII OSM Pipeline
---

## Overview

The **FLII OSM** task automates the processing of OpenStreetMap (OSM) data to generate infrastructure layers used in the *Forest Landscape Integrity Index (FLII)*.

This script fetches, processes, rasterizes, and exports OSM features into GeoTIFFs, and publishes them as Google Earth Engine (EE) assets for use in FLII generation.

It is recommended to run the FLII OSM pipeline inside a Docker container to ensure a consistent and reproducible environment.

---

## What does this task do?

1. **Fetch OSM PBF file**  
   Downloads the latest or year-specific `.osm.pbf` snapshot from official OSM mirrors.

2. **Convert PBF file → text**  
   Uses `osmium` to convert the OSM file into a `.txt` format.

3. **Split text into filtered CSVs with tag-specific weights**  
   - Extracts features (e.g. lines, points) matching the OSM tags and subcategories defined in [`osm_config.json`](osm_config.json).  
   - Writes one CSV per tag-value combination (e.g., `highway=footway.csv`) with the geographic WKT information, and subcategory-specific weights defined in `osm_config.json`.

4. **Rasterize each CSV file**  
   Converts each CSV into a GeoTIFF using `gdal_rasterize` (e.g., `highway_footway.tif`).

5. **Merge rasters into a final infrastructure layer**  
   Merges rasters into one weighted raster representing the final infrastructure layer.

6. **Export results to Google Cloud Storage (GCS)**  
   Uploads each GeoTIFF (and metadata) to the configured GCS bucket.

7. **Import to Google Earth Engine (EE)**  
   Publishes each raster as an EE asset.

---

## Environment Variables

All credentials and project configuration are loaded from a `.env` file.  
Refer to [`.env.example`](/.env.example) to setup your env locally.

## Structure

```text
FLII_OSM/
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
│     ├── 2024/
│     │     ├── csvs/
│     │     │     └── <category=subcategory>.csv
│     │     └── rasters/
│     │           ├── gee_export_metadata_2024.json
│     │           └── <category_subcategory>.tiff
│     ├── flii_infra_2024.tif
│     ├── osm_2024.osm.pbf
│     └── osm_2024.txt
├── .env
├── Dockerfile
├── Makefile
└── README.md
```

## Example usage
To run the task:

```
python src/task.py --upload_merged_only
```

By default, the code runs for the year to date, outputs 0.00269 degrees resolution rasters (300 meters near the Equator), and for global bounds (e.g. -90, -180, 90, 180). These are customizable settings.

If running for a specific year (not the current year), set the  `--year` flag:
```
python src/task.py --year 2024 --upload_merged_only
```

If you want to upload all tag-level rasters (e.g. highway_primary.tif, power_plant.tif, etc) to Cloud Storage and as an GEE asset, remove the    `--upload_merged_only` flag (more cloud/GEE storage space needed).
```
python src/task.py --year 2024
```

If you want to change the raster resolution, you can change the `--resolution` flag. E.g. 100 m (near the Equator):
```
python src/task.py --year 2024 –resolution 0.0009 --upload_merged_only
```

If you want to clean up intermediate files from the container/VM right away (e.g. CSV files, temporary merge rasters, etc), you can use the `--cleanup` flag.
```
python src/task.py --year 2024 --cleanup
```

## License
Copyright (C) 2025 Wildlife Conservation Society.

The files in this repository are part of the task framework for calculating 
the Forest Landscape Integrity Index [(https://www.forestintegrity.com/)](https://www.forestintegrity.com/) and are released under the GPL license: [https://www.gnu.org/licenses/#GPL](https://www.gnu.org/licenses/#GPL). See [LICENSE](./LICENSE) for details.