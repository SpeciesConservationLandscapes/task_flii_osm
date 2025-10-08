# FLII OSM Pipeline
---

## Overview

The **FLII OSM** task automates the processing of OpenStreetMap (OSM) data to generate spatial infrastructure indicators used in the *Forest Landscape Integrity Index (FLII)*.

This script fetches, processes, rasterizes, and exports OSM features into multiband GeoTIFFs, and publishes them as Google Earth Engine (EE) assets for use in FLII analyses.

---

## What does this task do?

1. **Fetch OSM PBF file**  
   Downloads the latest or year-specific `.osm.pbf` snapshot from official OSM mirrors.

2. **Convert PBF file → text**  
   Uses `osmium` to export the OSM file into a human-readable `.txt` format.

3. **Split text into filtered CSVs**  
   - Extracts points matching the OSM tags and subcategories defined in [`osm_config.json`](osm_config.json).  
   - Writes one CSV per tag-value combination (e.g., `highway=footway.csv`).

4. **Rasterize each CSV file**  
   Converts each CSV into a GeoTIFF using `gdal_rasterize` within specified geographic bounds and resolution (e.g., `highway_footway.tiff`).

5. **Apply weights and merge rasters**  
   Merges all individual rasters into one weighted multiband raster representing the final FLII infrastructure layer.

6. **Export results to Google Cloud Storage (GCS)**  
   Uploads each GeoTIFF (and metadata) to the configured GCS bucket.

7. **Import to Google Earth Engine (EE)**  
   Publishes each raster as an EE asset using a service account for automated cloud integration.

8. **Clean up**  
   Optionally removes intermediate files from both local storage and GCS after successful import.

---

## Environment Variables

All credentials and project configuration are loaded from a `.env` file.  
You can copy and edit the provided template:

## Output Structure

flii_outputs/
│
├── 2021/
├── 2022/
├── 2023/
└── 2024/
    ├── csvs/
    ├── rasters/
    ├── flii_osm_2024_merged.tif
    ├── osm_2024.osm.pbf
    ├── osm_2024.txt
    └── gee_export_metadata_2024.json



