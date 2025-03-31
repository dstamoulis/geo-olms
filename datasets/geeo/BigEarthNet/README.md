# BigEarthNet-S2v1.0 Data Ingestion & GeoPackage Creation

This repository contains a minimal example script for ingesting the BigEarthNet dataset and converting it into a GeoPackage file containing cleaned geospatial data. The code processes the BigEarthNet CSV labels file (`BENs2_info.csv`) and associated image information to generate an **LCC Layer**.

The LCC layer includes the following fields:
- **image_id** 
- **filename:** Relative path to the image file (e.g., `BigEarthNet-S2-v1.0-rgb/1016.jpg`)
- **cat_name:** The human-readable LCC category name
- **date:** Acquisition date in ISO 8601 format (UTC)
- **uoi:** A unique identifier assigned to each record
- **geometry:** A centroid point computed from the image’s bounding box

## Dataset Preparation

1. **Register and Download BigEarthNet:**
   - Visit [BigEarthNet Dataset Download Links](https://bigearth.net/#).
   - This script is built for the S2-v1.0 release (v2.0 support coming soon).

2. **Prep the Downloads:**
   - **Images:** Extract the image folder (by default named `BigEarthNet-S2-v1.0-rgb/`).
   - **Labels:** The CSV file (`BENs2_info.csv`) should be placed in the same folder.

3. **Folder Structure:**

   ```
   <DATASETS_PATH>/
   └── BigEarthNet/
       ├── BigEarthNet-S2-v1.0-rgb/
       └── BENs2_info.csv
   ```

   For example, if `<DATASETS_PATH>` is `/path/to/datasets`, then:
   - Labels: `/path/to/datasets/BigEarthNet/BENs2_info.csv`
   - Images: `/path/to/datasets/BigEarthNet/BigEarthNet-S2-v1.0-rgb/`

4. **Output GeoPackage:**

   The processed data will be saved under `<DATASETS_PATH>/gpkgs/` as:

   ```
   <DATASETS_PATH>/
   └── gpkgs/
         └── ben_lcc.gpkg
   ```

## What the Code Does

1. **Data Processing:**
   - Reads the `BENs2_info.csv` file.
   - For each image record:
     - Constructs a relative filename.
     - Parses the UTM projection string to extract zone, hemisphere, and ellipsoid.
     - Computes the image’s centroid from the upper-left (`ulx, uly`) and lower-right (`lrx, lly`) coordinates.
     - Converts the UTM centroid to latitude/longitude using the modern pyproj Transformer.
     - Expands the `cls_label` list to generate a row for each LCC class with its corresponding fields.

2. **Unique Identifier:**
   - Assigns a unique object identifier (`uoi`) to each record.

## Example Preview

**LCC Layer Preview:**
```      
   image_id                          filename                   cat_name                      date      uoi                   geometry
0         0  BigEarthNet-S2-v1.0-rgb/1016.jpg  Non-irrigated arable land 2017-06-13 10:10:31+00:00  ben_l_0  POINT (12.74030 48.11003)
1         0  BigEarthNet-S2-v1.0-rgb/1016.jpg        Broad-leaved forest 2017-06-13 10:10:31+00:00  ben_l_1  POINT (12.74030 48.11003)
2         0  BigEarthNet-S2-v1.0-rgb/1016.jpg               Mixed forest 2017-06-13 10:10:31+00:00  ben_l_2  POINT (12.74030 48.11003)
3         0  BigEarthNet-S2-v1.0-rgb/1016.jpg              Water courses 2017-06-13 10:10:31+00:00  ben_l_3  POINT (12.74030 48.11003)
4         1  BigEarthNet-S2-v1.0-rgb/1073.jpg  Non-irrigated arable land 2017-06-13 10:10:31+00:00  ben_l_4  POINT (12.75547 48.13193)
```

## Running the Code

1. **Clone the Repository**  
   Ensure you have the correct folder structure as described above.

2. **Install Dependencies:**  
   ```bash
   pip install pandas geopandas pyproj shapely tqdm
   ```

3. **Run the Script:**  
   ```bash
   python ingest_ben.py
   ```

The script will process the BigEarthNet CSV labels, compute image centroids, and write a GeoPackage file (`ben_lcc.gpkg`) with the layer `ben_lcc`.
