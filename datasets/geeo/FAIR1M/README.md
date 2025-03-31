# FAIR1M Data Ingestion & GeoPackage Creation

This repository contains a minimal example script for ingesting the FAIR1M dataset [paper](https://arxiv.org/abs/2103.05569) and converting it into two GeoPackage files containing cleaned geospatial data. The code processes the FAIR1M geojson (labels) and the associated image files, computes geometries, and splits the data into: 

- **Images Layer:** An aggregated view by image (using the point of each image, i.e., geometry column in image_centres_gdf parquet)  
- **Labels Layer:** The individual detections with full polygon geometries

Both layers include additional fields such as:
- `image_id`: The image filename (following the jpg_file_name column)
- `cat_id`: The numeric category ID (as defined in a simple mapping; see script)
- `cat_name`: The human-readable category name (as defined in parquets)
- `date`: FAIR1M doesn't provide dates, so we randomly sample ISO 8601 format (UTC) from 2015 to 2017 (dates mentioned in the paper)
- `filename`: The relative path to the image file (e.g., `FAIR1M/images/t_14154.jpg`)
- `geometry`: For images, the centroid point; for labels, the full polygon
- `uoi`: A unique identifier assigned to each record

## Dataset Preparation

1. **Download FAIR1M:**

   - Visit the FAIR1M website. Note: official website appears to be currently not available. 
   - Alternatively, there is an unofficial HuggingFace dataset card with a subset of the images [FAIRM1 HuggingFace](https://huggingface.co/datasets/blanchon/FAIR1M).
   - Our generation follows the entire set of 15k+ images (HuggingFace has a subset of 1,732)

2. **Unzip:**

   - The extracted folder should be named `images/` which has both the train (e.g., `t_14154.jpg`) and val images  (e.g., `v_707.jpg`)
   - The metadata files should be `geo_labels.parquet`, `image_centres_gdf.parquet`, `labels.parquet`, `image_boundaries_gdf.parquet`, `image_metadata.parquet`, `labels_geodata.parquet`.


3. **Folder Structure:**

   Create a folder structure as follows (the root folder is your `<DATASETS_PATH>`):

   ```
   <DATASETS_PATH>/
   └── FAIR1M/
       ├── images/
       └── *.parquet
   ```

   For example, if you set `DATASETS_PATH` to `/path/to/datasets`, then:
   - The geojson file will be at: `/path/to/datasets/FAIR1M/geo_labels.parquet`
   - The images will be in: `/path/to/datasets/FAIR1M/images/`


4. **Output GeoPackages:**

    The processed data will be saved in a dedicated folder inside your datasets directory. After running the script, you should have a folder structure similar to:

    ```
    <DATASETS_PATH>/
   └── gpkgs/
        ├── fair1m_images.gpkg
        └── fair1m_labels.gpkg
    ```


## What the Code Does

The provided script performs the following steps:

1. **Data Processing:**
   - Reads three parquet files: image centers, geo-labels (polygons and categories), and detection bounding boxes.
   - Converts original TIFF filenames to JPG paths using split-based prefixes.
   - Randomly assigns ISO 8601 UTC dates (from 2015 to 2017) to images.
   - Merges detection and geo-label data, and maps category names to both a numeric `cat_id` and a human-readable `cat_name`.
   - Constructs DataFrames with essential columns:
     - **Images Layer:** `filename`, `image_id`, `date`, `uoi`, `geometry` (centroid)
     - **Labels Layer:** `image_id`, `cat_id`, `cat_name`, `date`, `filename`, bounding box coordinates (`xmin`, `ymin`, `xmax`, `ymax`), `uoi`, `geometry` (full polygon)

2. **Data Splitting:**
   - Aggregates data by image to create an **Images Layer** (using the centroid as the geometry).
   - Retains individual detections as the **Labels Layer** (using the full polygon geometry).

3. **Unique Identifier:**
   - Assigns a unique identifier (`uoi`) to each record.


### Example Preview

**Images Layer Preview:**
```
                        filename  image_id                             date               uoi                     geometry
0      FAIR1M/images/t_14154.jpg         0 2015-11-26 05:40:43.973000+00:00      fair1m_img_0   POINT (122.10165 30.00458)
1        FAIR1M/images/t_412.jpg         1 2016-07-19 14:44:54.879000+00:00      fair1m_img_1  POINT (-111.84916 40.75937)
2      FAIR1M/images/t_12718.jpg         2 2016-01-14 04:06:12.417000+00:00      fair1m_img_2   POINT (118.12748 24.54292)
3       FAIR1M/images/t_8019.jpg         3 2015-05-03 08:49:00.415000+00:00      fair1m_img_3    POINT (139.70142 35.4895)
..                            ...       ...                       ...             ...                          ...
```

**Labels Layer Preview:**
```
        image_id                   filename  cat_id        cat_name                             date  xmin  ymin  xmax  ymax                 uoi                                           geometry
0              0  FAIR1M/images/t_14154.jpg      31         Warship 2015-11-26 05:40:43.973000+00:00     0   554   245   651       fair1m_bbox_0  POLYGON ((122.0989 30.00414, 122.10023 30.0037...))
1              0  FAIR1M/images/t_14154.jpg      31         Warship 2015-11-26 05:40:43.973000+00:00     0   526   255   618       fair1m_bbox_1  POLYGON ((122.09895 30.00428, 122.10029 30.003...))
2              0  FAIR1M/images/t_14154.jpg      11  Dry Cargo Ship 2015-11-26 05:40:43.973000+00:00   834   699   856   736       fair1m_bbox_2  POLYGON ((122.10356 30.00334, 122.10348 30.003...))
3              0  FAIR1M/images/t_14154.jpg      11  Dry Cargo Ship 2015-11-26 05:40:43.973000+00:00   864   906   878   931       fair1m_bbox_3  POLYGON ((122.1036 30.00239, 122.10365 30.0022...))
4              1    FAIR1M/images/t_412.jpg      14     Cargo Truck 2016-07-19 14:44:54.879000+00:00    94   328   126   339       fair1m_bbox_4  POLYGON ((-111.85016 40.75981, -111.85024 40.7...))
```

## Running the Code

1. **Clone the Repository**  
   Make sure you have the correct folder structure as described above.
2. **Install Dependencies:**  
   ```bash
   pip install pandas geopandas
   ```
3. **Run the Script:**  
   ```bash
   python ingest_fair1m.py
   ```
   The script will read the FAIR1M data, process it, and write two GeoPackage files in the specified output locations.

