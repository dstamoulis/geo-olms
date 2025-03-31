# xView1 Data Ingestion & GeoPackage Creation

This repository contains a minimal example script for ingesting the xView1 dataset and converting it into two GeoPackage files containing cleaned geospatial data. The code processes the xView1 geojson (labels) and the associated image files, computes geometries, and splits the data into:

- **Images Layer:** An aggregated view by image (using the centroid of each image’s polygon)  
- **Labels Layer:** The individual detections with full polygon geometries

Both layers include additional fields such as:
- `image_id`: The original image filename
- `cat_id`: The numeric category ID (as defined in a global mapping)
- `cat_name`: The human-readable category name
- `date`: Acquisition date in ISO 8601 format (with UTC timezone)
- `filename`: The relative path to the image file (e.g., `xView1/train_images/389.tif`)
- `geometry`: For images, the centroid point; for labels, the full polygon
- `uoi`: A unique identifier assigned to each record

## Dataset Preparation

1. **Register and Download xView1:**

   - Visit [xView Dataset Download Links](https://challenge.xviewdataset.org/download-links) (registration is required).
   - Download the Training Images and Training Labels .zips

2. **Unzip the Downloads:**

   - Unzip the **Training Images** zip file. By default, the extracted folder should be named `train_images/`.
   - Unzip the **Training Labels** zip file. The resulting file is `xView_train.geojson`.

3. **Folder Structure:**

   Create a folder structure as follows (the root folder is your `<DATASETS_PATH>`):

   ```
   <DATASETS_PATH>/
   └── xView1/
       ├── train_images/
       └── xView_train.geojson
   ```

   For example, if you set `DATASETS_PATH` to `/path/to/datasets`, then:
   - The geojson file will be at: `/path/to/datasets/xView1/xView_train.geojson`
   - The images will be in: `/path/to/datasets/xView1/train_images/`


4. **Output GeoPackages:**

    The processed data will be saved in a dedicated folder inside your datasets directory. After running the script, you should have a folder structure similar to:

    ```
    <DATASETS_PATH>/
   └── gpkgs/
        ├── xview1_images.gpkg
        └── xview1_labels.gpkg
    ```


## What the Code Does

The provided script performs the following steps:

1. **Data Processing:**
   - Reads the xView1 geojson and processes each feature.
   - Converts the `ingest_time` field to ISO 8601 format.
   - Maps the original category ID to both a numeric `cat_id` and a human-readable `cat_name` using global mapping dictionaries.
   - Extracts the full polygon geometry from the geojson and computes its centroid.
   - Constructs a DataFrame with essential columns:
     - `image_id`
     - `cat_id`
     - `cat_name`
     - `date`
     - `filename` (stored as a relative path like `xView1/train_images/389.tif`)
     - `geom` (the full polygon geometry)
     - `centroid` (the computed centroid)
2. **Data Splitting:**
   - Aggregates the data by image to create an **Images Layer** (using the centroid as the geometry).
   - Retains the individual features as the **Labels Layer** (using the full polygon geometry).
3. **Unique Identifier:**
   - Assigns a unique identifier (`uoi`) to each record.


### Example Preview

**Images Layer Preview:**
```
                         filename  image_id                      date             uoi                     geometry
0      xView1/train_images/10.tif    10.tif 2017-08-10 18:50:49+00:00    xview1_img_0   POINT (-23.47534 14.95408)
1     xView1/train_images/100.tif   100.tif 2017-08-31 18:34:42+00:00    xview1_img_1    POINT (10.25216 36.87206)
2     xView1/train_images/102.tif   102.tif 2017-08-14 22:59:32+00:00    xview1_img_2    POINT (10.21856 36.87498)
3    xView1/train_images/1036.tif  1036.tif 2017-09-15 15:29:41+00:00    xview1_img_3    POINT (103.98516 1.32192)
4    xView1/train_images/1037.tif  1037.tif 2017-09-13 14:59:01+00:00    xview1_img_4    POINT (103.98638 1.31742)
..                            ...       ...                       ...             ...                          ...
```

**Labels Layer Preview:**
```
        image_id                      filename  cat_id  cat_name                      date  xmin  ymin  xmax  ymax                 uoi                                           geometry
0       2355.tif  xView1/train_images/2355.tif      48  Building 2017-07-24 12:49:09+00:00  2712  1145  2746  1177       xview1_bbox_0  POLYGON ((-90.5317 14.56604, -90.5317 14.56614...))
1       2355.tif  xView1/train_images/2355.tif      48  Building 2017-07-24 17:26:05+00:00  2720  2233  2760  2288       xview1_bbox_1  POLYGON ((-90.53167 14.56222, -90.53167 14.562...))
2       2355.tif  xView1/train_images/2355.tif      48  Building 2017-07-24 12:45:09+00:00  2687  1338  2740  1399       xview1_bbox_2  POLYGON ((-90.53179 14.56527, -90.53179 14.565...))
3       2355.tif  xView1/train_images/2355.tif      48  Building 2017-07-24 12:49:09+00:00  2691  1201  2730  1268       xview1_bbox_3  POLYGON ((-90.53177 14.56572, -90.53177 14.565...))
4       2355.tif  xView1/train_images/2355.tif      48  Building 2017-07-24 13:20:38+00:00  2671   838  2714   869       xview1_bbox_4  POLYGON ((-90.53184 14.5671, -90.53184 14.5672...))
```

## Running the Code

1. **Clone the Repository**  
   Make sure you have the correct folder structure as described above.
2. **Install Dependencies:**  
   ```bash
   pip install pandas geopandas sqlalchemy geoalchemy2 psycopg2-binary shapely
   ```
3. **Run the Script:**  
   ```bash
   python ingest_xview1.py
   ```
   The script will read the xView1 data, process it, and write two GeoPackage files in the specified output locations.

