import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd

# -------------------------------
# Settings and input paths
# -------------------------------
# TODO!!! Set this to your location!!
DATASETS_PATH = '/path/to/datasets'
data_path = os.path.join(DATASETS_PATH, 'FAIR1M')
gpkg_path = os.path.join(DATASETS_PATH, 'gpkgs')

# Parquet filenames in the FAIR1M folder
geo_labels_fname = 'geo_labels.parquet'
image_centres_fname = 'image_centres_gdf.parquet'
labels_fname = 'labels.parquet'

# Full paths to input files
geo_labels_path = os.path.join(data_path, geo_labels_fname)
image_centres_path = os.path.join(data_path, image_centres_fname)
labels_path = os.path.join(data_path, labels_fname)

# -------------------------------
# 1. Process images (fair1m_images)
# -------------------------------
# Load image centres; these include the original tiff filename and split info.
gdf_images = gpd.read_parquet(image_centres_path)

# Create new filenames based on the original tiff file.
# For example: "train_images/14154.tif" becomes "FAIR1M/images/t_14154.jpg" if split=="Train"
def convert_filename(row):
    # Extract the numeric part (or basename without extension)
    base = os.path.splitext(os.path.basename(row['filename']))[0]
    # Use prefix based on split: "t_" for Train, "v_" for Val (adjust if additional splits exist)
    prefix = "t_" if row['split'] == "Train" else "v_"
    return f"FAIR1M/images/{prefix}{base}.jpg"

gdf_images['filename'] = gdf_images.apply(convert_filename, axis=1)

# Rename Img_ID to image_id to be consistent with target schema.
gdf_images = gdf_images.rename(columns={'Img_ID': 'image_id'})

# Generate random dates uniformly between 2015 and 2017.
# (Dates are generated as UTC timestamps.)
n_images = len(gdf_images)
start_date = pd.Timestamp("2015-01-01", tz="UTC")
end_date = pd.Timestamp("2017-12-31", tz="UTC")
start_ns = start_date.value  # nanosecond timestamp
end_ns = end_date.value

# Generate random nanosecond timestamps and convert to datetime with UTC timezone.
rand_ns = np.random.randint(start_ns, end_ns, n_images)
rand_dates = pd.to_datetime(rand_ns).tz_localize("UTC")
gdf_images['date'] = rand_dates

# Create a unique object identifier (uoi) for each image.
gdf_images['uoi'] = gdf_images.apply(lambda row: f"fair1m_img_{row.name}", axis=1)

# Keep only the desired columns and order them.
gdf_images = gdf_images[['filename', 'image_id', 'date', 'uoi', 'geometry']]

# -------------------------------
# 2. Process labels (fair1m_labels)
# -------------------------------
# Load the geo_labels which contains the geometry polygons and Category
gdf_geo_labels = gpd.read_parquet(geo_labels_path)

# Load the labels parquet which has bounding box coordinates and additional info.
df_labels = pd.read_parquet(labels_path)

# Merge the labels and geo_labels on the internal identifiers (Img_ID and Label_ID)
# These two datasets share the columns "Img_ID" and "Label_ID".
labels_merged = df_labels.merge(
    gdf_geo_labels[['Img_ID', 'Label_ID', 'geometry', 'Category']],
    on=['Img_ID', 'Label_ID'],
    how='left',
    suffixes=('_labels', '')
)

# Rename Img_ID to image_id for consistency.
labels_merged = labels_merged.rename(columns={'Img_ID': 'image_id'})

# Next, we want the label rows to have the same filename and date as the corresponding image.
# Merge with the images GeoDataFrame (or a subset of it) on image_id.
labels_merged = labels_merged.merge(
    gdf_images[['image_id', 'filename', 'date']],
    on='image_id',
    how='left'
)

# Create categorical columns:
# - cat_name comes from the Category column
# - cat_id is determined by a fixed mapping.
fair1m_cats_dict = {
    'Fishing Boat': 1,
    'Boeing787': 2,
    'A350': 3,
    'Passenger Ship': 4,
    'other-ship': 5,
    'Dump Truck': 6,
    'Boeing747': 7,
    'Liquid Cargo Ship': 8,
    'Tugboat': 9,
    'Motorboat': 10,
    'Dry Cargo Ship': 11,
    'Small Car': 12,
    'Truck Tractor': 13,
    'Cargo Truck': 14,
    'Boeing777': 15,
    'Engineering Ship': 16,
    'Trailer': 17,
    'Tractor': 18,
    'Baseball Field': 19,
    'Boeing737': 20,
    'Excavator': 21,
    'A220': 22,
    'Basketball Court': 23,
    'ARJ21': 24,
    'Football Field': 25,
    'A321': 26,
    'other-airplane': 27,
    'C919': 28,
    'Van': 29,
    'Tennis Court': 30,
    'Warship': 31,
    'A330': 32,
    'Bridge': 33,
    'other-vehicle': 34,
    'Bus': 35,
    'Roundabout': 36,
    'Intersection': 37
}

labels_merged['cat_name'] = labels_merged['Category']
labels_merged['cat_id'] = labels_merged['cat_name'].map(fair1m_cats_dict)

# Create unique identifier for each label (uoi)
labels_merged['uoi'] = labels_merged.apply(lambda row: f"fair1m_bbox_{row.name}", axis=1)

# Note: The xmin, ymin, xmax, ymax values come directly from the labels parquet.
# Adjust the column names if needed; here we assume the columns are named as follows.
# (If they are named differently—e.g., x_min instead of xmin—rename them accordingly.)
labels_merged = labels_merged.rename(columns={
    'x_min': 'xmin',
    'y_min': 'ymin',
    'x_max': 'xmax',
    'y_max': 'ymax'
})

# Select and order the final columns.
labels_final = labels_merged[[
    'image_id',    # corresponds to Img_ID from the original
    'filename',    # taken from the corresponding image row
    'cat_id',      # mapped from the Category string
    'cat_name',    # the original Category string
    'date',        # taken from the corresponding image row
    'xmin', 'ymin', 'xmax', 'ymax',
    'uoi',
    'geometry'
]]

# Convert to a GeoDataFrame. Use the same CRS as the images (if applicable).
gdf_labels = gpd.GeoDataFrame(labels_final, geometry='geometry', crs=gdf_images.crs)

# -------------------------------
# 3. Save to GeoPackage files
# -------------------------------
images_gpkg = os.path.join(gpkg_path, 'fair1m_images.gpkg')
labels_gpkg = os.path.join(gpkg_path, 'fair1m_labels.gpkg')

# Save the images GeoDataFrame with layer name 'fair1m_images'
gdf_images.to_file(images_gpkg, layer='fair1m_images', driver="GPKG")

# Save the labels GeoDataFrame with layer name 'fair1m_labels'
gdf_labels.to_file(labels_gpkg, layer='fair1m_labels', driver="GPKG")

print("GeoPackages successfully written:")
print(" - Images:", images_gpkg)
print(" - Labels:", labels_gpkg)
