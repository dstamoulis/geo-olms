import os
import json
import random
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Point

# ------------------------------
# CONFIGURATION -- TODO!!! Set this to your location!!
# ------------------------------
DATASETS_PATH = '/path/to/datasets'

XVIEW1_FOLDER = os.path.join(DATASETS_PATH, "xView1")
LABELS_PATH = os.path.join(XVIEW1_FOLDER, "xView_train.geojson")
IMAGES_PATH = os.path.join(XVIEW1_FOLDER, "train_images")

GPKG_FOLDER = os.path.join(DATASETS_PATH, "gpkgs")
os.makedirs(GPKG_FOLDER, exist_ok=True)

GPKG_PATH_IMAGES = os.path.join(GPKG_FOLDER, "xview1_images.gpkg")
GPKG_PATH_LABELS = os.path.join(GPKG_FOLDER, "xview1_labels.gpkg")

# ------------------------------
# CATEGORY MAPPINGS
# ------------------------------
old_dict = {
    11: 'Fixed-wing Aircraft', 12: 'Small Aircraft', 13: 'Passenger/Cargo Plane', 15: 'Helicopter',
    17: 'Passenger Vehicle', 18: 'Small Car', 19: 'Bus', 20: 'Pickup Truck', 21: 'Utility Truck',
    23: 'Truck', 24: 'Cargo Truck', 25: 'Truck Tractor w/ Box Trailer', 26: 'Truck Tractor',
    27: 'Trailer', 28: 'Truck Tractor w/ Flatbed Trailer', 29: 'Truck Tractor w/ Liquid Tank', 32: 'Crane Truck',
    33: 'Railway Vehicle', 34: 'Passenger Car', 35: 'Cargo/Container Car', 36: 'Flat Car', 37: 'Tank car',
    38: 'Locomotive', 40: 'Maritime Vessel', 41: 'Motorboat', 42: 'Sailboat', 44: 'Tugboat', 45: 'Barge',
    47: 'Fishing Vessel', 49: 'Ferry', 50: 'Yacht', 51: 'Container Ship', 52: 'Oil Tanker', 53: 'Engineering Vehicle',
    54: 'Tower crane', 55: 'Container Crane', 56: 'Reach Stacker', 57: 'Straddle Carrier', 59: 'Mobile Crane',
    60: 'Dump Truck', 61: 'Haul Truck', 62: 'Scraper/Tractor', 63: 'Front loader/Bulldozer', 64: 'Excavator',
    65: 'Cement Mixer', 66: 'Ground Grader', 71: 'Hut/Tent', 72: 'Shed', 73: 'Building', 74: 'Aircraft Hangar',
    76: 'Damaged Building', 77: 'Facility', 79: 'Construction Site', 83: 'Vehicle Lot', 84: 'Helipad',
    86: 'Storage Tank', 89: 'Shipping container lot', 91: 'Shipping Container', 93: 'Pylon', 94: 'Tower'
}
new_dict = {
    'Fixed-wing Aircraft': 0, 'Small Aircraft': 1, 'Passenger/Cargo Plane': 2, 'Helicopter': 3,
    'Passenger Vehicle': 4, 'Small Car': 5, 'Bus': 6, 'Pickup Truck': 7, 'Utility Truck': 8, 'Truck': 9,
    'Cargo Truck': 10, 'Truck Tractor w/ Box Trailer': 11, 'Truck Tractor': 12, 'Trailer': 13,
    'Truck Tractor w/ Flatbed Trailer': 14, 'Truck Tractor w/ Liquid Tank': 15, 'Crane Truck': 16,
    'Railway Vehicle': 17, 'Passenger Car': 18, 'Cargo/Container Car': 19, 'Flat Car': 20, 'Tank car': 21,
    'Locomotive': 22, 'Maritime Vessel': 23, 'Motorboat': 24, 'Sailboat': 25, 'Tugboat': 26, 'Barge': 27,
    'Fishing Vessel': 28, 'Ferry': 29, 'Yacht': 30, 'Container Ship': 31, 'Oil Tanker': 32,
    'Engineering Vehicle': 33, 'Tower crane': 34, 'Container Crane': 35, 'Reach Stacker': 36,
    'Straddle Carrier': 37, 'Mobile Crane': 38, 'Dump Truck': 39, 'Haul Truck': 40, 'Scraper/Tractor': 41,
    'Front loader/Bulldozer': 42, 'Excavator': 43, 'Cement Mixer': 44, 'Ground Grader': 45,
    'Hut/Tent': 46, 'Shed': 47, 'Building': 48, 'Aircraft Hangar': 49, 'Damaged Building': 50,
    'Facility': 51, 'Construction Site': 52, 'Vehicle Lot': 53, 'Helipad': 54, 'Storage Tank': 55,
    'Shipping container lot': 56, 'Shipping Container': 57, 'Pylon': 58, 'Tower': 59
}

def map_class(old_class):
    cat_name = old_dict.get(old_class, 'NA')
    cat_id_int = new_dict.get(cat_name, -1)
    return cat_id_int, cat_name

# ------------------------------
# DATA PROCESSING FUNCTIONS
# ------------------------------
def get_xview1_df(labels_path, images_path):
    """
    Reads and processes the xView1 geojson; returns a DataFrame with native geometries.
    For each feature, it extracts:
      - image_id (e.g., "389.tif")
      - date (converted to ISO 8601)
      - a relative filename (e.g., "xView1/train_images/389.tif")
      - the full polygon geometry as 'geom'
      - the centroid as 'centroid'
      - numeric (cat_id) and string (cat_name) category representations.
    """
    with open(labels_path, 'r') as f:
        data = json.load(f)
    features = data['features']
    rows = []
    for feature in features:
        props = feature['properties']
        image_id = props['image_id']       # e.g., "389.tif"
        rel_filename = os.path.join("xView1", "train_images", image_id)
        date_str = props['ingest_time']  # e.g., "2017/07/24 12:49:09.118+00"
        bbox = props['bounds_imcoords'].split(",")  #'1917,38,1958,64'
        try:
            daydate, time_part = date_str.split()
        except Exception:
            continue  # Skip if parsing fails.
        iso_date = '-'.join(daydate.split('/'))  # "YYYY-MM-DD"
        time_clean = time_part.split('.')[0]       # "HH:MM:SS"
        iso_datetime = f"{iso_date}T{time_clean}"
        orig_cat_id = props['type_id']
        cat_id_int, cat_name = map_class(orig_cat_id)
        geom = shape(feature['geometry'])
        centroid = geom.centroid
        rows.append([image_id, cat_id_int, cat_name, iso_datetime, rel_filename, geom, centroid, bbox[0], bbox[1], bbox[2], bbox[3],])
    columns = ['image_id', 'cat_id', 'cat_name', 'date', 'filename', 'geom', 'centroid', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(rows, columns=columns)
    df = df[df['image_id'] != '1395.tif']
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%dT%H:%M:%S", errors='coerce', utc=True)
    df = df.dropna(subset=['date'])
    return df

def gen_images(df):
    """
    Groups the input DataFrame by 'filename' and takes the first 'image_id' and 'centroid' found.
    It randomly selects one of the available 'date' values for the image.
    Returns a new DataFrame with columns: ['filename', 'image_id', 'centroid', 'date'].
    """
    im_df = df.groupby('filename').agg({
        'image_id': 'first',
        'centroid': 'first',
        'date': lambda x: random.choice(x.tolist())
    }).reset_index()
    im_df['date'] = pd.to_datetime(im_df['date'], utc=True)
    return im_df

def gen_labels(df):
    """
    Generates a labels DataFrame from the input DataFrame using the full polygon geometry ('geom').
    Keeps image_id, cat_id, cat_name, date, filename, and geom.
    """
    lb_df = df[['image_id', 'filename', 'cat_id', 'cat_name', 'date', 'geom', 'xmin', 'ymin', 'xmax', 'ymax']].copy()
    return lb_df

def create_uoi_img(row):
    return f"xview1_img_{row.name}"

def create_uoi_bbox(row):
    return f"xview1_bbox_{row.name}"


# ------------------------------
# MAIN INGESTION & GEOPACKAGE OUTPUT
# ------------------------------
def main():
    df = get_xview1_df(os.path.join(DATASETS_PATH, "xView1", "xView_train.geojson"), IMAGES_PATH)
    images_df = gen_images(df)
    labels_df = gen_labels(df)
    
    # Add unique identifiers.
    images_df['uoi'] = images_df.apply(create_uoi_img, axis=1)
    labels_df['uoi'] = labels_df.apply(create_uoi_bbox, axis=1)
    
    # Convert DataFrames to GeoDataFrames.
    gdf_images = gpd.GeoDataFrame(images_df, geometry='centroid', crs="EPSG:4326")
    gdf_labels = gpd.GeoDataFrame(labels_df, geometry='geom', crs="EPSG:4326")
    
    gdf_images.to_file(GPKG_PATH_IMAGES, layer='xview1_images', driver="GPKG")
    gdf_labels.to_file(GPKG_PATH_LABELS, layer='xview1_labels', driver="GPKG")
    
    print("Data successfully saved to GeoPackages:")
    print(f"  Images: {GPKG_PATH_IMAGES}")
    print(f"  Labels: {GPKG_PATH_LABELS}")

if __name__ == "__main__":
    main()
