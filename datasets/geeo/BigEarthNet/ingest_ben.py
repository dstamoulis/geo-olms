from pyproj import CRS, Transformer
import pandas as pd
import re, os
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point

def parse_projection(projection):
    utm_zone_pattern = re.compile(r'UTM zone (\d+)([NS])')
    datum_ellps_pattern = re.compile(r'DATUM\["([^"]+)",SPHEROID\["([^"]+)"')
    
    utm_match = utm_zone_pattern.search(projection)
    if utm_match:
        zone = utm_match.group(1)
        hemisphere = utm_match.group(2)
        north = True if hemisphere == 'N' else False
    else:
        zone, hemisphere, north = None, None, None

    datum_ellps_match = datum_ellps_pattern.search(projection)
    if datum_ellps_match:
        datum = datum_ellps_match.group(1).replace("_", "")
        ellps = datum_ellps_match.group(2).replace(" ", "")
        # Following original code, overwrite datum with ellps
        datum = ellps  
    else:
        datum, ellps = None, None
        
    return zone, hemisphere, datum, ellps, north

def get_images(df, img_path):
    lcc_data = []
    lcc_classes = []
    total_rows = len(df)
    
    # Dictionary to map filename to a unique image_id
    filename_to_image_id = {}
    next_image_id = 0

    debug_cnt = 0
    
    for index, row in tqdm(df.iterrows(), total=total_rows, desc="Processing data"):
        img_name = row['img_name']
        # Construct filename as a relative path (e.g., "BigEarthNet-S2-v1.0-rgb/01.jpg")
        filename = os.path.join("BigEarthNet-S2-v1.0-rgb", str(img_name) + ".jpg")

        # Assign a unique image_id per filename
        if filename not in filename_to_image_id:
            filename_to_image_id[filename] = next_image_id
            next_image_id += 1
        image_id = filename_to_image_id[filename]

        projection = row['projection']
        cls_label = row['cls_label']
        acquisition_time = row['acquisition_time']
        ulx, uly = row['ulx'], row['uly']
        lrx, lly = row['lrx'], row['lly']

        zone, hemisphere, datum, ellps, north = parse_projection(projection)
        # Determine the EPSG code for UTM.
        # For UTM, if north is True: EPSG code = 32600 + zone, else 32700 + zone.
        if zone is not None:
            zone = int(zone)
            epsg_code = 32600 + zone if north else 32700 + zone
        else:
            raise ValueError("Could not parse UTM zone from projection.")

        source_crs = CRS.from_epsg(epsg_code)
        target_crs = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        
        center_x = (ulx + lrx) / 2
        center_y = (uly + lly) / 2
        
        center_lon, center_lat = transformer.transform(center_x, center_y)
        centroid = Point(center_lon, center_lat)
        
        # cls_label is stored as a string representation of a list; evaluate it.
        cls_label_list = eval(cls_label)
        for cat_name in cls_label_list:
            new_row = [image_id, filename, cat_name, acquisition_time, centroid]
            lcc_data.append(new_row)
            lcc_classes.append(cat_name)

        # debug_cnt += 1
        # if debug_cnt > 100: break
            
    return lcc_data, list(set(lcc_classes))



# Define paths -- TODO!!! Set this to your location!!
DATASETS_PATH = '/path/to/datasets'


BEN_FOLDER = os.path.join(DATASETS_PATH, "BigEarthNet")
BEN_LABELS = os.path.join(BEN_FOLDER, "BENs2_info.csv")
IMAGES_PATH = os.path.join(BEN_FOLDER, "BigEarthNet-S2-v1.0-rgb")
GPKG_FOLDER = os.path.join(DATASETS_PATH, "gpkgs/GEEO25")
os.makedirs(GPKG_FOLDER, exist_ok=True)
GPKG_PATH_LCC = os.path.join(GPKG_FOLDER, "ben_lcc.gpkg")

# Read CSV info
df = pd.read_csv(BEN_LABELS)
lcc_data, lcc_classes = get_images(df, IMAGES_PATH)

# Build a DataFrame with essential columns:
COLUMNS = ['image_id', 'filename', 'cat_name', 'date', 'geometry']
lcc_df = pd.DataFrame(lcc_data, columns=COLUMNS)
lcc_df['date'] = pd.to_datetime(lcc_df['date'], format="%Y-%m-%d %H:%M:%S", errors='coerce', utc=True)

# Create unique identifier (uoi) for each row
lcc_df['uoi'] = lcc_df.apply(lambda row: f"ben_l_{row.name}", axis=1)

# Reorder columns as: filename, cat_name, date, uoi, geometry
lcc_df = lcc_df[['image_id', 'filename', 'cat_name', 'date', 'uoi', 'geometry']]

# Convert to GeoDataFrame with CRS EPSG:4326 and save as a GeoPackage.
gdf_lcc = gpd.GeoDataFrame(lcc_df, geometry='geometry', crs="EPSG:4326")
gdf_lcc.to_file(GPKG_PATH_LCC, layer='ben_lcc', driver="GPKG")
