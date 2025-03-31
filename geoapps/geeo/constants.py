import os

# Define the path to the GeoPackage
GPKG_FOLDER = 'datasets/gpkgs'

DATASETS_INFO = {
    "xView1": {
        "images_file": "geeo25/xview1_images.gpkg",
        "images_layer": "xview1_images",
        "images_date_column": "date",
        "labels_file": "geeo25/xview1_labels.gpkg",
        "labels_layer": "xview1_labels",
        "labels_date_column": "date",
        },
    "FAIR1M": {
        "images_file": "geeo25/fair1m_images.gpkg",
        "images_layer": "fair1m_images",
        "images_date_column": "date",
        "labels_file": "geeo25/fair1m_labels.gpkg",
        "labels_layer": "fair1m_labels",
        "labels_date_column": "date",
        },
    "BigEarthNet": {
        "images_file": "geeo25/ben_lcc.gpkg",
        "images_layer": "ben_lcc",
        "images_date_column": "date",
        "labels_file": "geeo25/ben_lcc.gpkg",
        "labels_layer": "ben_lcc",
        "labels_date_column": "date",
        }
}

DATASETS_KEYS = list(DATASETS_INFO.keys())

DATASET_ERROR_MSG = """
Error loading dataset. {dataset} does not exist in the database. 
Please confirm the correct name is selected. Available datasets: {datasets}.
"""

IMGS_NONE_MSG = (
    "No images loaded (images GeoDataFrame is None). Make sure to query images first "
    "for the specified user criteria."
)

IMGS_EMPTY_MSG = (
    "No images found (images GeoDataFrame is empty). No areas with available imagery "
    "for the specified user criteria."
)

DETS_NONE_MSG = (
    "No detections (detections GeoDataFrame is None) for . Make sure to load the underlying imagery first "
    "for the specified user criteria and/or run detection with the proper detector model."
)

DETS_EMPTY_MSG = (
    "No detections found (detections GeoDataFrame is empty). "
    "It is probable that the specified category does not contain any objects in the location of interest. "
    "No further action needed wrt this operation."
)


LCC_NONE_MSG = (
    "No LCC classes (lcc GeoDataFrame is None) for . Make sure to load the underlying imagery first "
    "for the specified user criteria and/or run LCC classification with the proper detector model."
)

LCC_EMPTY_MSG = (
    "No LCC classes found (lcc GeoDataFrame is empty). "
    "It is probable that the specified LCC category does not contain any objects in the location of interest. "
    "No further action needed wrt this operation."
)

DATASET_CATEGORY_ERROR_MSG = """
Error selecting object category. {category_name} does not exist in the dataset categories (classes) names. 
Please confirm the correct category name is selected. Available dataset object categories: {category_names}.
"""

# Date range: 2017-06-13 10:10:31+00:00 to 2018-05-29 11:54:01+00:00 # BigEarthNet
# Date range: 2015-01-01T00:34:59.526001+00:00 to 2017-12-30T23:28:49.844002+00:00 # FAIR1M
# Date range: 2017-06-29 14:14:28+00:00 to 2017-10-18 17:28:31+00:00 # xView1

DATASETS_CATEGORIES = {}
DATASETS_CATEGORIES["xView1"] = ['Truck Tractor w/ Liquid Tank', 'Ground Grader', 'Tower', 'Locomotive', 'Dump Truck', 'Fixed-wing Aircraft',
                      'Maritime Vessel', 'Straddle Carrier', 'Tugboat', 'Hut/Tent', 'Motorboat', 'Small Car', 'Engineering Vehicle',
                      'Tank car', 'Truck Tractor', 'Ferry', 'Cargo Truck', 'Container Ship', 'Building', 'Truck', 'Trailer',
                      'Vehicle Lot', 'Aircraft Hangar', 'Facility', 'Fishing Vessel', 'Sailboat', 'Reach Stacker', 'Scraper/Tractor',
                      'Shipping container lot', 'Cement Mixer', 'Shed', 'Storage Tank', 'Excavator', 'Truck Tractor w/ Flatbed Trailer',
                      'Tower crane', 'Front loader/Bulldozer', 'Helicopter', 'Railway Vehicle', 'Container Crane', 'Pickup Truck',
                      'Flat Car', 'Passenger Car', 'Passenger Vehicle', 'Barge', 'Small Aircraft', 'Helipad', 'Passenger/Cargo Plane',
                      'Utility Truck', 'Haul Truck', 'Crane Truck', 'Yacht', 'Construction Site', 'Oil Tanker', 'Mobile Crane',
                      'Shipping Container', 'Bus', 'Truck Tractor w/ Box Trailer', 'Damaged Building', 'Cargo/Container Car', 'Pylon']
DATASETS_CATEGORIES["FAIR1M"] = ['Fishing Boat', 'Boeing787', 'A350', 'Passenger Ship', 'other-ship', 'Dump Truck', 'Boeing747', 'Liquid Cargo Ship',
                      'Tugboat', 'Motorboat', 'Dry Cargo Ship', 'Small Car', 'Truck Tractor', 'Cargo Truck', 'Boeing777', 'Engineering Ship',
                      'Trailer', 'Tractor', 'Baseball Field', 'Boeing737', 'Excavator', 'A220', 'Basketball Court', 'ARJ21',
                      'Football Field', 'A321', 'other-airplane', 'C919', 'Van', 'Tennis Court', 'Warship', 'A330', 'Bridge',
                      'other-vehicle', 'Bus', 'Roundabout', 'Intersection']
DATASETS_CATEGORIES["BigEarthNet"] = ['Port areas', 'Fruit trees and berry plantations', 'Annual crops associated with permanent crops', 'Mineral extraction sites', 
                        'Permanently irrigated land', 'Natural grassland', 'Estuaries', 'Bare rock', 'Salt marshes', 'Burnt areas', 'Olive groves', 
                        'Intertidal flats', 'Sparsely vegetated areas', 'Peatbogs', 'Airports', 'Land principally occupied by agriculture, with significant areas of natural vegetation', 
                        'Transitional woodland/shrub', 'Salines', 'Discontinuous urban fabric', 'Continuous urban fabric', 'Inland marshes', 'Sea and ocean', 'Coastal lagoons', 
                        'Industrial or commercial units', 'Sport and leisure facilities', 'Non-irrigated arable land', 'Construction sites', 'Agro-forestry areas', 'Water bodies', 
                        'Dump sites', 'Pastures', 'Green urban areas', 'Broad-leaved forest', 'Beaches, dunes, sands', 'Sclerophyllous vegetation', 'Vineyards', 'Complex cultivation patterns', 
                        'Rice fields', 'Water courses', 'Coniferous forest', 'Moors and heathland', 'Road and rail networks and associated land', 'Mixed forest']
