import os
import pandas as pd
import geopandas as gpd
import datetime
from shapely.geometry import box
from typing import Optional

from geoplatform.utils import build_where_date_clause
from geoapps.geeo.constants import (
    IMGS_NONE_MSG, IMGS_EMPTY_MSG, 
    DATASET_ERROR_MSG,
    DATASETS_INFO, DATASETS_KEYS, GPKG_FOLDER
)

from agent_core.modules.toolset import agent_tool

class Database:

    def __init__(self):
        """
        Initialize with the path to the GeoPackage.
        """
        self.name = "database"
        self.images_gdf = {}

    def reset_database(self):
        self.images_gdf = {}

    def _load_gpkg_images(self, dataset, start_date=None, end_date=None):
        """
        Internal helper that loads images for the given date range using SQL filtering.
        Returns a GeoDataFrame.
        """
        where_clause = build_where_date_clause(start_date, end_date, DATASETS_INFO[dataset].get("images_date_column", "date")) # WHERE: [start_date, end_date)
        gpkg_path = os.path.join(GPKG_FOLDER, DATASETS_INFO[dataset]["images_file"])
        gpkg_layer = DATASETS_INFO[dataset]["images_layer"]
        return gpd.read_file(gpkg_path, layer=gpkg_layer, where=where_clause)


    def _update_images_gdf(self, dataset: str, _gdf: gpd.GeoDataFrame) -> None:

        # If self.gdf_images[dataset] is None, simply assign _gdf
        if dataset not in self.images_gdf:
            self.images_gdf[dataset] = _gdf.copy()
        else:
            # Concatenate and drop duplicates using the 'uoi' column
            merged_gdf = pd.concat([self.images_gdf[dataset], _gdf], ignore_index=True)
            merged_gdf = merged_gdf.drop_duplicates(subset=['uoi'])
            self.images_gdf[dataset] = merged_gdf

    @agent_tool
    def query_dataset_images(
        self,
        dataset: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> str:
        """
        Return all images for a specified dataset (and if set, within the given date range).
        This function filters and loads imagery from the images GeoPackage 
        based on the provided dataset (and start and end dates, if given).

        Args:
            dataset (str): The satellite imagery dataset to use.
            start_date (str, optional): The start date in ISO format (YYYY-MM-DD).
            end_date (str, optional): The end date in ISO format (YYYY-MM-DD).

        Returns:
            str: A message indicating successful completion and loading into system memory.
        """
        if dataset not in DATASETS_INFO: return DATASET_ERROR_MSG.format(dataset=dataset, datasets=DATASETS_KEYS)
        _gdf = self._load_gpkg_images(dataset, start_date, end_date)
        self._update_images_gdf(dataset, _gdf)
        return f"The satellite {dataset} imagery is loaded. Num images: {len(self.images_gdf[dataset])}."


    @agent_tool
    def query_images_by_aoi_coords(
        self, 
        dataset: str,
        lat_min: float, 
        lat_max: float, 
        lon_min: float, 
        lon_max: float,
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> str:
        """
        Return images that intersect with the AOI defined by the given latitude and longitude bounds.
        Optionally, filter by a specified date range.

        Args:
            dataset (str): The satellite imagery dataset to use.
            lat_min (float): Minimum latitude of the AOI.
            lat_max (float): Maximum latitude of the AOI.
            lon_min (float): Minimum longitude of the AOI.
            lon_max (float): Maximum longitude of the AOI.
            start_date (str, optional): The start date in ISO format (YYYY-MM-DD).
            end_date (str, optional): The end date in ISO format (YYYY-MM-DD).

        Returns:
            str: A message indicating successfully completion and loading into system memory.
        """
        if dataset not in DATASETS_INFO: return DATASET_ERROR_MSG.format(dataset=dataset, datasets=DATASETS_KEYS)

        _gdf = self._load_gpkg_images(dataset, start_date, end_date)
        if _gdf.empty: return IMGS_EMPTY_MSG
        
        # Create an AOI bounding box. Note: shapely.geometry.box takes (minx, miny, maxx, maxy) 
        # where x corresponds to longitude and y to latitude.
        aoi = box(lon_min, lat_min, lon_max, lat_max)
        
        # Filter the images GeoDataFrame to keep only those whose geometry intersects with the AOI.
        _gdf = _gdf[_gdf.geometry.intersects(aoi)]
        
        # Update the stored images with the filtered result.
        self._update_images_gdf(dataset, _gdf)
        return f"The satellite {dataset} imagery within the specified AOI is loaded in system memory. Num images: {len(self.images_gdf[dataset])}."
