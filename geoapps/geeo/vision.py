import os
import pandas as pd
import datetime
import geopandas as gpd
from typing import Optional


from geoplatform.utils import build_where_date_clause
from geoapps.geeo.constants import (
    IMGS_NONE_MSG, IMGS_EMPTY_MSG, DATASET_ERROR_MSG,
    DATASETS_INFO, GPKG_FOLDER
)
from geoapps.geeo.model_cards import (
    DETECTOR_MODELS, DETECTOR_MODEL_ERROR_MSG, DETECTOR_DATASET_ERROR_MSG,
    LCC_CLASSIFIER_MODELS, LCC_CLASSIFIER_DATASET_ERROR_MSG, LCC_CLASSIFIER_MODEL_ERROR_MSG
)

from agent_core.modules.toolset import agent_tool


class Vision:
    def __init__(self, database) -> None:
        """
        Initialize with Database object
        
        Args:
            database (Database)
        """
        self.database = database
        self.images_gdf = self.database.images_gdf
        self.detections_gdf = {}
        self.lcc_gdf = {}
        self.name = "vision"

    def reset_vision(self):
        self.detections_gdf = {}
        self.lcc_gdf = {}
    
    def _ingest_offline_labels(self, images_gdf, dataset) -> None:
        """
        Preload detections for the images of interest.
        
        Steps:
            1. Extract the earliest and latest dates from self.images_gdf.
            2. Build a where clause to query detections within that date range.
            3. Load detections from labels (ground-truths precomputed offline).
            4. Filter detections by image_id based on those in self.images_gdf.
            6. Set self.gpkg_dets to the filtered detections.
        
        Returns:
            None
        """
        # Extract date range from self.images_gdf and build WHERE clause  [start_date, end_date)
        where_clause = build_where_date_clause(
            min(images_gdf['date']), 
            max(images_gdf['date']), 
            DATASETS_INFO[dataset].get("labels_date_column", "date")
        )

        # Load vessel detections from the GeoPackage
        gpkg_path = os.path.join(GPKG_FOLDER, DATASETS_INFO[dataset]["labels_file"])
        gpkg_layer = DATASETS_INFO[dataset]["labels_layer"]
        _gpkg_labels = gpd.read_file(gpkg_path, layer=gpkg_layer, where=where_clause)

        # Filter detections by image_id; only keep detections whose image_id is in the images dataset
        valid_image_ids = set(images_gdf['image_id'].unique())
        _gpkg_labels = _gpkg_labels[_gpkg_labels['image_id'].isin(valid_image_ids)]

        # print(set(_gpkg_dets['cat_name'].to_list()))
        return _gpkg_labels

    @agent_tool
    def run_detector(
        self, 
        dataset: str, 
        detector_name: str, 
    ) -> str:
        """
        Run detector on imagery.

        Args:
            dataset (str): The satellite imagery dataset to use.
            detector_name (str): The detector model to use.

        Returns:
            str: A message indicating that detection has completed successfully.
        """
        images_gdf_dict = self.database.images_gdf

        if dataset not in DATASETS_INFO: 
            return DATASET_ERROR_MSG.format(dataset=dataset, datasets=list(DATASETS_INFO.keys()))
        if dataset not in DETECTOR_MODELS: 
            return DETECTOR_DATASET_ERROR_MSG.format(dataset=dataset, dataset_names=list(DETECTOR_MODELS.keys()))

        if detector_name not in DETECTOR_MODELS[dataset]: 
            return DETECTOR_MODEL_ERROR_MSG.format(
                detector_name=detector_name,
                dataset=dataset,
                detector_names=DETECTOR_MODELS[dataset]
            )

        if dataset not in images_gdf_dict: return IMGS_NONE_MSG
        images_gdf = images_gdf_dict[dataset]
        if images_gdf.empty: return IMGS_EMPTY_MSG
        
        # NOTE: We emulate running the Vision model (since we have ground-truths, we preload them)
        _gpkg_labels = self._ingest_offline_labels(images_gdf, dataset)

        # Save the filtered detections to self.detections_gdf
        if dataset not in self.detections_gdf: self.detections_gdf[dataset] = {}
        self.detections_gdf[dataset][detector_name] = _gpkg_labels        

        return f"Detection has successfully completed with model {detector_name} on {len(images_gdf)} {dataset} images."


    @agent_tool
    def run_land_coverage_classifier(
        self, 
        dataset: str, 
        classifier_name: str, 
    ) -> str:
        """
        Run LCC (land-coverage classification) model on imagery.

        Args:
            dataset (str): The satellite imagery dataset to use.
            classifier_name (str): The classification model to use.

        Returns:
            str: A message indicating that classification has completed successfully.
        """
        images_gdf_dict = self.database.images_gdf

        if dataset not in DATASETS_INFO: 
            return DATASET_ERROR_MSG.format(dataset=dataset, datasets=list(DATASETS_INFO.keys()))
        if dataset not in LCC_CLASSIFIER_MODELS: 
            return DETECTOR_DATASET_ERROR_MSG.format(dataset=dataset, dataset_names=list(LCC_CLASSIFIER_MODELS.keys()))

        if classifier_name not in LCC_CLASSIFIER_MODELS[dataset]: 
            return DETECTOR_MODEL_ERROR_MSG.format(
                classifier_name=classifier_name,
                dataset=dataset,
                classifier_names=LCC_CLASSIFIER_MODELS[dataset]
            )

        if dataset not in images_gdf_dict: return IMGS_NONE_MSG
        images_gdf = images_gdf_dict[dataset]
        if images_gdf.empty: return IMGS_EMPTY_MSG
        
        # NOTE: We emulate running the Vision model (since we have ground-truths, we preload them)
        _gpkg_labels = self._ingest_offline_labels(images_gdf, dataset)

        # Save the filtered LCC results to self.lcc_gdf
        if dataset not in self.lcc_gdf: self.lcc_gdf[dataset] = {}
        self.lcc_gdf[dataset][classifier_name] = _gpkg_labels       

        return f"Land-cover classification has successfully completed with model {classifier_name} on {len(images_gdf)} {dataset} images."
