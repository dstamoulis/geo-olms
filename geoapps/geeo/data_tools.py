import os
import pandas as pd
import datetime
import geopandas as gpd
from typing import Optional

from geoplatform.utils import build_where_date_clause
from geoapps.geeo.constants import (
    DETS_NONE_MSG, DETS_EMPTY_MSG, DATASET_ERROR_MSG,
    DATASETS_INFO, GPKG_FOLDER,  IMGS_NONE_MSG, IMGS_EMPTY_MSG,
    DATASET_CATEGORY_ERROR_MSG, DATASETS_CATEGORIES,
    LCC_NONE_MSG, LCC_EMPTY_MSG, 
)
from geoapps.geeo.model_cards import (
    DETECTOR_MODELS, DETECTOR_MODEL_ERROR_MSG, DETECTOR_DATASET_ERROR_MSG,
    LCC_CLASSIFIER_MODELS, LCC_CLASSIFIER_DATASET_ERROR_MSG, LCC_CLASSIFIER_MODEL_ERROR_MSG
)
from agent_core.modules.toolset import agent_tool


class DataTools:
    def __init__(self, database, vision):
        self.database = database
        self.vision = vision
        self.name = "data_tools"

    def dynamic_tool(self, task: str) -> str:
        # TODO!!! code_to_run = llm_code_agent(task) # Have an LLM suggesting df ops code to complete the task
        return "Dynamic tool (code) execution has successfully completed"

    @agent_tool
    def count_filtered_images(
        self,
        dataset: str,
    ) -> str:
        """
        Counts the number of images (already filtered for particular user criteria) for a specified dataset

        Args:
            dataset (str): The name of the satellite imagery dataset.

        Returns:
            str: A message indicating the number of filtered images.
        """
        images_gdf_dict = self.database.images_gdf
        if dataset not in DATASETS_INFO: 
            return DATASET_ERROR_MSG.format(dataset=dataset, datasets=list(DATASETS_INFO.keys()))
        if dataset not in images_gdf_dict: return IMGS_NONE_MSG
        images_gdf = images_gdf_dict[dataset]
        if images_gdf.empty: return IMGS_EMPTY_MSG
        return f"There are {len(images_gdf)} {dataset} images for particular user criteria."


    @agent_tool
    def count_detections_by_category(
        self,
        dataset: str,
        detector_name: str,
        category_name: str,
    ) -> str:
        """
        Counts the number of detections for a particular category in a specified satellite dataset.

        Args:
            dataset (str): The name of the satellite dataset from which detections were derived.
            detector_name (str): The name of the detection model used.
            category_name (str): The specific object category class to count.

        Returns:
            str: A message indicating the number of detections found in the selected category.
        """
        detections_gdf_dict = self.vision.detections_gdf
        if dataset not in DATASETS_INFO: 
            return DATASET_ERROR_MSG.format(dataset=dataset, datasets=list(DATASETS_INFO.keys()))
        if dataset not in detections_gdf_dict: return DETS_NONE_MSG
        if detector_name not in DETECTOR_MODELS[dataset]: 
            return DETECTOR_MODEL_ERROR_MSG.format(
                detector_name=detector_name,
                dataset=dataset,
                detector_names=DETECTOR_MODELS[dataset]
            )
        if detector_name not in detections_gdf_dict[dataset]: return DETS_NONE_MSG

        detections_gdf = detections_gdf_dict[dataset][detector_name]
        if detections_gdf.empty: return DETS_EMPTY_MSG

        if category_name not in DATASETS_CATEGORIES[dataset]: 
            return DATASET_CATEGORY_ERROR_MSG.format(category_name=category_name, category_names=DATASETS_CATEGORIES[dataset])
        detections_gdf = detections_gdf[detections_gdf['cat_name'] == category_name]
        if detections_gdf.empty: return DETS_EMPTY_MSG

        return f"There are {len(detections_gdf)} {category_name} detections with detector {detector_name} for dataset {dataset}."


    @agent_tool
    def count_lcc_classification_results_by_category(
        self,
        dataset: str,
        classifier_name: str,
        category_name: str,
    ) -> str:
        """
        Count LCC classification results for a particular LCC category in a specified satellite dataset.

        Args:
            dataset (str): The satellite dataset from LCC results were derived.
            classifier_name (str): The name of the LCC classifier used.
            category_name (str): The specific category from LCC results to count.

        Returns:
            str: A message indicating the number of LCC classification results found in the selected category.
        """
        lcc_gdf_dict = self.vision.lcc_gdf
        if dataset not in DATASETS_INFO: 
            return DATASET_ERROR_MSG.format(dataset=dataset, datasets=list(DATASETS_INFO.keys()))
        if dataset not in lcc_gdf_dict: return LCC_NONE_MSG
        if classifier_name not in LCC_CLASSIFIER_MODELS[dataset]: 
            return DETECTOR_MODEL_ERROR_MSG.format(
                classifier_name=classifier_name,
                dataset=dataset,
                classifier_names=LCC_CLASSIFIER_MODELS[dataset]
            )

        if classifier_name not in lcc_gdf_dict[dataset]: return LCC_NONE_MSG
        lcc_gdf = lcc_gdf_dict[dataset][classifier_name]
        if lcc_gdf.empty: return LCC_EMPTY_MSG

        if category_name not in DATASETS_CATEGORIES[dataset]: 
            return DATASET_CATEGORY_ERROR_MSG.format(category_name=category_name, category_names=DATASETS_CATEGORIES[dataset])
        lcc_gdf = lcc_gdf[lcc_gdf['cat_name'] == category_name]
        if lcc_gdf.empty: return LCC_EMPTY_MSG

        return f"There are {len(lcc_gdf)} {category_name} LCC classification results for classifer model {classifier_name} and dataset {dataset}."
