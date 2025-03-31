
DETECTOR_MODELS = {}
DETECTOR_MODELS["xView1"] = ["Swin-L", "YOLO-v8", "CSWin-L"]
DETECTOR_MODELS["FAIR1M"] = ["Swin-L", "YOLO-v8"]

DETECTOR_DATASET_ERROR_MSG = """
Error selecting object detector. {dataset} does not exist in the model-zoo models. 
Please confirm the supported dataset is selected. Available datasets: {dataset_names}.
"""

DETECTOR_MODEL_ERROR_MSG = """
Error selecting object detector. {detector_name} does not exist in the model-zoo models for this dataset. 
Please confirm the correct detector name is selected. Available object detectors for {dataset} dataset: {detector_names}.
"""

LCC_CLASSIFIER_MODELS = {}
LCC_CLASSIFIER_MODELS["BigEarthNet"] = ["ResNet-32", "YOLO-v6"]

LCC_CLASSIFIER_DATASET_ERROR_MSG = """
Error selecting LCC classification model. {dataset} does not exist in the model-zoo models. 
Please confirm the supported dataset is selected. Available datasets: {dataset_names}.
"""

LCC_CLASSIFIER_MODEL_ERROR_MSG = """
Error selecting LCC classification. {classifier_name} does not exist in the model-zoo models for this dataset. 
Please confirm the correct classifier name is selected. Available LCC classifiers for {dataset} dataset: {classifier_names}.
"""
