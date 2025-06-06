import plotly.graph_objects as go
from typing import Optional
from agent_core.modules.toolset import agent_tool

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

class MapTools:
    
    def __init__(self, database, vision, map_style="open-street-map"):
        self.database = database
        self.vision = vision
        self.map_style = map_style
        self.map_fig = self.init_map()
        self.name = "map_tools"

    def init_map(self):
        fig = go.Figure(go.Scattermapbox())
        fig.update_layout(
            mapbox_style=self.map_style,
            hovermode="closest",
            mapbox=dict(
                center=go.layout.mapbox.Center(lat=0, lon=0),
                zoom=1,
                bearing=0,
                pitch=0
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        return fig

    def _add_scatter(self, plot_geom, trace_name, customdata):
        """
        Helper function: adds a scatter trace to the map figure
        """

        lat = plot_geom.y.tolist()
        lon = plot_geom.x.tolist()
        
        scatter_trace = go.Scattermapbox(
            name=trace_name,
            lat=lat,
            lon=lon,
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=8,
                color="red"
            ),
            customdata=customdata,
            hovertemplate="<b>File</b>: %{customdata}<extra></extra>"
        )
        self.map_fig.add_trace(scatter_trace)


    @agent_tool
    def plot_images_scatter_map(
        self,
        dataset: str,
    ) -> str:
        """
        Plots (adds) a scatter trace to the map based on satellite images from memory.

        Args:
            dataset (str): The satellite imagery dataset to use.

        Returns:
            str: A message indicating the number of images plotted on the map.
        """
        images_gdf_dict = self.database.images_gdf
        if dataset not in DATASETS_INFO: 
            return DATASET_ERROR_MSG.format(dataset=dataset, datasets=list(DATASETS_INFO.keys()))
        if dataset not in images_gdf_dict: return IMGS_NONE_MSG
        images_gdf = images_gdf_dict[dataset]
        if images_gdf.empty: return IMGS_EMPTY_MSG
        plot_geom = images_gdf.geometry  # assumed to be POINTs (e.g., from a 'centroid' column)
        customdata = images_gdf["filename"].tolist() if "filename" in images_gdf.columns else ["" for _ in range(len(gdf))]
        trace_name = f"{dataset} images"
        self._add_scatter(plot_geom, trace_name, customdata)
        return f"Images scatter plot added with {len(images_gdf)} {dataset} images."


    @agent_tool
    def plot_detections_catergory_scatter_map(
        self,
        dataset: str,
        detector_name: str,
        category_name: str,
    ) -> str:
        """
        Plots (adds scatter trace) detections for a particular category of satellite dataset.

        Args:
            dataset (str): The satellite dataset from detection results to plot.
            detector_name (str): The name of the detector from detection results to plot.
            category_name (str): The specific category from detection results to plot

        Returns:
            str: A message indicating the number of detections plotted on the map.
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

        # For detections, use centroids for plotting.
        # FIXED!! plot_geom = detections_gdf.geometry.centroid
        # NOTE: detections_gdf is saved with 'geometry' column with POLYGONs in EPSG:4326
        # gpd.GeoDataFrame(gdf, geometry='geom', crs="EPSG:4326")
        # Reproject to a projected CRS (e.g., EPSG:3857) to compute centroids correctly
        gdf_proj = detections_gdf.to_crs(epsg=3857)
        # Compute centroids in the projected CRS
        centroids_proj = gdf_proj.geometry.centroid
        # Reproject the centroids back to EPSG:4326 for plotting
        plot_geom = centroids_proj.to_crs(epsg=4326)

        # Compute bounding box from each polygon.
        bounds_df = detections_gdf.geometry.bounds  # DataFrame with columns: minx, miny, maxx, maxy
        customdata = []
        for idx, row in detections_gdf.iterrows():
            fname = row.get("filename", "")
            # Get bounding box for the current row.
            xmin=row.get("xmin", -1)
            ymin=row.get("ymin", -1)
            xmax=row.get("xmax", -1)
            ymax=row.get("ymax", -1)
            cat_name = row.get("cat_name", "")
            # Build customdata as a list: [filename, minx, miny, maxx, maxy, cat_name]
            customdata.append([fname, xmin, ymin, xmax, ymax, cat_name])
        trace_name = f"{dataset} {detector_name} detections"

        self._add_scatter(plot_geom, trace_name, customdata)
        return f"Scatter plot added with {len(detections_gdf)} {category_name} detections with detector {detector_name} for dataset {dataset}."



    @agent_tool
    def plot_land_cover_class_scatter_map(
        self,
        dataset: str,
        classifier_name: str,
        category_name: str,
    ) -> str:
        """
        Plots (adds scatter trace) land-cover classes for a particular LCC category of satellite dataset.

        Args:
            dataset (str): The satellite dataset from LCC results to plot.
            classifier_name (str): The name of the LCC classifier from LCC results to plot.
            category_name (str): The specific category from LCC results to plot

        Returns:
            str: A message indicating the number of LCC classes plotted on the map.
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

        # For LCC classes, we have the corresponding (actual) image so use its coords
        plot_geom = lcc_gdf.geometry  # assumed to be POINTs (e.g., from a 'centroid' column)

        customdata = lcc_gdf["filename"].tolist() if "filename" in lcc_gdf.columns else ["" for _ in range(len(gdf))]
        trace_name = f"{dataset} {classifier_name} LCC classification results"
        self._add_scatter(plot_geom, trace_name, customdata)
        return f"Scatter plot added with {len(lcc_gdf)} {category_name} LCC classification results for classifer model {classifier_name} and dataset {dataset}."


    @agent_tool
    def reset_map(self) -> str:
        """
        Reset the map figure to its initial state.

        Returns:
            str: A confirmation message indicating that the map has been reset.
        """
        self.map_fig = self.init_map()
        return "Map reset."


    @agent_tool
    def zoom_map(self, lat: float, lon: float, zoom: float = 8) -> str:
        """
        Update the map view to center at the specified latitude and longitude with the given zoom level.

        Args:
            lat (float): The latitude of the new center.
            lon (float): The longitude of the new center.
            zoom (float, optional): The desired zoom level. Defaults to 8.

        Returns:
            str: A confirmation message indicating that the zoom has been updated.
        """
        self.map_fig.update_layout(
            mapbox=dict(
                center=go.layout.mapbox.Center(lat=lat, lon=lon),
                zoom=zoom
            )
        )
        return "Zoom updated."

    def current_map_view(self):
        mapbox = self.map_fig['layout']['mapbox']
        center = mapbox.get("center", {"lat": 0, "lon": 0})
        zoom = mapbox.get("zoom", 1)
        return center["lat"], center["lon"], zoom
