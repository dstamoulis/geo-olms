from dash import Dash, html, dcc, Input, Output, State, no_update, callback
from dash_extensions import Keyboard
from dash.exceptions import PreventUpdate
import datetime, os, cv2, math
import plotly.express as px
import plotly.graph_objects as go

# Images -- TODO!!! Set this to your location!!
DATASETS_PATH = '/path/to/datasets'

marker_colors_bgr = {    
    'xview1': (0, 0, 255),      # red
    'fair1m': (0, 0, 0),        # black
    'bigearthnet': (0, 255, 255), # yellow
    }


class UI:
    def __init__(self, data_memory, model_zoo, map_ops, detection_obj, data_analytics, agent=None):
        self.data_memory = data_memory
        self.model_zoo = model_zoo
        self.map_ops = map_ops
        self.detection_obj = detection_obj
        self.data_analytics = data_analytics
        self.agent = agent
        self.app = self.create_app()

    def create_app(self):
        app = Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        app.title = "Geo-OLMs"
        app.layout = self.layout()
        self.register_callbacks(app)
        return app

    def layout(self):
        return html.Div([
            # Header with title
            html.Div([
                html.H4("Geo-OLMs", style={'display': 'inline-block', 'align-items': 'center'})
            ], style={'display': 'flex', 'align-items': 'center', 'padding': '0px', 'background-color': '#f5f5f5'}),
            
            # Main content area: Map (left) and Chat (right)
            html.Div([
                # Left column: Map
                html.Div([
                    dcc.Graph(id='map-graph', figure=self.map_ops.map_fig, style={'height': '900px'}, clear_on_unhover=True),
                    dcc.Tooltip(id="graph-tooltip"),
                ], className='eight columns', style={'padding': '0px'}),
                # Right column: Chat window
                html.Div([
                    html.H4("Chat"),
                    Keyboard(
                        dcc.Textarea(
                            id='chat-input',
                            placeholder='Type your message...',
                            style={'width': '100%', 'height': '100px'}
                        ),
                        captureKeys=["Enter"],
                        id='chat-input-keyboard'
                    ),
                    html.Button("Send", id='chat-send', n_clicks=0),
                    html.Div(id='chat-output', children="",
                             style={'margin-top': '10px', 'border': '1px solid #ccc', 'padding': '1px', 'height': '300px', 'overflowY': 'scroll', 'whiteSpace': 'pre-wrap'}),
                    dcc.Markdown(
                        id='chat-output-markdown',
                        dangerously_allow_html=True,
                        children="",
                        style={
                            'margin-top': '10px',
                            'border': '1px solid #ccc',
                            'padding': '5px',
                            'height': '350px',
                            'overflowY': 'scroll',
                            'whiteSpace': 'pre-wrap'
                        }
                    )
                ], className='four columns', style={'padding': '0px'})
            ], className='row'),
            
            # Bottom controls: Dataset dropdown, Detector dropdown, date pickers, and action buttons.
            html.Div([
                html.Div([
                    html.Label("Dataset:"),
                    dcc.Dropdown(
                        id='dataset-dropdown',
                        options=[{'label': key, 'value': key} for key in self.data_memory.mem_images.keys()],
                        value=list(self.data_memory.mem_images.keys())[0]
                    )
                ], style={'display': 'inline-block', 'width': '12%', 'padding': '1px', 'vertical-align': 'middle'}),
                html.Div([
                    html.Label("Detector:"),
                    dcc.Dropdown(
                        id='detector-dropdown',
                        options=[],  # To be populated based on dataset selection.
                        value=None
                    )
                ], style={'display': 'inline-block', 'width': '12%', 'padding': '1px', 'vertical-align': 'middle'}),
                html.Div([
                    html.Label("Category:"),
                    dcc.Dropdown(
                        id='category-dropdown',
                        options=[],  # To be populated based on dataset.
                        value=None
                    )
                ], style={'display': 'inline-block', 'width': '12%', 'padding': '1px', 'vertical-align': 'middle'}),
                html.Div([
                    html.Label("Start Date:"),
                    dcc.DatePickerSingle(
                        id='start-date-picker',
                        date="2017-01-01"
                    )
                ], style={'display': 'inline-block', 'width': '8%', 'padding': '1px', 'vertical-align': 'middle'}),
                html.Div([
                    html.Label("End Date:"),
                    dcc.DatePickerSingle(
                        id='end-date-picker',
                        date="2017-12-31"
                    )
                ], style={'display': 'inline-block', 'width': '8%', 'padding': '1px', 'vertical-align': 'middle'}),
            ], style={'background-color': '#f5f5f5', 'padding': '1px'}),
            html.Div([
                html.Div([
                    html.Button("Plot Images", id='plot-images-button', n_clicks=0),
                    html.Button("Run Detector", id='run-detector-button', n_clicks=0, style={'margin-left': '10px'}),
                    html.Button("Plot Detections", id='plot-detections-button', n_clicks=0, style={'margin-left': '10px'}),
                    html.Button("Reset Map", id='reset-map-button', n_clicks=0, style={'margin-left': '10px'})
                ], style={'display': 'inline-block', 'width': '30%', 'padding': '10px'})], style={'background-color': '#f5f5f5', 'padding': '1px'}),
            # Div for output status messages.
            html.Div(id='output-div', children="No output yet.", style={'margin-top': '10px', 'font-weight': 'bold'})
        ])

    def register_callbacks(self, app):
        # Update detector-dropdown options based on selected dataset.
        @app.callback(
            Output('detector-dropdown', 'options'),
            Input('dataset-dropdown', 'value')
        )
        def update_detector_options(dataset):
            if dataset in self.model_zoo.models:
                options = [{'label': det.detector_name, 'value': det.detector_name} for det in self.model_zoo.models[dataset]]
                return options
            return []
        
        # Update category-dropdown options based on selected dataset.
        @app.callback(
            Output('category-dropdown', 'options'),
            [Input('dataset-dropdown', 'value'),
            Input('run-detector-button', 'n_clicks')]
        )
        def update_category_options(dataset, n_clicks):
            if dataset in self.data_memory.mem_detections:
                unique_cats = self.data_memory.mem_detections[dataset]['cat_name'].unique()
                options = [{'label': cat, 'value': cat} for cat in unique_cats]
                return options
            return []

        # Callback for Plot Images: fetch images from memory and update the map.
        @app.callback(
            [Output('output-div', 'children', allow_duplicate=True),
             Output('map-graph', 'figure', allow_duplicate=True)],
            Input('plot-images-button', 'n_clicks'),
            State('dataset-dropdown', 'value'),
            State('start-date-picker', 'date'),
            State('end-date-picker', 'date'),
            prevent_initial_call=True
        )
        def plot_images(n_clicks, dataset, start_date, end_date):
            # Guard: If button has not been clicked, return without updating.
            if n_clicks is None or n_clicks < 1:
                return no_update, no_update
            
            # For MVP, use fixed bounding box values.
            lat_min, lat_max = 19.0, 20.0
            long_min, long_max = -100.0, -98.0
            filters = {
                'lat_min': lat_min,
                'lat_max': lat_max,
                'long_min': long_min,
                'long_max': long_max,
                'start_date': start_date,
                'end_date': end_date
            }
            load_msg = self.data_memory.fetch_images(dataset, **filters)
            scatter_msg = self.map_ops.add_scatter(dataset, data_type="images",
                                                   sensor_type=None, detector_name=None,
                                                   lat_min=lat_min, lat_max=lat_max,
                                                   long_min=long_min, long_max=long_max,
                                                   start_date=start_date, end_date=end_date)
            updated_fig = self.map_ops.map_fig
            return f"Plot Images: {load_msg} | {scatter_msg}", updated_fig

        # Callback for Run Detector: run detection process.
        @app.callback(
            Output('output-div', 'children', allow_duplicate=True),
            Input('run-detector-button', 'n_clicks'),
            State('dataset-dropdown', 'value'),
            State('detector-dropdown', 'value'),
            State('start-date-picker', 'date'),
            State('end-date-picker', 'date'),
            prevent_initial_call=True
        )
        def run_detector(n_clicks, dataset, detector_name, start_date, end_date):
            if n_clicks is None or n_clicks < 1:
                return no_update
            lat_min, lat_max = 19.0, 20.0
            long_min, long_max = -100.0, -98.0
            message = self.detection_obj.run_detection(
                dataset=dataset,
                sensor_type=None,
                detector_name=detector_name,
                target_category=None,
                lat_min=lat_min,
                lat_max=lat_max,
                long_min=long_min,
                long_max=long_max,
                start_date=start_date,
                end_date=end_date
            )
            return f"Run Detector: {message}"

        # Callback for Plot Detections: update map with detection results.
        @app.callback(
            [Output('output-div', 'children', allow_duplicate=True),
             Output('map-graph', 'figure', allow_duplicate=True)],
            Input('plot-detections-button', 'n_clicks'),
            State('dataset-dropdown', 'value'),
            State('detector-dropdown', 'value'),
            State('category-dropdown', 'value'),
            State('start-date-picker', 'date'),
            State('end-date-picker', 'date'),
            prevent_initial_call=True
        )
        def plot_detections(n_clicks, dataset, detector_name, target_category, start_date, end_date):
            if n_clicks is None or n_clicks < 1:
                return no_update, no_update
            lat_min, lat_max = 19.0, 20.0
            long_min, long_max = -100.0, -98.0
            scatter_msg = self.map_ops.add_scatter(dataset, data_type="detections",
                                                   sensor_type=None, detector_name=detector_name, 
                                                   target_category=target_category,
                                                   lat_min=lat_min, lat_max=lat_max,
                                                   long_min=long_min, long_max=long_max,
                                                   start_date=start_date, end_date=end_date)
            updated_fig = self.map_ops.map_fig
            return f"Plot Detections: {scatter_msg}", updated_fig

        @app.callback(
            [Output('chat-output', 'children'),
             Output('chat-input', 'value'),
             Output('output-div', 'children', allow_duplicate=True),
             Output('map-graph', 'figure', allow_duplicate=True),
             Output('chat-output-markdown', 'children', allow_duplicate=True)],
            [Input('chat-input-keyboard', 'n_keydowns'),
            Input('chat-send', 'n_clicks')],
            [State('chat-input', 'value'),
            State('chat-output', 'children'),
            State("chat-input-keyboard", "keydown")],
            prevent_initial_call=True
        )
        def submit_message(n_keydowns, n_clicks, message, current_chat, key_event):
            # We assume that if the Send button is clicked or if the Keyboard reports at least one keydown, 
            # and there is text, then we process the message.
            print(key_event, n_clicks)
            if key_event and key_event['shiftKey']:
                raise PreventUpdate
            if not message:
                raise PreventUpdate
            if (n_clicks is None or n_clicks < 1) and key_event is None:
                # return current_chat
                raise PreventUpdate

            self.data_analytics.new_fig = False # HACK!
            # Here, call your agent's run_query; using a dummy agent for demonstration.
            response = self.agent.run_query(message, ui_mode=True)
            updated_chat = (current_chat or "") + f"\n\n----------\n\nUser: {message}\nAgent: {response}"
            updated_fig = self.map_ops.map_fig

            # markdown_str = """
            # ![My Local Image](assets/myimage.png)
            # """
            if self.data_analytics.new_fig:
                markdown_str = f'<img src="/assets/{self.data_analytics.generated_figure_name}" style="max-width:80%; height:auto;"/>'
            else:
                markdown_str = no_update

            # markdown_str = """
            # <img src="/assets/myimage.png" width="200"/>
            # """
            # Plot temporal distribution of xview1 images for 2017!
            return updated_chat, "", "Agent replied", updated_fig, markdown_str

        @app.callback(
            Output("graph-tooltip", "show"),
            Output("graph-tooltip", "bbox"),
            Output("graph-tooltip", "children"),
            Input("map-graph", "hoverData")
        )
        def display_hover(hoverData):
            # If there's no hover data, hide the tooltip.
            if hoverData is None:
                return False, None, None

            # Extract the first point's data.
            point = hoverData['points'][0]
            customdata = point.get("customdata", None)
            if not customdata:
                return False, None, None

            # Assume that if customdata is a string, we are in IMAGE mode;
            # if it is a list (or a string that encodes extra info), we are in DETECTION mode.
            # You can modify this logic to suit your actual data format.
            if isinstance(customdata, list) and len(customdata) >= 6:
                # DETECTION mode: we expect customdata to contain additional items:
                # For example: [filename, ..., box_category, x_min, y_min, x_max, y_max]
                file_path = os.path.join(DATASETS_PATH, customdata[0])
                # Also, detection data may be passed to allow drawing bounding boxes.
                try:
                    box_category = customdata[5]
                    x_min, y_min, x_max, y_max = map(float, customdata[1:5])
                except Exception as e:
                    box_category = None
            else:
                # IMAGE mode: customdata is a simple filename.
                file_path = os.path.join(DATASETS_PATH, customdata)
                box_category = None

            # If file does not exist, use a default blank image.
            if not os.path.isfile(file_path):
                # You can define a constant path to a blank placeholder.
                file_path = os.path.join(DATASETS_PATH, "blobs", "blank.png")
                if not os.path.isfile(file_path):
                    return False, None, None

            # Load the image using OpenCV.
            im = cv2.imread(file_path)
            if im is None:
                return False, None, None
            try:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            except Exception:
                return False, None, None

            init_h, init_w, _ = im.shape
            target_size = 512
            longest_side = max(init_h, init_w)
            downsize_factor = longest_side / target_size if longest_side > target_size else 1

            # Optionally, crop or simply use the full image.
            zoom = 1  # Currently unused.
            zoom_h, zoom_w = round(init_h / zoom), round(init_w / zoom)
            im = im[0:zoom_h, 0:zoom_w]

            # If we're in detection mode and have a bounding box, draw it.
            if box_category is not None:
                # Use a default color if the category is not in our marker_colors_bgr.
                _color = marker_colors_bgr.get(box_category, (0, 255, 0))
                # Convert coordinates to integers.
                try:
                    x_min_int, y_min_int, x_max_int, y_max_int = map(int, [x_min, y_min, x_max, y_max])
                    cv2.rectangle(im, (x_min_int, y_min_int), (x_max_int, y_max_int), _color, thickness=math.ceil(3/zoom))
                    # Optionally, add text.
                    font_size = math.ceil(2/zoom)
                    font_thickness = math.ceil(2/zoom)
                    cv2.putText(im, box_category, (x_min_int, y_max_int-10),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, _color, font_thickness)
                except Exception as e:
                    pass  # If drawing fails, ignore.
            
            new_width = int(init_w / downsize_factor)
            new_height = int(init_h / downsize_factor)

            # Create a Plotly Express figure from the image.
            fig2 = px.imshow(im)
            fig2.update_layout(height=new_height, width=new_width, xaxis_visible=False, yaxis_visible=False)
            fig2.update_layout(margin={"r":0, "t":0, "l":0, "b":0})

            # Use the provided bounding box from hoverData for positioning the tooltip.
            hover_location = point.get("bbox", None)
            # Use the basename for display.
            _filename = os.path.basename(file_path)

            # Build tooltip content.
            tooltip_children = html.Div([
                html.P(f"Asset: xview1 || Location: {point.get('lat', 'N/A'):.3f} x {point.get('lon', 'N/A'):.3f}"),
                html.P(f"Asset: {_filename}"),
                dcc.Graph(id="graph-basic-2", figure=fig2, clear_on_unhover=True)
            ])

            return True, hover_location, tooltip_children

        # Callback for Reset Map: reset the map and update the map-graph figure.
        @app.callback(
            [Output('output-div', 'children', allow_duplicate=True),
             Output('map-graph', 'figure', allow_duplicate=True)],
            Input('reset-map-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def reset_map_callback(n_clicks):
            if n_clicks is None or n_clicks < 1:
                return no_update, no_update
            reset_message = self.map_ops.reset_map()
            updated_fig = self.map_ops.map_fig
            return f"Map reset: {reset_message}", updated_fig

    def __call__(self):
        # self.app.run_server()
        self.app.run_server(debug=True, dev_tools_hot_reload = False)
        # self.app.run_server(debug=True)
