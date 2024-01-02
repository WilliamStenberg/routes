from typing import List, Dict, Optional
import dash
from dash import html
from dash import dcc
from dash.development.base_component import Component
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
from datetime import datetime

import db as db
import parser as parser
import maps as maps

external_stylesheets = []
current_route_data = ([], [], None)
current_time_function = lambda x: x
current_route = None
current_layout = None

class Model:
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        with db.sess() as sess:
            self.routes: List[db.Route] = db.routes(sess)
            self.set_layout()

        @self.app.callback(
            Output(component_id='graph-output', component_property='children'),
            Input(component_id='route-dropdown', component_property='value'),
            prevent_initial_call=True
        )
        def update_output_div(route_id: str) -> Optional[List[Component]]:
            with db.sess() as sess:
                if (route := sess.get(db.Route, route_id)) is not None:
                    df = parser.parse_file(route.file_name)
                    graph = generate_speed_graph(df)
                    set_global_route_state(df, route)
                    return graph


        @self.app.callback(
            [Output('route-map-figure', 'figure'),
             Output('route-map-figure', 'style')],
            Input('route-map-figure', 'relayoutData'),
            State('route-map-figure', 'figure'),
            State('route-map-figure', 'style'),
            prevent_initial_call=True)
        def update_map_if_zoomed(relayoutData=dict(), fig=go.Figure(), style=dict()):
            if current_layout:
                # TODO: If the current view is zoomed in more than x % of the current background image,
                # then load a transient image based on the coordinates
                pass
            return fig, style

        @self.app.callback(
            [Output('route-map-figure', 'figure', allow_duplicate=True),
             Output('route-map-figure', 'style', allow_duplicate=True)],
            Input('speed-graph', 'relayoutData'),
            State('route-map-figure', 'figure'),
            State('route-map-figure', 'style'),
            prevent_initial_call=True)
        def boummary(relayoutData=dict(), fig=go.Figure(), style=dict()):
            if current_layout:
                xs, ys, customdata = current_route_data

                if relayoutData and 'xaxis.autorange' in relayoutData.keys():
                    start = 0
                    stop = len(xs) - 1
                elif relayoutData and 'xaxis.range[0]' in relayoutData.keys():
                    start = current_time_function(relayoutData['xaxis.range[0]'])
                    stop = current_time_function(relayoutData['xaxis.range[1]'])
                elif relayoutData and 'xaxis.range' in relayoutData.keys():
                    start, stop = relayoutData['xaxis.range']
                    start = current_time_function(start)
                    stop = current_time_function(stop)
                else:
                    return generate_map_figure(current_route, current_layout,
                                               xs, ys, customdata)
                if len(fig['data']) > 0:  # type: ignore
                    fig['data'][0]['x'] = xs[start:stop]  # type: ignore
                    fig['data'][0]['y'] = ys[start:stop]  # type: ignore
                    fig['data'][0]['customdata'] = customdata[start:stop]  # type: ignore
            return fig, style

    def run_server(self):
        self.app.run_server(debug=True)

    def set_layout(self):
        self.app.layout = html.Div(children=[
            html.H1(children='Routes'),

            html.Div(children='Interactive running statistics'),
            *generate_route_dropdown(self.routes,
                                     multiple=False),
            html.Div(id='graph-output', children=[
                dcc.Graph(id='speed-graph', style={'visibility': 'hidden'})
            ]),
            html.Div(id='map-output', children=[
                dcc.Graph(id='route-map-figure', figure=go.Figure(),
                          style={'visibility': 'hidden'})
            ]),
        ])


def generate_route_dropdown(routes: List[db.Route],
                            multiple: bool) -> List[Component]:
    """ Route object selector, optionally allowing multiple objects """
    def make_option(route: db.Route) -> Dict:
        return {'label': route.title, 'value': str(route.id)}
    dropdown_id = 'route-dropdown'
    if multiple:
        dropdown_id += '-multiple'
    return [
        html.Label('Choose which route to display'),
        dcc.Dropdown(
            id=dropdown_id,
            options=[make_option(r) for r in routes],
            multi=multiple
        )
    ]


def graph_speed_lines(df: pd.DataFrame):
    """ Line graph of speed over time for given dataframe """
    speed_hovertext = 'Distance: %{customdata[0]:.3f}km'
    speed_lines = go.Scatter(
        x=df['timestamp'], y=df['speed'],
        mode='lines', name='Speed [m/s]', showlegend=True,
        customdata=[(t['distance'],) for _, t in df.iterrows()],
        hovertemplate=speed_hovertext,
        yaxis='y'
    )
    yaxis_dict = dict(
        anchor='x',
        autorange=True,
        domain=[0, 0.7],
        linecolor='#673ab7',
        mirror=True,
        showline=True,
        side='left',
        tickfont={'color': '#673ab7'},
        tickmode='auto',
        ticks='',
        titlefont={'color': '#673ab7'},
        type='linear',
        zeroline=False)
    return speed_lines, yaxis_dict


def graph_heartrate_lines(df: pd.DataFrame):
    """ Line graph of heartrate over time for given dataframe """
    heartrate_lines = go.Scatter(
        x=df['timestamp'], y=df['heart_rate'],
        mode='lines', name='Heartrate [bpm]', showlegend=True,
        yaxis='y2',
        line=dict(color='#e91e63')
    )
    yaxis_dict = dict(
        anchor='y',
        autorange=True,
        domain=[0.7, 1],
        linecolor='#e91e63',
        mirror=False,
        showline=True,
        side='left',
        tickfont={'color': '#e91e63'},
        tickmode='auto',
        ticks='',
        titlefont={'color': '#e91e63'},
        type='linear',
        zeroline=False)
    return heartrate_lines, yaxis_dict


def graph_pace_markers(df: pd.DataFrame, sections: List[parser.SectionPaceInfo]
                       ) -> go.Scatter:
    """ Scatter graph of pace markers for given dataframe and sections list """
    distances: List[float] = [0]  # Accumulated distance at each marker
    for pace_info in sections:
        distances.append(distances[-1] + pace_info.distance)
    # Removing initial accumulator zero
    distances = distances[1:]

    pace_hovertext = 'Pace: %{customdata[0]} per km<br>'
    pace_hovertext += 'Distance: %{customdata[1]:.1f} km'
    pace_markers = go.Scatter(
        x=[df.loc[pace_info.end_index, 'timestamp']
           for pace_info in sections],
        y=[df.loc[pace_info.end_index, 'speed']
           for pace_info in sections],
        mode='markers', name='Past Section Pace',
        showlegend=True,
        customdata=[(pace_info.formatted_pace(), acc_distance)
                    for pace_info, acc_distance in zip(sections, distances)],
        hovertemplate=pace_hovertext)
    return pace_markers


def generate_speed_graph(df: pd.DataFrame) -> List[Component]:
    """
    Creates a plot of speed over time with markers indicating
    average kilometer pace.
    """
    fig = go.Figure()
    xrange = [df['timestamp'].iloc[0], df['timestamp'].iloc[len(df) - 1]]
    xaxis_dict = dict(
        autorange=True,
        range=xrange,
        rangeslider=dict(autorange=True, range=xrange),
        type='date')

    speed_lines, speed_ydict = graph_speed_lines(df)
    sections = parser.section_pace_infos(df, kilometer_distance_steps=1,
                                  include_total=False)
    pace_markers = graph_pace_markers(df, sections)
    fig.add_trace(speed_lines)
    fig.add_trace(pace_markers)
    fig.update_layout(
        xaxis=xaxis_dict,
        yaxis=speed_ydict,
        hovermode='x',
    )
    if 'heart_rate' in df.columns:
        heartrate_lines, heartrate_ydict = graph_heartrate_lines(df)
        fig.add_trace(heartrate_lines)
        fig.update_layout(yaxis2=heartrate_ydict)

    fig.update_xaxes(
        tickformat='%H:%M:%S')
    graph = dcc.Graph(
        id='speed-graph',
        figure=fig
    )
    return [graph]


def map_layout(route_map) -> go.Layout:
    # TODO replace image path with FileField and load in file here
    image = Image.open(route_map.image_path)
    layout = go.Layout(
        uirevision=route_map.image_path,  # Only reset zoom etc on map change
        title='Geodata',
        images=[go.layout.Image(
            source=image,
            xref='x',
            yref='y',
            x=0,
            y=route_map.image_height,
            sizex=route_map.image_width,
            sizey=route_map.image_height,
            sizing='stretch',
            opacity=1,
            layer='below')])
    return layout


def generate_map_figure(route, layout, xs, ys, customdata):
    route_map = route.map
    map_hovertext = 'Speed: %{customdata[0]:.3f}m/s<br>'
    map_hovertext += 'Distance: %{customdata[1]:.1f}'
    scat = go.Scatter(x=xs, y=ys,
                      mode='lines', name='Route', showlegend=True,
                      customdata=customdata,
                      hovertemplate=map_hovertext)

    fix = go.Scatter(
        x=[0, route_map.image_width], y=[0, route_map.image_height],
        mode='markers', marker_opacity=0, showlegend=False)
    fig = go.Figure(data=[scat, fix], layout=layout)
    ratio = route_map.image_width / route_map.image_height
    width_percent = 80
    height_percent = width_percent / ratio
    style = {'width': f'{width_percent}vw', 'height': f'{height_percent}vw'}
    return fig, style


def set_global_route_state(df: pd.DataFrame, route: db.Route) -> None:
    global current_route, current_route_data, \
        current_layout, current_time_function
    route_map = route.map
    image_shape = (route_map.image_height, route_map.image_width)
    xs, ys = maps.transform_geodata(
        df, image_shape, route.map.mercator_bounding_box.bounding_box())

    def timestring_to_index(time_string: str) -> int:
        ref = df['timestamp'].min()
        try:
            timestamp = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            timestamp = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S.%f')
        return int((timestamp.astimezone(ref.tz) - ref).seconds)
    current_time_function = timestring_to_index
    current_route_data = (xs, ys, list(zip(df['speed'], df['distance'])))
    current_layout = map_layout(route_map)
    current_route = route
