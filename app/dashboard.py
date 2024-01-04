from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import dash
from dash import html
from dash import dcc
from dash.development.base_component import Component
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

import db as db
import parser as parser
import maps as maps
import model as model

external_stylesheets = []

@dataclass
class RouteData:
    xs: List[float]
    ys: List[float]
    labels: Any

class Model:
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        self.route = None
        self.map: model.Map
        self.layout = None
        self.route_data = RouteData(xs=[], ys=[], labels=None)
        with db.sess() as sess:
            self.map = None  # type: ignore
            self.routes: List[db.Route] = db.routes(sess)
            self.set_layout()

    def render_part_of_route(self, relayout_data=dict(), fig=go.Figure(), style=dict()):
        if self.layout:
            if relayout_data and 'xaxis.autorange' in relayout_data.keys():
                start = 0
                stop = len(self.route_data.xs) - 1
            elif relayout_data and 'xaxis.range[0]' in relayout_data.keys():
                start = self.timestring_to_index(relayout_data['xaxis.range[0]'])
                stop = self.timestring_to_index(relayout_data['xaxis.range[1]'])
            elif relayout_data and 'xaxis.range' in relayout_data.keys():
                start, stop = relayout_data['xaxis.range']
                start = self.timestring_to_index(start)
                stop = self.timestring_to_index(stop)
            else:
                return fig, style
            if len(fig['data']) > 0:  # type: ignore
                fig['data'][0]['x'] = self.route_data.xs[start:stop]  # type: ignore
                fig['data'][0]['y'] = self.route_data.ys[start:stop]  # type: ignore
                fig['data'][0]['customdata'] = self.route_data.labels[start:stop]  # type: ignore
        return fig, style

    def update_output_div(self, route_id: str) -> Optional[Tuple[List[Component], Any, Any]]:
        with db.sess() as sess:
            if (route := sess.get(db.Route, route_id)) is not None:
                self.df = parser.parse_file(route.file_name)
                graph = generate_speed_graph(self.df)
                self.route = route
                return graph, *self.initial_route_image()

    def initial_route_image(self):
        with db.sess() as sess:
            _, map = db.ensure_persistent_map(sess, model.Timeseries(self.df))
            fig, style = self.load_route_image(map)
            return fig, style

    def load_route_image(self, map: model.Map):
        self.map = map
        xs, ys = maps.transform_geodata(
            self.df, map)
        self.route_data = RouteData(xs=xs, ys=ys, labels=list(zip(self.df['speed'], self.df['distance'])))
        self.layout = map_layout(map)
        fig, style = self.map_figure()
        return fig, style

    def update_map_if_zoomed(self, relayout_data, fig, style):
        if self.layout and relayout_data:
            if 'xaxis.autorange' in relayout_data:
                print('autorange')
                return self.initial_route_image()
            elif 'autosize' in relayout_data:
                print('autosize')
                return fig, style
            elif 'dragmode' in relayout_data:
                return self.initial_route_image()
            else:
                print(relayout_data)
                min_x = relayout_data['xaxis.range[0]']
                max_x = relayout_data['xaxis.range[1]']
                min_y = relayout_data['yaxis.range[0]']
                max_y = relayout_data['yaxis.range[1]']
                old_box = model.merc_box_to_latlong(self.map.mercator_box)
                # Compute ratio of current image to fetch as new (higher res) image
                start_x = min_x / self.map.image.width
                end_x = max_x / self.map.image.width
                zoom_x = end_x - start_x
                start_y = min_y / self.map.image.height
                end_y = max_y / self.map.image.height
                zoom_y = end_y - start_y
                if zoom_x < 0.5 or zoom_y < 0.5:
                    x_diff = old_box.east - old_box.west
                    y_diff = old_box.north - old_box.south
                    new_box = model.BoundingBox(
                            north = old_box.south + end_y * y_diff,
                            east = old_box.west + end_x * x_diff,
                            south = old_box.south + start_y * y_diff,
                            west = old_box.west + start_x * x_diff
                    )
                    new_map = maps.transient_map(new_box)
                    return self.load_route_image(new_map)

        return fig, style

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

    def map_figure(self):
        fig = map_route(self.route_data, self.map, self.layout)
        height= self.map.image.height
        width = self.map.image.width
        ratio = width / height
        width_percent = 80
        height_percent = width_percent / ratio
        style = {'width': f'{width_percent}vw', 'height': f'{height_percent}vw'}
        return fig, style

    def timestring_to_index(self, time_string: str) -> int:
        ref = self.df['timestamp'].min()
        try:
            timestamp = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            timestamp = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S.%f')
        return int((timestamp.astimezone(ref.tz) - ref).seconds)


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

def map_route(route_data_in_view: RouteData, map: model.Map, layout):
    map_hovertext = 'Speed: %{customdata[0]:.3f}m/s<br>'
    map_hovertext += 'Distance: %{customdata[1]:.1f}'
    scat = go.Scatter(x=route_data_in_view.xs, y=route_data_in_view.ys,
                      mode='lines', name='Route', showlegend=True,
                      customdata=route_data_in_view.labels,
                      hovertemplate=map_hovertext)

    fix = go.Scatter(
        x=[0, map.image.width], y=[0, map.image.height],
        mode='markers', marker_opacity=0, showlegend=False)
    fig = go.Figure(data=[scat, fix], layout=layout)
    return fig



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


def map_layout(map) -> go.Layout:
    layout = go.Layout(
        uirevision=None,  # f'{map.image.width}x{map.image.height}',  # Only reset zoom etc on map change
        title='Geodata',
        autosize=False,
        xaxis=dict(range=[0, map.image.width]),
        yaxis=dict(range=[0, map.image.height]),
        images=[go.layout.Image(
            source=map.image,
            xref='x',
            yref='y',
            x=0,
            y=map.image.height,
            sizex=map.image.width,
            sizey=map.image.height,
            sizing='stretch',
            opacity=1,
            layer='below')])
    return layout


def run():
   model: Model = Model()
   @model.app.callback(
       [Output('graph-output', 'children'),
       Output('route-map-figure', 'figure', allow_duplicate=True),
       Output('route-map-figure', 'style', allow_duplicate=True)],
       Input('route-dropdown', 'value'),
       prevent_initial_call=True
   )
   def _(route_id: str):
       return model.update_output_div(route_id)

   @model.app.callback(
       [Output('route-map-figure', 'figure', allow_duplicate=True),
        Output('route-map-figure', 'style', allow_duplicate=True)],
       Input('route-map-figure', 'relayoutData'),
       State('route-map-figure', 'figure'),
       State('route-map-figure', 'style'),
       prevent_initial_call=True)
   def _(relayout_data=dict(), fig=go.Figure(), style=dict()):
       return model.update_map_if_zoomed(relayout_data, fig, style)

   @model.app.callback(
       [Output('route-map-figure', 'figure', allow_duplicate=True),
        Output('route-map-figure', 'style', allow_duplicate=True)],
       Input('speed-graph', 'relayoutData'),
       State('route-map-figure', 'figure'),
       State('route-map-figure', 'style'),
       prevent_initial_call=True)
   def _(relayout_data=dict(), fig=go.Figure(), style=dict()):
       return model.render_part_of_route(relayout_data, fig, style)

   model.app.run_server(debug=True)

