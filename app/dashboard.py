from typing import List, Dict
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.development.base_component import Component
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output
from PIL import Image

from db import Route
from parser import parse_file, section_pace_infos, SectionPaceInfo
from maps import transform_geodata

external_stylesheets = []

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def generate_route_dropdown(routes: List[Route],
                            multiple: bool) -> List[Component]:
    """ Route object selector, optionally allowing multiple objects """
    def make_option(route: Route) -> Dict:
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


def graph_speed_lines(df: pd.DataFrame) -> go.Scatter:
    """ Line graph of speed over time for given dataframe """
    speed_hovertext = '%{customdata[0]}<br>Distance: %{customdata[1]:.3f}km'
    speed_lines = go.Scatter(
        x=df['timestamp'], y=df['speed'],
        mode='lines', name='Speed [m/s]', showlegend=True,
        customdata=[(t['timestamp'].strftime('%H:%M:%S'),
                     t['distance']) for _, t in df.iterrows()],
        hovertemplate=speed_hovertext)
    return speed_lines


def graph_pace_markers(df: pd.DataFrame, sections: List[SectionPaceInfo]
                       ) -> go.Scatter:
    """ Scatter graph of pace markers for given dataframe and sections list """
    distances = [0]  # Accumulated distance at each marker
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
    speed_lines = graph_speed_lines(df)
    sections = section_pace_infos(df, kilometer_distance_steps=1,
                                  include_total=False)
    pace_markers = graph_pace_markers(df, sections)
    fig = go.Figure(data=[speed_lines, pace_markers])
    fig.update_xaxes(
        tickformat='%H:%M:%S')
    graph = dcc.Graph(
        id='speed-graph',
        figure=fig
    )
    return [graph]


def generate_route_map_figure(df: pd.DataFrame,
                              route: Route) -> List[Component]:
    route_map = route.map_ref
    image_shape = (route_map.image_height, route_map.image_width)
    xs, ys = transform_geodata(
        df, image_shape, route_map.image_mercator_extent_dict)
    map_hovertext = 'Speed: %{customdata[0]:.3f}m/s<br>'
    map_hovertext += 'Distance: %{customdata[1]:.1f}'
    scat = go.Scatter(x=xs, y=ys,
                      mode='lines', name='Route', showlegend=True,
                      customdata=list(zip(df['speed'], df['distance'])),
                      hovertemplate=map_hovertext)

    # TODO replace image path with FileField and load in file here
    image = Image.open(route_map.image_path)
    layout = go.Layout(
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
    fig = go.Figure(data=[scat], layout=layout)
    ratio = route_map.image_width / route_map.image_height
    width_percent = 80
    height_percent = width_percent / ratio
    graph = dcc.Graph(
        id='route-map-figure',
        figure=fig,
        style={'width': f'{width_percent}vw', 'height': f'{height_percent}vw'}
    )
    return [graph]


app.layout = html.Div(children=[
    html.H1(children='Routes'),

    html.Div(children='Interactive running statistics'),

    *generate_route_dropdown(Route.objects.order_by('-datetime'),
                             multiple=False),
    html.Div(id='graph-output'),
])


@app.callback(
    Output(component_id='graph-output', component_property='children'),
    Input(component_id='route-dropdown', component_property='value')
)
def update_output_div(route_id: str) -> List[Component]:
    if not route_id:
        return None
    route = Route.objects.get(id=route_id)
    df = parse_file(route.file_name)
    graph = generate_speed_graph(df)
    map_figure = generate_route_map_figure(df, route)
    return [*graph, *map_figure]
