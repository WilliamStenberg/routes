import pandas as pd
import plotly.graph_objects as go

import dash
from dash import html, dcc, dash_table
from dash.dash_table.Format import Format, Scheme
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

import db as db
import parser as parser
import maps as maps
import model as model
from ui import map as uimap

external_stylesheets = []


class Model:
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        self.route = None
        self.uiroutemap = None
        self.uiactmap = None

    def render_part_of_route(self, relayout_data=dict(), fig=go.Figure(), style=dict()):
        if self.uiroutemap:
            if relayout_data and 'xaxis.autorange' in relayout_data.keys():
                start = 0
                stop = len(self.uiroutemap.xs) - 1
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
                fig['data'][0]['x'] = self.uiroutemap.xs[start:stop]  # type: ignore
                fig['data'][0]['y'] = self.uiroutemap.ys[start:stop]  # type: ignore
                fig['data'][0]['customdata'] = self.uiroutemap.labels[start:stop]  # type: ignore
        return fig, style

    def update_output_div(self, route_id: str) -> Optional[Tuple[List[Component], Any, Any]]:
        with db.sess() as sess:
            if (route := sess.get(db.Route, route_id)) is not None:
                self.df = parser.parse_file(route.file_name)
                graph = generate_speed_graph(self.df)
                self.route = route
                return graph, *self.initial_route_image()

    def initial_activity_image(self):
        with db.sess() as sess:
            self.routes: List[db.Route] = db.routes(sess)
            box = model.box_around_latlong_points([(r.start_lat, r.start_long) for r in self.routes])
            _, map = db.ensure_persistent_map(sess, box)
            fig, style = self.load_activity_image(map)
            return fig, style

    def load_activity_image(self, map: model.Map):
        lats = [r.start_lat for r in self.routes]
        longs = [r.start_long for r in self.routes]
        xs, ys = maps.transform_points(lats, longs, map)
        self.uiactmap = uimap.Map(map, xs, ys, ['One' for _ in xs])
        fig, style = self.map_activity_figure()
        return fig, style

    def initial_route_image(self):
        with db.sess() as sess:
            ts = model.Timeseries(self.df)
            _, map = db.ensure_persistent_map(sess, ts.bounding_box())
            fig, style = self.load_route_image(map)
            return fig, style

    def load_route_image(self, map: model.Map):
        xs, ys = maps.transform_geodata(
            self.df, map)

        def pacify(dist: float):
            return model.pace_to_str(model.moment_pace(dist))

        labels = list(zip(self.df['speed'].apply(pacify), self.df['distance']))
        self.uiroutemap = uimap.Map(map, xs, ys, labels)
        fig, style = self.map_figure()
        return fig, style

    def update_map_if_zoomed(self, relayout_data, fig=go.Figure(), style=dict()):
        def f(new_box):
            new_map = maps.transient_map(new_box)
            return self.load_route_image(new_map)
        return with_zoomed_box(relayout_data, fig, style, self.uiroutemap, self.initial_route_image, f)

    def update_act_map_if_zoomed(self, relayout_data, fig=go.Figure(), style=dict()):
        def f(new_box):
            new_map = maps.transient_map(new_box)
            self.routes = db.routes_in_box(self.routes, model.merc_box_to_latlong(new_map.mercator_box))
            return self.load_activity_image(new_map)
        fig, style = with_zoomed_box(relayout_data, fig, style, self.uiactmap, self.initial_activity_image, f)
        return fig, style, self.activity_records()

    def map_figure(self):
        if (m := self.uiroutemap) is not None:
            fig = map_route(m)
            style = m.image_style()
            return fig, style
        return go.Figure(), dict()

    def timestring_to_index(self, time_string: str) -> int:
        ref = self.df['timestamp'].min()
        try:
            timestamp = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            timestamp = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S.%f')
        return int((timestamp.astimezone(ref.tz) - ref).seconds)

    def map_activity_figure(self):
        if (m := self.uiactmap) is not None:
            fig = map_for_activities(m)
            style = m.image_style()
            return fig, style
        return go.Figure(), dict()

    def make_activity_figure(self):
        fig, style = self.initial_activity_image()
        return html.Div(id='act-map-output', children=[
            dcc.Graph(id='map-figure', figure=fig, style=style)
        ])

    def make_activity_table(self):
        cols = [
            dict(id='id', name='id'),
            dict(id='title', name='title'),
            dict(id='distance', name='distance',
                 type='numeric', format=Format(precision=1, scheme=Scheme.fixed)),
            dict(id='avg_pace', name='avg_pace'),
            dict(id='avg_heartrate', name='avg_heartrate',
                 type='numeric', format=Format(precision=0, scheme=Scheme.fixed)),
            dict(id='created_at', name='created_at')
        ]
        records = self.activity_records()
        component = dash_table.DataTable(
            id='table',
            data=records,
            columns=cols,
            row_selectable='single')
        return component

    def activity_records(self):
        df = pd.DataFrame([r.properties() for r in self.routes])
        try:

            return df.sort_values(by='created_at', ascending=False).to_dict('records')
        except Exception:
            return df.to_dict('records')


def with_zoomed_box(relayout_data, fig, style, uimap, initial_func, func):
    if relayout_data:
        if 'xaxis.autorange' in relayout_data:
            print('autorange')
            return initial_func()
        elif 'autosize' in relayout_data:
            print('autosize')
            return fig, style
        elif 'dragmode' in relayout_data:
            return initial_func()
        elif 'xaxis.range' in relayout_data and 'yaxis.range' in relayout_data:
            return initial_func()
        else:
            print(relayout_data)
            min_x = relayout_data['xaxis.range[0]']
            max_x = relayout_data['xaxis.range[1]']
            min_y = relayout_data['yaxis.range[0]']
            max_y = relayout_data['yaxis.range[1]']

            if (m := uimap) is not None:
                im = m.inner_map.image
                old_box = model.merc_box_to_latlong(m.inner_map.mercator_box)
                # Compute ratio of current image to fetch as new (higher res) image
                start_x = min_x / im.width
                end_x = max_x / im.width
                start_y = min_y / im.height
                end_y = max_y / im.height
                x_diff = old_box.east - old_box.west
                y_diff = old_box.north - old_box.south
                new_box = model.BoundingBox(
                    north=old_box.south + end_y * y_diff,
                    east=old_box.west + end_x * x_diff,
                    south=old_box.south + start_y * y_diff,
                    west=old_box.west + start_x * x_diff
                )
                return func(new_box)
    return fig, style


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


def map_route(uimap):
    map_hovertext = 'Pace: %{customdata[0]}<br>'
    map_hovertext += 'Distance: %{customdata[1]:.1f}'
    return uimap.fig(map_hovertext)


def graph_speed_lines(df: pd.DataFrame):
    """ Line graph of speed over time for given dataframe """
    pace_hovertext = 'Pace: %{customdata[1]} per km<br>'
    pace_hovertext += 'Distance: %{customdata[0]:.1f} km'
    pace = df['speed'].apply(model.moment_pace)
    pace_as_float = pace.apply(lambda t: t.seconds / 60)
    speed_lines = go.Scatter(
        x=df['timestamp'], y=pace_as_float,
        mode='lines', name='Pace [min/km]', showlegend=True,
        customdata=[(t['distance'], model.pace_to_str(p)) for (_, t), p in zip(df.iterrows(), pace)],
        hovertemplate=pace_hovertext,
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
        y=[model.moment_pace(df.loc[pace_info.end_index, 'speed']).seconds / 60
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
    sections = parser.section_pace_infos(
        df, kilometer_distance_steps=1, include_total=False)
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


def map_for_activities(uimap):
    map_hovertext = '%{customdata}<br>'
    return uimap.fig(map_hovertext, mode='markers')


def run():
    model: Model = Model()
    model.app.layout = html.Div(children=[
        html.H1(children='Routes'),
        html.Div(children='Interactive running statistics'),
        model.make_activity_figure(),
        model.make_activity_table(),
        *generate_route_dropdown(model.routes,
                                 multiple=False),
        html.Div(id='graph-output', children=[
            dcc.Graph(id='speed-graph', style={'visibility': 'hidden'})
        ]),
        html.Div(id='route-map-output', children=[
            dcc.Graph(id='route-map-figure', figure=go.Figure(),
                      style={'visibility': 'hidden'})
        ]),
    ])

    @model.app.callback(
        [Output('map-figure', 'figure', allow_duplicate=True),
         Output('map-figure', 'style', allow_duplicate=True),
         Output('table', 'data', allow_duplicate=True)],
        Input('map-figure', 'relayoutData'),
        State('map-figure', 'figure'),
        State('map-figure', 'style'),
        prevent_initial_call=True)
    def _(relayout_data=dict(), fig=go.Figure(), style=dict()):
        return model.update_act_map_if_zoomed(relayout_data, fig, style)

    @model.app.callback(
        [Output('graph-output', 'children', allow_duplicate=True),
         Output('route-map-figure', 'figure', allow_duplicate=True),
         Output('route-map-figure', 'style', allow_duplicate=True)],
        Input('table', 'selected_row_ids'),
        prevent_initial_call=True)
    def _(selected_row_ids):
        if selected_row_ids and (route_id := selected_row_ids[0]) is not None:
            return model.update_output_div(route_id)

    @model.app.callback(
        [Output('graph-output', 'children', allow_duplicate=True),
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
