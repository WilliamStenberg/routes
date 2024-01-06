
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import dash
from dash import html, dcc, dash_table
from dash.dash_table.Format import Format, Scheme
from dash.development.base_component import Component
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

from ui import map as uimap
import db as db
import parser as parser
import maps as maps
import model as model

external_stylesheets = []

class Model:
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    def make_activity_graph(self):
        return html.Div(id='graph-output', children=[
                dcc.Graph(id='activity-graph', figure=go.Figure(), style={})
                          
            ])

    def make_map_figure(self):
        fig, style = self.initial_image()
        return html.Div(id='map-output', children=[
                dcc.Graph(id='map-figure', figure=fig, style=style)
            ])

    def make_activity_table(self):
        cols = [
            dict(id='id', name='id'),
            dict(id='title', name='title'),
            dict(id='distance', name='distance', type='numeric', format=Format(precision=1, scheme=Scheme.fixed)),
            dict(id='created_at', name='created_at')
        ]
        records = self.activity_records()
        component = dash_table.DataTable(id='table', data=records, columns=cols)
        return component

    def activity_records(self):
        df = pd.DataFrame([r.properties() for r in  self.routes])
        return df.sort_values(by='created_at', ascending=False).to_dict('records')


    def initial_image(self):
        with db.sess() as sess:
            self.routes: List[db.Route] = db.routes(sess)
            box = model.box_around_latlong_points([(r.start_lat, r.start_long) for r in self.routes])
            _, map = db.ensure_persistent_map(sess, box)
            fig, style = self.load_image(map)
            return fig, style

    def load_image(self, map: model.Map):
        self.map = map
        lats = [r.start_lat for r in self.routes]
        longs = [r.start_long for r in self.routes]
        xs, ys = maps.transform_points(lats, longs, map)
        self.uimap = uimap.Map(map, xs, ys, ['One' for _ in xs])
        
        fig, style = self.map_figure()
        return fig, style

    def update_map_if_zoomed(self, relayout_data, fig, style):
        if self.map and relayout_data:
            if 'xaxis.autorange' in relayout_data:
                print('autorange')
                return self.initial_image(), self.activity_records()
            elif 'autosize' in relayout_data:
                print('autosize')
                return fig, style, self.activity_records()
            elif 'dragmode' in relayout_data:
                return self.initial_image(), self.activity_records()
            if 'xaxis.range' in relayout_data:
                return self.initial_image(), self.activity_records()
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
                    self.routes = db.routes_in_box(self.routes, model.merc_box_to_latlong(new_map.mercator_box))
                    fig, style = self.load_image(new_map)
                    return fig, style, self.activity_records()
        return fig, style, self.activity_records()

    def map_figure(self):
        fig = map_for_activities(self.uimap)
        style = uimap.image_style(self.uimap)
        return fig, style


def map_for_activities(uimap):
    map_hovertext = 'Pace: %{customdata[0]}<br>'
    map_hovertext += 'Distance: %{customdata[1]:.1f}'
    scat = go.Scatter(x=uimap.xs, y=uimap.ys,
                      mode='markers', name='Route', showlegend=True,
                      customdata=uimap.labels,
                      hovertemplate=map_hovertext)

    fix = go.Scatter(
        x=[0, uimap.inner_map.image.width], y=[0, uimap.inner_map.image.height],
        mode='markers', marker_opacity=0, showlegend=False)
    fig = go.Figure(data=[scat, fix], layout=uimap.layout)
    return fig


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
        y=[model.moment_pace(df.loc[pace_info.end_index, 'speed']).seconds/60
           for pace_info in sections],
        mode='markers', name='Past Section Pace',
        showlegend=True,
        customdata=[(pace_info.formatted_pace(), acc_distance)
                    for pace_info, acc_distance in zip(sections, distances)],
        hovertemplate=pace_hovertext)
    return pace_markers



def run():
   model: Model = Model()

   model.app.layout = html.Div(children=[
            html.H1(children='Routes'),

            html.Div(children='All routes recorded'),
            model.make_map_figure(),
            model.make_activity_graph(),
            model.make_activity_table()

        ])


   @model.app.callback(
       [Output('map-figure', 'figure', allow_duplicate=True),
        Output('map-figure', 'style', allow_duplicate=True),
        Output('table','data')],
       Input('map-figure', 'relayoutData'),
       State('map-figure', 'figure'),
       State('map-figure', 'style'),
       prevent_initial_call=True)
   def _(relayout_data=dict(), fig=go.Figure(), style=dict()):
       return model.update_map_if_zoomed(relayout_data, fig, style)

   #@model.app.callback(
   #    [Output('map-figure', 'figure', allow_duplicate=True),
   #     Output('map-figure', 'style', allow_duplicate=True)],
   #    Input('activity-graph', 'relayoutData'),
   #    State('map-figure', 'figure'),
   #    State('map-figure', 'style'),
   #    prevent_initial_call=True,
   #    )
   #def _(relayout_data=dict(), fig=go.Figure(), style=dict()):
   #    return fig, style

   #model.initial_image()
   model.app.run_server(debug=True)

