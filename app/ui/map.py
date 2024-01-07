from typing import List
import plotly.graph_objects as go

import model as model

def map_layout(map) -> go.Layout:
    layout = go.Layout(
        uirevision=None,  # f'{map.image.width}x{map.image.height}',  # Only reset zoom etc on map change
        title='Geodata',
        autosize=False,
        xaxis=dict(range=[0, map.image.width], showgrid=False),
        yaxis=dict(range=[0, map.image.height], showgrid=False),
        #width=map.image.width,
        minreducedwidth=map.image.width,
        #height=map.image.height,
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


class Map:
    def __init__(self, map, xs, ys, labels):
        self.inner_map: model.Map = map
        self.xs: List[float] = xs
        self.ys: List[float] = ys
        self.labels: List[str] = labels
        self.layout = map_layout(map)

    def fig(self, hovertext: str, mode='lines'):
        scat = go.Scatter(x=self.xs, y=self.ys,
                          mode=mode, showlegend=False,
                          customdata=self.labels,
                          hovertemplate=hovertext)

        fix = go.Scatter(
            x=[0, self.inner_map.image.width], y=[0, self.inner_map.image.height],
            mode='markers', marker_opacity=0, showlegend=False)
        fig = go.Figure(data=[scat, fix], layout=self.layout)
        return fig

    def image_style(self):
        image = self.inner_map.image
        height= image.height
        width = image.width
        ratio = width / height
        width_percent = 50
        height_percent = int(width_percent / ratio)
        style = {'width': f'{width_percent}vw', 'height': f'{height_percent}vw'}
        return style

