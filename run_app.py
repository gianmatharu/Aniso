import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime

import pandas 

import cv2
import numpy as np
from PIL import Image

from aniso.tensor import DiffusionTensor2D
from aniso.derivatives import apply_isotropic_smoothing
from aniso.dash_reusable_components import b64_to_numpy, b64_to_pil

import pandas as pd
import base64



logo = 'test_images/aniso.png'
encoded_logo = base64.b64encode(open(logo, 'rb').read())

image = np.asarray(Image.open('test_images/crabnebula.jpg'))
dtensor = DiffusionTensor2D.fromimage(image)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

markdown_intro = '''
### An anisotropic image smoother

Adjust the smoothing parameter to increase (or decrease) smoothing. 
The leftmost image is the original image, the middle is the image after
isotropic smoothing and the rightmost image is the anisotropically smoothed image.
'''

markdown_instr = '''
Go head, try and upload your own image!
'''

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Img(src='data:image/png;base64,{}'.format(encoded_logo.decode()), style={'textAlign': 'center'}),
    dcc.Markdown(children=markdown_intro),
    dcc.Slider(
        id='alpha-slider',
        min=1,
        max=100,
        value=10,
        marks={str(alpha): str(alpha) for alpha in np.arange(0, 110, 10)},
        step=5
    ),
    dcc.Graph(id='graph'),
    dcc.Markdown(children=markdown_instr),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    dcc.Graph(id='user-graph'),
    dcc.Slider(
        id='user-alpha-slider',
        min=1,
        max=100,
        value=50,
        marks={str(alpha): str(alpha) for alpha in np.arange(0, 110, 10)},
        step=5
    )
])

@app.callback(
    Output('graph', 'figure'),
    Input('alpha-slider', 'value'))
def update_image(alpha):
    fig = make_subplots(1,3)
    fig.add_trace(go.Image(z=image), 1, 1)

    iso_smooth = np.zeros_like(image)
    for i in range(3):
        iso_smooth[:, :, i] = apply_isotropic_smoothing(image[:, :, i], alpha=alpha)

    fig.add_trace(go.Image(z=iso_smooth), 1, 2)

    aniso_smooth = dtensor.smooth(image, alpha=alpha)
    fig.add_trace(go.Image(z=aniso_smooth), 1, 3)

    fig.update_layout(transition_duration=500)

    return fig

@app.callback(
    Output('user-graph', 'figure'),
    Input('upload-image', 'contents'),
    Input('user-alpha-slider', 'value'))
def smooth_user_image(contents, alpha):

    image = load_image(contents)
    user_dtensor = DiffusionTensor2D.fromimage(image)

    fig = make_subplots(1,3)
    fig.add_trace(go.Image(z=image), 1, 1)

    iso_smooth = np.zeros_like(image)
    for i in range(3):
        iso_smooth[:, :, i] = apply_isotropic_smoothing(image[:, :, i], alpha=alpha)

    fig.add_trace(go.Image(z=iso_smooth), 1, 2)

    aniso_smooth = user_dtensor.smooth(image, alpha=alpha)
    fig.add_trace(go.Image(z=aniso_smooth), 1, 3)

    fig.update_layout(transition_duration=500)

    return fig

def load_image(contents):
    # Convert to numpy array
    content_string = contents.split(';base64,')[-1]
    image = np.asarray(b64_to_pil(content_string))

    height, width = image.shape[:2]
    max_height, max_width = (1024, 1024)

    height, width = image.shape[:2]

    if height > max_height or width > max_width:
        image = downsize_image(image, max_height, max_width)

    return image


def downsize_image(image, max_height, max_width):

    height, width = image.shape[:2]
    
    if height > width:
        scaling_factor = max_height / float(height)
    else:
        scaling_factor = max_width / float(width)

    image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return image

if __name__ == '__main__':
    app.run_server(debug=True)