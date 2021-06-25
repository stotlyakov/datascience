import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly
import plotly.graph_objs as go
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from sklearn.metrics.pairwise import cosine_similarity

layout = html.Div([dbc.Container([
    
        dbc.Row([dbc.Col(dbc.Card(
        dbc.Row([
        dcc.Link(html.A('GitHub'), href="https://github.com/stotlyakov/datascience/blob/main/apps/hw15.py", style={'color': 'white', 'text-decoration': 'underline'}, target="_blank")]),
        body=True, color="dark"))]),
        html.Br(),

        dbc.Row([dbc.Col(html.H1(children='Explore the power of Cosine similarity algorithm'), className="mb-2")]),

        html.P("Compare using cosine similarity algorithm provided by sklearn.metrics.pairwise and then compare with manual operation with the help of numpy. This is often used by additional recommender algorithms to find similarities in data and recommend close matches."),

        html.Br(),html.Br(),
        html.Pre(children = [html.Label(children = 'PY'), 
            html.Code(children= "" + 
                "# vectors \n" +
                "v1 = np.array([1,2,3,4,6,4,8]) \n" +
                "v2 = np.array([1,1,4,6,7,8,4]) \\nn" +
 
                "# manually compute cosine similarity using numpy \n" +
                "dot = np.dot(v1, v2) \n" +
                "norma = np.linalg.norm(v1) \n" +
                "normb = np.linalg.norm(v2) \n" +
                "cos_manual = dot / (norma * normb) \n\n" +
 
                "# use library, operates on sets of vectors \n" +
                "v1v1 = v1.reshape(1,7) \n" +
                "v2v1 = v2.reshape(1,7) \n" +
                "cos_sim = cosine_similarity(v1v1, v2v1) \n\n" +
 
                "print( \n" +
                "    cos_manual, \n" +
                "    cos_lib[0][0] \n" +
                ") \n" +
            "\n")], className= 'code code-py'),

        html.P("Similarity using manual: 0.8870866213920899"),
        html.P("Similarity using sklearn: 0.8870866213920899")
         ])])

