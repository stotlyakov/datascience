
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
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose

cwd = os.getcwd()
path = cwd + '/data/walmart.csv'
walmart = pd.read_csv(path)
walmart['Date'] = pd.to_datetime(walmart['Date'])
walmart.set_index('Date', inplace=True)
store1 = walmart[walmart.Store == 1][['Weekly_Sales']].resample('W').sum()
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(store1['Weekly_Sales'], lags=60)

plt.figure(9)
plt.figure(figsize=(800 / 96,800 / 96))

plot_acf(store1['Weekly_Sales'], lags=60)
plt.savefig(cwd + "/assets/wallmartSales.jpeg",dpi=96)

plt.figure(10)
decomposition = seasonal_decompose(store1.Weekly_Sales, freq=13)   
decomposition.plot() 
plt.savefig(cwd + "/assets/wallmartSalesDecompose.jpeg",dpi=96)




layout = html.Div([dbc.Container([dbc.Row([dbc.Col(dbc.Card(dbc.Row([dcc.Link(html.A('GitHub'), href="https://github.com/stotlyakov/datascience/blob/main/apps/hw17.py", style={'color': 'white', 'text-decoration': 'underline'}, target="_blank")]),
        body=True, color="dark"))]),
        html.Br(),

        dbc.Row([dbc.Col(html.H1(children='Explore Time Series'), className="mb-2")]),

        html.P("Using Tine Series, analyze Walmart's weekly sales data over a two-year period from 2010 to 2012. The data set is separated by store and department, but we'll focus on analyzing one store for simplicity."),
        html.P("Create autocorolation for single store weekly sales"),
               html.Img(src="/assets/wallmartSales.jpeg", height='800px'),
        html.P("Create decomposition plot for the store sales data. Where we see increasing sales and w shape trend that shows steady grow."),
        
               html.Img(src="/assets/wallmartSalesDecompose.jpeg", height='800px'),])])

