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

cwd = os.getcwd()
path = cwd + '/data/bankruptcy.csv'
df = pd.read_csv(path)

corr_matrix = df.drop("bankruptcy_label", axis=1).corr()
correlation_pairs = corr_matrix[(corr_matrix > 0.9) | (corr_matrix < -0.9)].stack()
corr_df = correlation_pairs.reset_index(name="correlation")
df1 = pd.DataFrame(np.sort(corr_df[['level_0','level_1']], axis=1))

plt.figure(8)
plt.figure(figsize=(1020/96,300/96))
fig, ax = plt.subplots(figsize=(18, 18))

corr_matrix_to_plot = corr_matrix.copy()

# reduce column names to N chars max
max_chars = 15
corr_matrix_to_plot.columns = [c[:max_chars] for c in corr_matrix_to_plot.columns]
# same for index (i.e. row labels)
corr_matrix_to_plot.index = [c[:max_chars] for c in corr_matrix_to_plot.index]

sns.heatmap(
    data=corr_matrix_to_plot,
    vmin=-1,
    vmax=1, # explicitly set the boundaries
    center=0,
    square=True,
    linewidths=1,
    ax=ax
)



plt.savefig(cwd + "/assets/logisticRegrHeatMap.jpeg",dpi=96)


# use negative indexing (~ operator) to only keep what ISN'T a duplicate pair
corr_df = corr_df.reset_index(drop=True)[~df1.duplicated()]

# show top 10
#corr_df.sort_values("abs_correlation", ascending=False).head(10)

layout = html.Div([dbc.Container([
    
        dbc.Row([dbc.Col(dbc.Card(
        dbc.Row([
        dcc.Link(html.A('GitHub'), href="https://github.com/stotlyakov/datascience/blob/main/apps/hw9.py", style={'color': 'white', 'text-decoration': 'underline'}, target="_blank"),
        dcc.Link(html.A('Data set: sacramento_real_estate_transactions.csv'), href="https://github.com/stotlyakov/datascience/blob/main/data/bankruptcy.csv", style={'color': 'white', 'text-decoration': 'underline', 'margin-left':'10px','margin-right':'10px'},target="_blank")]),
        body=True, color="dark"))]),
        html.Br(),

        dbc.Row([dbc.Col(html.H1(children=' Exploratory data analysis on bankruptcy data'), className="mb-2")]),

        dbc.Row([dbc.Col(dbc.Card(html.H3(children='Visualizing heat map of the correlation matrix', className="text-center text-light bg-dark"), body=True, color="dark"))]),
        html.P("We display how features correlate to each other, to identify columns that perhaps encode the same information. We'll do this by calculating the correlation matrix (how every column correlates with every other) and inspect it visually as a heatmap."),

        html.Br(),html.Br(),
        html.Img(src="/assets/logisticRegrHeatMap.jpeg", height='800px'),
        html.Br(),html.Br(),
        dbc.Row([dbc.Col(dbc.Card(html.H3(children='Dataset', className="text-center text-light bg-dark"), body=True, color="dark"))]),
        html.Br(),
         dash_table.DataTable(id='datatable_h6',
       style_table={'overflowY': 'scroll',
                    'height': 400,},
        style_header={'backgroundColor': '#343a40', 'color': 'white'},
        style_cell={
            'backgroundColor': 'white',
            'color': 'black',
            'fontSize': 10,
            'font-family': 'Nunito Sans'},
        columns=[{"name": i, "id": i} for i in df.columns],
         style_data_conditional=[{
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }],
        sort_action="native",
        sort_mode="multi",
        page_size=100,
        data=df.to_dict('records')),])])

