import plotly.graph_objects as go
import pandas as pd
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from app import app
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from plotly.tools import mpl_to_plotly
import dash_bootstrap_components as dbc

shd = pd.read_csv('data/sacramento_real_estate_transactions.csv')

# Zip code is better as a string object (categorical) so I will have to convert
# it.
shd['zip'] = shd['zip'].astype(str)

# Looks like we have some values that seem out of place being that there are
# houses with 0 bedrooms, 0 baths, a negative sqr footage and a negative price.
# There are also some bizarre longitudes/latitudes.  A house in Antartica
# perhaps.
# Check out the cities.  Most cities with very few observations.
# Given the large value of houses that have 0 beds, 0 baths and 0 square feet
# I am going to make an assumption that these are plots of land that have yet
# to have anything built on them.
# As a result I will *not* be dropping them.

# Looks like the house with a negative price is also the one with a negative
# squarefeet.
# It is time to make a choice.  Assume that the data was entered improperly and
# is meant
# to be possitive or drop the data.

# Side note, the state is actually labeled wrong as well so drop it, only one
# row.
shd.drop(703, inplace = True)

x = shd['sq__ft'].values
y = shd['price'].values
    # Using other libraries for standard Deviation and Pearson Correlation
    # Coef.
    # Note that in SLR, the correlation coefficient multiplied by the standard
    # deviation of y divided by standard deviation of x is the optimal slope.
optSlope = (scipy.stats.pearsonr(x,y)[0]) * (np.std(y) / np.std(x))
    
    # Pearson Co.  Coef returns a tuple so it needs to be sliced/indexed
    # the optimal beta is found by: mean(y) - b1 * mean(x)
optYint = np.mean(y) - (optSlope * np.mean(x)) 

# Creating a list of predicted values
y_pred = []

for x in shd['sq__ft']:
    y = optYint + (optSlope * x)
    y_pred.append(y)

# Appending the predicted values to the Sacramento housing dataframe to do DF
# calcs
shd['Pred'] = y_pred

# Residuals equals the difference between Y-True and Y-Pred
shd['Residuals'] = abs(shd['price'] - shd['Pred'])
# the mean of our residuals is aproximately $96,000, which means that is
# on average how off our prediction is.

# Plot showing out linear forcast
fig = plt.figure(figsize=(20,20))

# change the fontsize of minor ticks label
plot = fig.add_subplot(111)
plot.tick_params(axis='both', which='major', labelsize=20)

# get the axis of that figure
ax = plt.gca()

# plot a scatter plot on it with our data
ax.scatter(x= shd['sq__ft'], y=shd['price'], c='k')
ax.plot(shd['sq__ft'], shd['Pred'], color='r')

plt.savefig('assets/testLinear.png')

# change to app.layout if running as single page app instead
layout = html.Div([dbc.Container([dbc.Row([dbc.Col(html.H1(children='Simple Linear Regression with Sacramento Real Estate Data'), className="mb-2")]),

        dbc.Row([dbc.Col(dbc.Card(html.H3(children='Visualising out linear forcast, Price vs Sq feet', className="text-center text-light bg-dark"), body=True, color="dark"))]),

        html.Br(),html.Br(),
        html.Img(src="/assets/testLinear.png", height='800px'),
        html.Br(),
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
        columns=[{"name": i, "id": i} for i in shd.columns],
         style_data_conditional=[{
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }],
        sort_action="native",
        sort_mode="multi",
        page_size=100,
        data=shd.to_dict('records')),])])
