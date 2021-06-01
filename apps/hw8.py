import plotly.graph_objects as go
import pandas as pd
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from app import app
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt8
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# read in the training dataset
full_training = pd.read_csv('data/int_rates_training.csv')
# read in the holdout dataset
testing = pd.read_csv('data/int_rates_testing.csv')

# the target
y = full_training['interest_rate_change']
X = full_training.drop(['interest_rate_change', 'TITLE'], axis=1)


# split into training and validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,  random_state=42 )

lr = LinearRegression()

cross_val_score(lr,
                X=X_train, # use the 100-column version obviously!
                y=y_train,
                cv=5,
                scoring="neg_root_mean_squared_error")
# fit the model
lr.fit(X_train, y_train)

# view the coefficients as a dataframe
coeffs = pd.DataFrame(list(zip(X.columns, lr.coef_)), columns=['feature', 'coefficient'])
coeffs=coeffs.set_index('feature', drop=True)
top10 = coeffs.sort_values('coefficient', ascending=False).head(10)
bottom10 = coeffs.sort_values('coefficient', ascending=False).tail(10)
finalcoeffs = pd.concat([top10, bottom10], axis=0)

figCoefinit = px.bar(x=finalcoeffs.index, y=finalcoeffs['coefficient'])

figCoef = go.Figure(figCoefinit)

# the target
y_test = testing['interest_rate_change']

# the features
X_test = testing.drop(['interest_rate_change', 'TITLE'], axis=1)

# predictions on the testing dataset
y_preds = lr.predict(X_test)
fig=sns.regplot(x=y_preds, y=y_test)
#Can't figure out how to display sns figure on html so show the image created fom the python notebook.

layout = html.Div([dbc.Container([
    
        dbc.Row([dbc.Col(dbc.Card(
        dbc.Row([
        dcc.Link(html.A('GitHub'), href="https://github.com/stotlyakov/datascience/blob/main/apps/hw8.py", style={'color': 'white', 'text-decoration': 'underline'}, target="_blank"),
        dcc.Link(html.A('Testing data set: int_rates_testing.csv'), href="https://github.com/stotlyakov/datascience/blob/main/data/int_rates_testing.csv", style={'color': 'white', 'text-decoration': 'underline', 'margin-left':'10px','margin-right':'10px'},target="_blank"),
        dcc.Link(html.A('Training data set: int_rates_training.csv'), href="https://github.com/stotlyakov/datascience/blob/main/data/int_rates_training.csv", style={'color': 'white', 'text-decoration': 'underline', 'margin-left':'10px','margin-right':'10px'},target="_blank")
        ]),

        body=True, color="dark"))]),
        html.Br(),

        dbc.Row([dbc.Col(html.H1(children='Train-test split: Interest Rates'), className="mb-2")]),

        dbc.Row([dbc.Col(html.H6(children='Examine government bond interest rates from the US treasury and business/financial news headlines from various sources, to see if we can predict the changes in interest rates from the news. The question we are getting at is: do news headlines give us an indication of future changes in interest rates?'), className="mb-4")]),
        
        dbc.Row([dbc.Col(dbc.Card(html.H3(children='Visualize the top 10 and bottom 10 coefficients', className="text-center text-light bg-dark"), body=True, color="dark"))]),
        html.Br(),
        dcc.Graph(id='figCoef', figure=figCoef),

        html.Br(),
        dbc.Row([dbc.Col(dbc.Card(html.H3(children='Compare test with train data!', className="text-center text-light bg-dark"), body=True, color="dark"))]),
        html.Br(),
        html.Img(src="/assets/hw8_compare2.png", height='400px'),
        html.Br(),
        html.Br(),
        dbc.Row([dbc.Col(dbc.Card(html.H3(children='Train Data set', className="text-center text-light bg-dark"), body=True, color="dark"))]),
        html.Br(),
       dash_table.DataTable(id='datatable_trainDs',
       style_table={'overflowY': 'scroll',
                    'height': 400,},
        style_header={'backgroundColor': '#343a40', 'color': 'white'},
        style_cell={
            'backgroundColor': 'white',
            'color': 'black',
            'fontSize': 10,
            'font-family': 'Nunito Sans'},
        columns=[{"name": i, "id": i} for i in full_training.columns],
         style_data_conditional=[{
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }],
        sort_action="native",
        sort_mode="multi",
        page_size=100,
        data=full_training.to_dict('records')),
         ])])
