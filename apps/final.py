import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import dash
import dash_table
from sklearn.decomposition import PCA
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import os
from app import app
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nlp.nlpModelProvider import NlpModelProvider

cwd = os.getcwd()
path = f'{cwd}/data/Live Site Issues.csv'

liveSiteAll = pd.read_csv(path)
liveSiteAll.drop(['App Created By','App Modified By','Content Type','Modified','Modified By','Item Type','Item Type','Defect number','Assigned To','Status','Time Taken CE to Resolve the Issue','Folder Child Count','Path','Item Child Count','Time taken to unblock the customer'], inplace=True, axis=1)

liveSiteAllDescSol = liveSiteAll['Issue']

liveSiteAllDescSolSubsetTodisplay = liveSiteAll[['Issue', 'Description', 'Solution']]
liveSiteAllDescSolSubsetTodisplayTop = liveSiteAllDescSolSubsetTodisplay.head(20)

liveSiteAllDescSolSubset = liveSiteAllDescSol.head(20)


data = liveSiteAllDescSolSubset.values.tolist()

modlelProvider = NlpModelProvider()

nlpTrainedModel = modlelProvider.getModel()

if nlpTrainedModel == None:
    nlpTrainedModel = modlelProvider.generateModel(data)
    
    

# Some more info about the data we have avaialble for support tickets
texts = data

# vectorization of the texts
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts).todense()
# used words (axis in our multi-dimensional space)
words = vectorizer.get_feature_names()
print("words", words)


n_clusters = 7
number_of_seeds_to_try = 10
max_iter = 300
number_of_process = 2 # seads are distributed
model = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=number_of_seeds_to_try, n_jobs=number_of_process).fit(X)

vectorizer.vocabulary_
sort_orders = sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1], reverse=True)[:20]

xplot = []
yplot = []

for i in sort_orders:
    xplot.append(i[1])
    yplot.append(i[0])
    
import plotly.graph_objects as go

dict_of_fig = dict({
    "data": [{"type": "bar",
              "x": xplot,
              "y": yplot}],
    "layout": {"title": {"text": "Show the most common words in the support ticket issue description"}}
})

fig = go.Figure(dict_of_fig)

layout = html.Div([dbc.Container([dbc.Row([dbc.Col(dbc.Card(dbc.Row([dcc.Link(html.A('GitHub'), href="https://github.com/stotlyakov/datascience/blob/main/apps/hw14.py", style={'color': 'white', 'text-decoration': 'underline'}, target="_blank"),
        dcc.Link(html.A('Python notebook for HW 14'), href="https://github.com/stotlyakov/datascience/blob/main/notebooks/HW14.ipynb", style={'color': 'white', 'text-decoration': 'underline', 'margin-left':'10px','margin-right':'10px'},target="_blank"),]),

        body=True, color="dark"))]),
        html.Br(),
        dbc.Row([dbc.Col(html.H1(children='Use Natural Language Processing to provide solutions to customer support problems..'), className="mb-4")]),
         html.Br(),
          html.Br(),
        dbc.Row([dbc.Col(dbc.Card(html.H3(children='Introduction', className="text-center text-light bg-dark"), body=True, color="dark"))]),
         html.Br(),
         html.P(children="The Goal of this report is to explore customer support tickets data, analyze the text using Natural Language Processing (NLP) and propose solutions to customer issues based on existing information and trained unsupervised model. "),
         html.P(children='''
         The business problem we try to solve is to provide immediate answers to issues raised by the customer while saving company resources. 
         The system we want to provide can be consumed by support, sales operations and even internal engineering teams. 
         The data sets to be used, contain large volume of available information of how to solve wide array of product and system related issues. 
         We want to provide more fluent and natural way of accessing this information and obtaining the resolution without the need to manually dig
            the data or relay on specific person domain knowledge.
         '''),

        dbc.Row([dbc.Col(html.H6(children='We are going to vecorize service support helpers descriptions and we will find what are the most common words in the issues sections.'), className="mb-4")]),

        html.Br(),

        dbc.Row([dbc.Col(dbc.Card(html.H3(children='Uses of the trained model', className="text-center text-light bg-dark"), body=True, color="dark"))]),
        html.Br(),
        html.H6("Enter your question here and the NLP model will suggest the best matching exisitng issue and the possible solutions!"),
        html.Div([dcc.Input(id='my-input', value='',placeholder='Enter your question here', type='text', className="form-control", style={'width': '100%'})]),

        html.Br(),
        html.Div(id='my-output'),

        dash_table.DataTable(id='datatable_trainDs',
           style_table={'width':1080, 'margin-left':'15px'},
            style_header={'backgroundColor': '#343a40', 'color': 'white'},
            style_cell={
                'backgroundColor': 'rgb(233, 233, 233)',
                'color': 'rgb(81, 81, 81)',
                'fontSize': 15,
                'font-family': 'Nunito Sans',
                'textAlign': 'left',
                'minWidth': '180px',
                },
            columns=[{"name": i, "id": i} for i in liveSiteAllDescSolSubsetTodisplayTop.columns],
             style_data_conditional=[{
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            },  {
                "if": {"state": "selected"},
                "backgroundColor": "inherit !important",
                "border": "inherit !important",
            }],
            style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
        'lineHeight': '15px',
        'padding':'5px'
            },
            fill_width=False,
            page_size=100,
            editable=True,
            data=liveSiteAllDescSolSubsetTodisplayTop.to_dict('records')),

        html.Br(),
        html.Br(),

        dcc.Graph(id="fig", figure = fig, style={ "width" :"730px", "margin-bottom":"10px","display": "inline-block"}),
        dbc.Row([dbc.Col(dbc.Card(html.H3(children='Train Data set', className="text-center text-light bg-dark"), body=True, color="dark"))]),
        html.Br(),
         ])])
#Working
#@app.callback(
#    Output(component_id='my-output', component_property='children'),
#    [Input(component_id='my-input', component_property='value')]
#)

#https://community.plotly.com/t/multiple-outputs-in-dash-now-available/19437
@app.callback([Output('my-output', 'children'),
     Output('datatable_trainDs', 'data')],
    [Input(component_id='my-input', component_property='value')])

def update_output_div(input_value):
    tokens = input_value.split()
    vector = nlpTrainedModel.infer_vector(tokens)
    most_similar = nlpTrainedModel.dv.most_similar([vector])

    #https://pandas.pydata.org/pandas-docs/stable/getting_started/intro_tutorials/03_subset_data.html
    liveSiteAllDescSolSubsetTodisplayTopUpdated = liveSiteAllDescSolSubsetTodisplayTop[(liveSiteAllDescSolSubsetTodisplayTop['Issue'] == data[int(most_similar[0][0])]) | (liveSiteAllDescSolSubsetTodisplayTop['Issue'] == data[int(most_similar[1][0])]) | (liveSiteAllDescSolSubsetTodisplayTop['Issue'] == data[int(most_similar[2][0])]) | (liveSiteAllDescSolSubsetTodisplayTop['Issue'] == data[int(most_similar[3][0])])]
            
    return (html.Table([html.Tr([html.Td(f"Accuracy -> {str(round(most_similar[0][1], 4))}"), html.Td(f"{data[int(most_similar[0][0])]}")]),
        html.Tr([html.Td(f"Accuracy -> {str(round(most_similar[1][1], 4))}"), html.Td(f"{data[int(most_similar[1][0])]}")]),
        html.Tr([html.Td(f"Accuracy -> {str(round(most_similar[2][1], 4))}"), html.Td(f"{data[int(most_similar[2][0])]}")]),
        html.Tr([html.Td(f"Accuracy -> {str(round(most_similar[3][1], 4))}"), html.Td(f"{data[int(most_similar[3][0])]}")])], className='table')), liveSiteAllDescSolSubsetTodisplayTopUpdated.to_dict('records')
