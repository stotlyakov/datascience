import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import os
import sys
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

cwd = os.getcwd()
path = cwd + '/data/Live Site Issues small.csv'
liveSiteAll = pd.read_excel(path)
liveSiteAll.drop(['App Created By','App Modified By','Content Type','Modified','Modified By','Item Type','Item Type','Defect number','Assigned To','Status','Time Taken CE to Resolve the Issue','Folder Child Count','Path','Item Child Count','Time taken to unblock the customer'], inplace=True, axis=1)

liveSiteAllDescSol = liveSiteAll['Issue']

liveSiteAllDescSolSubset = liveSiteAllDescSol.head(20)
data = liveSiteAllDescSolSubset.values.tolist()

#https://www.nltk.org/api/nltk.tokenize.html
#tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
stop_words = set(stopwords.words('english'))
def clenTokens(word_tokens):
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    filtered_sentence = []
 
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence
    
tagged_data = [TaggedDocument(words=clenTokens(word_tokenize(_d.lower())), tags=[str(i)]) for i, _d in enumerate(data)]



# hyper parameters
#https://radimrehurek.com/gensim/models/doc2vec.html
max_epochs = 500
vec_size =200
alpha = 0.03
minimum_alpha = 0.0025
reduce_alpha = 0.0002


texts = data

# vectorization of the texts
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts).todense()
# used words (axis in our multi-dimensional space)
words = vectorizer.get_feature_names()


model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=minimum_alpha,
                dm =1,#distributed memory (PV-DM) 
               min_count=1,
               workers=4)#very critical, if min is 2 ormore the result is inacurate
model.build_vocab(tagged_data)

# Train the model based on epochs parameter
for epoch in range(max_epochs):
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=20)
   


texts = data

# vectorization of the texts
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts).todense()
# used words (axis in our multi-dimensional space)
words = vectorizer.get_feature_names()
print("words", words)


n_clusters=7
number_of_seeds_to_try=10
max_iter = 300
number_of_process=2 # seads are distributed
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

layout = html.Div([dbc.Container([
    
        dbc.Row([dbc.Col(dbc.Card(
        dbc.Row([
        dcc.Link(html.A('GitHub'), href="https://github.com/stotlyakov/datascience/blob/main/apps/hw14.py", style={'color': 'white', 'text-decoration': 'underline'}, target="_blank"),
        dcc.Link(html.A('Python notebook for HW 14'), href="https://github.com/stotlyakov/datascience/blob/main/notebooks/HW14.ipynb", style={'color': 'white', 'text-decoration': 'underline', 'margin-left':'10px','margin-right':'10px'},target="_blank"),
        ]),

        body=True, color="dark"))]),
        html.Br(),
        dbc.Row([dbc.Col(html.H6(children='We are going to vecorize service support helpers descriptions and we will find what are the most common words in the issues sections.'), className="mb-4")]),

        html.Br(),

        dcc.Graph(id="fig", figure = fig, style={ "width" :"730px", "margin-bottom":"10px","display": "inline-block"}),
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
        columns=[{"name": i, "id": i} for i in liveSiteAll.columns],
         style_data_conditional=[{
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }],
        sort_action="native",
        sort_mode="multi",
        page_size=100,
        data=liveSiteAll.to_dict('records')),
         ])])