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
import re
from app import app
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nlp.nlpModelProvider import NlpModelProvider

cwd = os.getcwd()
path = f'{cwd}/data/Live Site Issues.csv'
supportCloudTermsImageUrl = cwd + "/assets/supportCloudTerms.png"

liveSiteAll = pd.read_csv(path)
liveSiteAll.drop(['App Created By','App Modified By','Content Type','Modified','Modified By','Item Type','Item Type','Defect number','Assigned To','Status','Time Taken CE to Resolve the Issue','Folder Child Count','Path','Item Child Count','Time taken to unblock the customer'], inplace=True, axis=1)

liveSiteAllDescSol = liveSiteAll['Issue']
liveSiteAllDescTerms = liveSiteAll['Description']

liveSiteAllDescSolSubsetTodisplay = liveSiteAll[['Issue', 'Description', 'Solution']]
liveSiteAllDescSolSubsetTodisplayTop = liveSiteAllDescSolSubsetTodisplay.head(20)

liveSiteAllDescSolSubset = liveSiteAllDescSol.head(20)


data = liveSiteAllDescSolSubset.values.tolist()

modlelProvider = NlpModelProvider()

nlpTrainedModel = modlelProvider.getModel()

if nlpTrainedModel == None:
    nlpTrainedModel = modlelProvider.generateModel(data)
 
tagged_data = modlelProvider.tagged_data

# Some more info about the data we have avaialble for support tickets
wordcl = WordCloud(width=550, height=450).generate(" ".join(liveSiteAllDescTerms.values.tolist()).replace(r'\w*[0-9]\w*', ""))
wordcl.to_file(supportCloudTermsImageUrl)

texts = liveSiteAllDescSol.values.tolist()
cleanedText = []
for t in texts:
    cleanedText.append(re.sub('\d', '', t))

# vectorization of the texts
vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit_transform(cleanedText).todense()
sort_orders = sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1], reverse=True)[:10]

xplot = []
yplot = []

for i in sort_orders:
    xplot.append(i[1])
    yplot.append(i[0])
    

dict_of_fig = dict({
    "data": [{"type": "bar",
              "x": xplot,
              "y": yplot}],
    "width":555,
    "layout": {"title": {"text": "Most common words in the support ticket issues"}}
})

figMostCommon = go.Figure(dict_of_fig)




layout = html.Div([dbc.Container([dbc.Row([dbc.Col(dbc.Card(dbc.Row([dcc.Link(html.A('GitHub'), href="https://github.com/stotlyakov/datascience/blob/main/apps/hw14.py", style={'color': 'white', 'text-decoration': 'underline'}, target="_blank"),
        dcc.Link(html.A('Python notebook for HW 14'), href="https://github.com/stotlyakov/datascience/blob/main/notebooks/HW14.ipynb", style={'color': 'white', 'text-decoration': 'underline', 'margin-left':'10px','margin-right':'10px'},target="_blank"),]),

        body=True, color="dark"))]),
        html.Br(),
        dbc.Row([dbc.Col(html.H1(children='Use Natural Language Processing to provide solutions to customer support problems.'), className="mb-4")]),
         html.Br(),
          html.Br(),
        dbc.Row([dbc.Col(dbc.Card(html.H3(children='Introduction', className="text-center text-light bg-dark"), body=True, color="dark"))]),
         html.Br(),
         html.P(children="The goal of this report is to explore customer support tickets data, analyze the text using Natural Language Processing (NLP) and propose solutions to customer issues based on existing information and trained unsupervised model. "),
         html.P(children='''
         The business problem we try to solve is to provide immediate answers to issues raised by the customer while saving company resources. 
         The system can be consumed by support, sales operations and even internal engineering teams. 
         The data sets to be used, contain large volume of available information of how to solve wide array of product and system related issues. 
         We want to provide more fluent and natural way of accessing this information and obtaining the resolution without the need to manually dig
            the data or relay on specific person's domain knowledge.
         '''),

         html.Ol([html.Li(children="Consume the spread sheet and clean up the data using pandas."),
             html.Li(children="Explore data and visualize the data for common support issues."),
             html.Li(children=["Train the NLP Model with the dataset using ", html.Code("genism"), " and ", html.Code("nltk"), "."]),
             html.Li(children="Use the provided UI to state your problem and the resulting information will relate to the top found results using similar sentence matching algorithm."),]),

        dbc.Row([dbc.Col(dbc.Card(html.H3(children='Implementation', className="text-center text-light bg-dark"), body=True, color="dark"))]),
        html.Br(),

        html.P(children='''First, we will consume spread sheet that contains large amount of information related to different support
        tickets or issues discovered by our monitoring and alerting system. The data is displayed in the following table:'''),

        dash_table.DataTable(id='datatable_initialData',
           style_table={'width':1110,'overflowY': 'scroll','height': 400},
            style_header={'backgroundColor': '#343a40', 'color': 'white'},
            style_cell={
                'backgroundColor': 'rgb(233, 233, 233)',
                'color': 'rgb(81, 81, 81)',
                'fontSize': 15,
                'font-family': 'Nunito Sans',
                'textAlign': 'left',
                'minWidth': '180px',
                },
            css=[{'selector': '.row', 'rule': 'margin: 0'}],#Fix the first column of the table
            columns=[{"name": i, "id": i} for i in liveSiteAll.columns],
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
            data=liveSiteAll.to_dict('records')),

        html.P(children=''''''),
        html.P(children = "We are going to train our NLP model on the “Issue” column. But, before we do this we have to do some cleaning of the data. "),
        html.P(children = ["First, we will drop the columns that are not needed in our research. We will obfuscate all customer and PII information from the text. Finally, we will remove all the “stop” words from the text using ", html.Code("nltk"), "."]),

        dbc.Row([dbc.Col(dbc.Card([dbc.CardHeader("Stop words"),
        dbc.CardBody([html.H5("nltk.download('stopwords')", className="card-title"),
                    html.P("‘ourselves’, ‘hers’, ‘between’, ‘yourself’, ‘but’, ‘again’, ‘there’, ‘about’, ‘once’, ‘during’, ‘out’, ‘very’, ‘having’, ‘with’, ‘they’, ‘own’, ‘an’, ‘be’, ‘some’, ‘for’, ‘do’, ‘its’, ‘yours’, ‘such’, ‘into’, ‘of’, ‘most’, ‘itself’, ‘other’, ‘off’, ‘is’, ‘s’, ‘am’, ‘or’, ‘who’, ‘as’, ‘from’, ‘him’, ‘each’, ‘the’, ‘themselves’, ‘until’, ‘below’, ‘are’, ‘we’, ‘these’, ‘your’, ‘his’, ‘through’, ‘don’, ‘nor’, ‘me’, ‘were’, ‘her’, ‘more’, ‘himself’, ‘this’, ‘down’, ‘should’, ‘our’, ‘their’, ‘while’, ‘above’, ‘both’, ‘up’, ‘to’, ‘ours’, ‘had’, ‘she’, ‘all’, ‘no’, ‘when’, ‘at’, ‘any’, ‘before’, ‘them’, ‘same’, ‘and’, ‘been’, ‘have’, ‘in’, ‘will’, ‘on’, ‘does’, ‘yourselves’, ‘then’, ‘that’, ‘because’, ‘what’, ‘over’, ‘why’, ‘so’, ‘can’, ‘did’, ‘not’, ‘now’, ‘under’, ‘he’, ‘you’, ‘herself’, ‘has’, ‘just’, ‘where’, ‘too’, ‘only’, ‘myself’, ‘which’, ‘those’, ‘i’, ‘after’, ‘few’, ‘whom’, ‘t’, ‘being’, ‘if’, ‘theirs’, ‘my’, ‘against’, ‘a’, ‘by’, ‘doing’, ‘it’, ‘how’, ‘further’, ‘was’, ‘here’, ‘than’",
                        className="card-text",),]),], color="light"))]),
        html.Br(),

        html.P(children="The code bellow will give some idea about the process used."),
        html.Pre(children = [html.Label(children = 'PY'), 
                    html.Code(children= "" + 
                              "from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator \n" + 
                              "from sklearn.feature_extraction.text import TfidfVectorizer \n" + 
                              "import pandas as pd \n\n" + 
                           
                              "# Create Bar Chart from the Most used words in the Issues column\n" + 
                              "liveSiteAllDescSol = liveSiteAll['Issue'] \n" + "texts = liveSiteAllDescSol.values.tolist() \n\n" + 
                              "# vectorization of the texts \n" + 
                              "vectorizer = TfidfVectorizer(stop_words='english') \n" + 
                              "words = vectorizer.get_feature_names()"
                    "\n")], className= 'code code-py'),
                html.P(children="The value of words would contain the features names for all vectors."),

             dbc.Row([dbc.Col(dbc.Card([dbc.CardHeader("Word vector features"),
        dbc.CardBody([html.H5("vectorizer.get_feature_names()", className="card-title"),
                    html.P("['account', 'activate', 'address', 'business', 'ca', 'calling', 'ce', 'center', 'cloud', 'commerce', 'consent', 'contract', 'create', 'created', 'customer', 'data', 'dates', 'deleted', 'duplicate', 'effective', 'elements', 'email', 'endpoint', 'entitlement', 'erp', 'error', 'exception', 'failed', 'finding', 'flare', 'gone', 'intervention', 'invalid', 'issue', 'loguseraccess', 'maintenancesku', 'manual', 'message', 'microsoft', 'missing', 'missng', 'ms', 'needs', 'office', 'order', 'payroll', 'placed', 'prevented', 'protection', 'reach', 'reinstated', 'request', 'requires', 'sage', 'service', 'services', 'signon', 'sls', 'suddenly', 'suspended', 'sync', 'tenant', 'tenantid', 'tenants', 'trade', 'traffic', 'transfer', 'uk', 'useraccesslogsvc', 'validated', 'validation']",
                        className="card-text",),]),], color="light"))]),
             html.Br(),
                html.P(children="To display the data on the Bar chart, sort the data desc and get top 10 results."),

                html.Pre(children = [html.Label(children = 'PY'), 
                    html.Code(children= "" + 
                              "sort_order = sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1], reverse=True)[:10] \n\n" + 
                              "xplot = [] \n" + "yplot = [] \n" + "for i in sort_order: \n" + 
                              "    xplot.append(i[1]) \n" + 
                              "   yplot.append(i[0]) \n" + 
                              "dict_of_fig = dict({ \n" + 
                              "   'data': [{'type': 'bar', \n" + 
                              "             'x': xplot, \n" + 
                              "             'y': yplot}], \n" + 
                              "  'layout': {'title': {'text': 'Most common words in the support ticket issues'}} \n" + 
                              "})\n" + "figMostCommon = go.Figure(dict_of_fig) \n\n" +
                              "# Create WordCloud from the Description column in the data \n" + 
                              "liveSiteAll = pd.read_csv(path) \n" + 
                              "wordcl = WordCloud(width=550, height=450).generate(" ".join(liveSiteAllDescTerms.values.tolist()).replace(r'\w*[0-9]\w*', "")) \n" + 
                              "wordcl.to_file(supportCloudTermsImageUrl) \n\n" + 
                    "\n")], className= 'code code-py'),
        html.P(children="Before we go ahead and train the model for our purpose, we will gather some information about our data."),
        html.P(children="After vectorization of all words in the text, we display the top most occuring terms in the Issues section, on the left and Description section on the right."),
        html.Table([html.Tr([html.Td(dcc.Graph(id="figMostCommonWords", figure = figMostCommon, style={"display": "inline-block","width":"550"})),
           html.Td([html.Img(src="/assets/supportCloudTerms.png", width='550', style={"margin-left":"10px","margin-bottom":"5px"})])])]),



        html.Br(),
        dbc.Row([dbc.Col(html.H6(children=[html.Span('After we analyze the most common terms in the Issues and Description texts, we can train our model using '), html.Code('genism Doc2Vec')]), className="mb-4")]),

        html.P(children=["The code creating this model can be found in its dedicated class",dcc.Link(html.A('here'), href="", style={'color': 'white', 'text-decoration': 'underline', 'margin-left':'10px','margin-right':'10px'},target="_blank")]),
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

        dbc.Row([dbc.Col(dbc.Card(html.H3(children='Resources', className="text-center text-light bg-dark"), body=True, color="dark"))]),
        html.Br(),])])
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
