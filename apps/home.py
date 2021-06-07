import dash_html_components as html
import dash_bootstrap_components as dbc


layout = html.Div([dbc.Container([dbc.Row([dbc.Col(html.H1("Welcome to Svet's dashboard menu", className="text-center")
                    , className="mb-5 mt-5")]),
        dbc.Row([dbc.Col(html.H5(children='This app is build using Azure, Plotly, Dash, Bootstrap and range of data science Python libraries! ')
                    , className="mb-4")]),

        dbc.Row([dbc.Col(dbc.Card(children=[html.H3(children='HW 1-6 - Pandas and Plotly intro',
                                               className="text-center"),
                                       dbc.Row([dbc.Col(dbc.Button("GitHub", href="https://github.com/stotlyakov/datascience/blob/main/apps/hw1to6.py",
                                                                   color="light", target="_blank"),
                                                        className="mt-3"),
                                                dbc.Col(dbc.Button("HW 1-6", href="/hw1to6",
                                                                   color="light"),
                                                        className="mt-3")], justify="center")],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),

            dbc.Col(dbc.Card(children=[html.H3(children='HW 7 - Linear Regression',
                                               className="text-center"),
                                      
                                       dbc.Row([dbc.Col(dbc.Button("GitHub", href="https://github.com/stotlyakov/datascience/blob/main/apps/hw7.py",target="_blank",
                                                                   color="light"),
                                                        className="mt-3"),
                                                dbc.Col(dbc.Button("HW 7", href="/hw7",
                                                                   color="light"),
                                                        className="mt-3")], justify="center")],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),


            dbc.Col(dbc.Card(children=[html.H3(children='HW 8 - Train, Test, Split',
                                               className="text-center"),
                                       dbc.Row([dbc.Col(dbc.Button("GitHub", href="https://github.com/stotlyakov/datascience/blob/main/apps/hw8.py",target="_blank",
                                                                   color="light"),
                                                        className="mt-3"),
                                                dbc.Col(dbc.Button("HW 8", href="/hw8",
                                                                   color="light"),
                                                        className="mt-3")], justify="center")],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),

            dbc.Col(dbc.Card(children=[html.H3(children='HW 9 - Logistic Regression (Under development)',
                                               className="text-center"),
                                       dbc.Row([dbc.Col(dbc.Button("GitHub", href="",target="_blank",
                                                                   color="light"),
                                                        className="mt-3"),
                                                dbc.Col(dbc.Button("HW 9", href="",
                                                                   color="light"),
                                                        className="mt-3")], justify="center")],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),
            dbc.Col(dbc.Card(children=[html.H3(children='HW 10 - K Nearest Neighbors (Under development)',
                                               className="text-center"),
                                       dbc.Row([dbc.Col(dbc.Button("GitHub", href="",target="_blank",
                                                                   color="light"),
                                                        className="mt-3"),
                                                dbc.Col(dbc.Button("HW 10", href="",
                                                                   color="light"),
                                                        className="mt-3")], justify="center")],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),
             dbc.Col(dbc.Card(children=[html.H3(children='HW 11 - Decision Trees (Under development)',
                                               className="text-center"),
                                       dbc.Row([dbc.Col(dbc.Button("GitHub", href="",target="_blank",
                                                                   color="light"),
                                                        className="mt-3"),
                                                dbc.Col(dbc.Button("HW 11", href="",
                                                                   color="light"),
                                                        className="mt-3")], justify="center")],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),
             dbc.Col(dbc.Card(children=[html.H3(children='HW 12-13 - Clustering and PCA',
                                               className="text-center"),
                                       dbc.Row([dbc.Col(dbc.Button("GitHub", href="https://github.com/stotlyakov/datascience/blob/main/apps/hw12_13.py",target="_blank",
                                                                   color="light"),
                                                        className="mt-3"),
                                                dbc.Col(dbc.Button("HW 11", href="/hw12_13",
                                                                   color="light"),
                                                        className="mt-3")], justify="center")],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),
             ], className="mb-5"),])])
