import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from app import server
from app import app
# import all pages in the app
from apps import hw1to6, hw7, hw8,hw9, hw15,hw17, final, home

# building the navigation bar
# https://github.com/facultyai/dash-bootstrap-components/blob/master/examples/advanced-component-usage/Navbars.py
dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Home", href="/home"),
        dbc.DropdownMenuItem("HW 1-6", href="/hw1to6"),
        dbc.DropdownMenuItem("HW 7", href="/hw7"),
        dbc.DropdownMenuItem("HW 8", href="/hw8"),
        dbc.DropdownMenuItem("HW 9", href="/hw9"),
        dbc.DropdownMenuItem("HW 15", href="/hw15"),
        dbc.DropdownMenuItem("HW 17", href="/hw17"),
        #dbc.DropdownMenuItem("HW 12-13", href="/hw12_13"),
        #dbc.DropdownMenuItem("HW 14", href="/hw14"),
         dbc.DropdownMenuItem("Final", href="/final"),
    ],
    nav = True,
    in_navbar = True,
    label = "Explore HW",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/ds3.png", height="40px")),
                        dbc.Col(dbc.NavbarBrand("Svet's Data Science Dash", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/home",
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    # right align dropdown menu with ml-auto className
                    [dropdown], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    className="mb-4",
)

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/hw1to6':
        return hw1to6.layout
    elif pathname == '/hw7':
        return hw7.layout
    elif pathname == '/hw8':
        return hw8.layout
    elif pathname == '/hw12_13':
        return hw12_13.layout
    elif pathname == '/hw14':
        return hw14.layout
    elif pathname == '/final':
        return final.layout
    elif pathname == '/hw9':
        return hw9.layout
    elif pathname == '/hw15':
        return hw15.layout
    elif pathname == '/hw17':
        return hw17.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(debug=True)

