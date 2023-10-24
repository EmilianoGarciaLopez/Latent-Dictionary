import dash_bootstrap_components as dbc
from dash import dcc, html

DEFAULT_WORD = "the"


index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <!-- Open Graph Meta Tag -->
        <meta property="og:image" content="https://static.friendsforduke.org/og_image.png">
        <meta name="description" content="LatentDictionary: Explore word embeddings in 3D and generate real-time semantic visualizations">

    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app_layout = dbc.Container(
    [
        # Search Bar and Closest Words Overlay
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.InputGroup(
                            [
                                dcc.Input(
                                    id="word-input",
                                    type="text",
                                    placeholder="Enter a word...",
                                    value=DEFAULT_WORD,
                                    className="form-control my-2 p-2",
                                    style={
                                        "borderRadius": "15px",
                                        "fontSize": "1.1em",
                                        "backgroundColor": "#f5f5f5aa",
                                        "paddingRight": "40px",
                                    },
                                    n_submit=0,
                                ),
                                dbc.Button(
                                    "?",
                                    id="instructions-button",
                                    className="position-absolute top-50 end-0 translate-middle-y border-0 rounded-circle",
                                    style={
                                        "background": "transparent",
                                        "color": "#808080",
                                        "fontSize": "1.2em",
                                        "right": "15px",
                                    },
                                ),
                                dbc.Tooltip(
                                    "Enter a word or list of words separated by commas and press enter. Words not in Oxford 3000 will generate on the fly.",
                                    target="instructions-button",
                                    placement="bottom",
                                ),
                            ],
                            size="lg",
                        ),
                        dcc.Loading(
                            children=html.Div(
                                id="closest-words", className="text-center my-2"
                            ),
                            type="default",
                        ),
                    ],
                    # Adjust width for different screen sizes
                    xs=12,  # 100% on extra small screens
                    sm=10,  # 83.33% on small screens
                    md=8,  # 66.66% on medium screens
                    lg=6,  # 50% on large screens
                    xl=5,  # 41.66% on extra large screens
                    className="position-fixed top-0 start-50 translate-middle-x p-2 rounded",
                    style={"zIndex": 1000},
                )
            ],
        ),
        # 3D Plot
        dbc.Row(
            [dbc.Col(dcc.Graph(id="3d-plot", style={"height": "100vh"}), width=12)],
        ),
        # Footer Component
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Span("Built by "),
                            html.A(
                                "Emiliano García-López",
                                href="https://github.com/EmilianoGarciaLopez",
                                target="_blank",
                            ),
                            html.Span(". Original concept by "),
                            html.A(
                                "Grant",
                                href="https://twitter.com/granawkins/status/1715231557974462648",
                                target="_blank",
                            ),
                            html.Span("."),
                            html.Br(),  # Break line for aesthetic spacing
                            html.A(
                                "View on Github",
                                href="https://github.com/EmilianoGarciaLopez/Latent-Dictionary",
                                target="_blank",
                            ),
                        ],
                        className="text-end mt-5",
                        style={"fontSize": "0.8em"},
                    ),
                    width=12,
                ),
            ],
            className="position-fixed bottom-0 end-0 mb-2 me-2",
        ),
    ],
    style={"paddingBottom": "50px"},
    fluid=True,
)
