import os
import pickle

import dash
import dash_bootstrap_components as dbc
import numpy as np
import openai
import plotly.graph_objects as go
import requests
from dash import dcc, html
from dash.dependencies import ALL, Input, Output, State
from dotenv import load_dotenv
from flask import Flask
from scipy.spatial import distance
from sklearn.decomposition import PCA

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")


DEFAULT_WORD = "the"
model = "text-embedding-ada-002"


# Fetch the Oxford 3000 words
def fetch_oxford_3000():
    url = "https://raw.githubusercontent.com/sapbmw/The-Oxford-3000/master/The_Oxford_3000.txt"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error {response.status_code}: Unable to fetch data")
        return []
    return response.text.splitlines()


def filter_alpha(strings):
    return [s.lower() for s in strings if s.isalpha()]


def save_embeddings_to_file(embeddings_map, filename="embeddings.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(embeddings_map, f)


def load_embeddings_from_file(filename="embeddings.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


oxford_3000 = filter_alpha(fetch_oxford_3000())

EMBEDDINGS_FILE = "embeddings.pkl"
if os.path.exists(EMBEDDINGS_FILE):
    word_embeddings_map = load_embeddings_from_file(EMBEDDINGS_FILE)
else:
    word_embeddings_map = {}
    model = "text-embedding-ada-002"
    n_items = len(oxford_3000)
    batch_size = 1000
    n_batches = (n_items + batch_size - 1) // batch_size
    for i in range(n_batches):
        start, end = i * batch_size, (i + 1) * batch_size
        input = oxford_3000[start:end]
        response = openai.Embedding.create(input=input, model=model)
        embeddings = [i["embedding"] for i in response["data"]]
        for word, embedding in zip(input, embeddings):
            word_embeddings_map[word] = embedding
    save_embeddings_to_file(word_embeddings_map, EMBEDDINGS_FILE)

words = list(word_embeddings_map.keys())
embeddings = np.array(list(word_embeddings_map.values()))

# Perform 3D PCA
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings)


def closest_words(target_embedding, embeddings, words, n=8):
    distances = [distance.euclidean(target_embedding, emb) for emb in embeddings]
    sorted_indices = np.argsort(distances)
    return [words[idx] for idx in sorted_indices[: min(n, len(words))]]


def fetch_and_add_embeddings(words_to_embed):
    global words, embeddings, reduced_embeddings

    if not isinstance(words_to_embed, list):
        words_to_embed = [words_to_embed]

    try:
        response = openai.Embedding.create(input=words_to_embed, model=model)
        new_embeddings = [word_data["embedding"] for word_data in response["data"]]
        for word, embedding in zip(words_to_embed, new_embeddings):
            word_embeddings_map[word] = embedding
            if word not in words:
                words.append(word)
    except Exception as e:
        print(f"Failed to fetch embeddings. Reason: {str(e)}")
        print(f"Words causing the issue: {words_to_embed}")

    embeddings = np.array(list(word_embeddings_map.values()))
    reduced_embeddings = pca.fit_transform(embeddings)


app = Flask(__name__)
dash_app = dash.Dash(
    __name__,
    server=app,
    routes_pathname_prefix="/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

dash_app.title = "LatentDictionary | Embeddings as a Dictionary"
dash_app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <!-- Open Graph Meta Tag -->
        <meta property="og:image" content="https://static.friendsforduke.org/og_image.png">
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


@dash_app.callback(
    [Output("closest-words", "children"), Output("word-input", "value")],
    [
        Input("word-input", "n_submit"),  # Changed from "value" to "n_submit"
        Input({"type": "word-tile", "index": ALL}, "n_clicks"),
    ],
    [
        State("word-input", "value"),
        State({"type": "word-tile", "index": ALL}, "children"),
    ],  # Added the word-input's value to the state
)
def display_closest_words(
    word_submit_count, tile_clicks, word_input_value, tile_labels
):  # Changed parameters
    ctx = dash.callback_context
    component_id = ""  # Initialize component_id here

    if not ctx.triggered:
        word_to_highlight = DEFAULT_WORD
    else:
        component_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if "word-tile" in component_id:
            clicked_button_idx = next(
                (i for i, n in enumerate(tile_clicks) if n and n > 0), None
            )
            if clicked_button_idx is not None:
                word_to_highlight = tile_labels[clicked_button_idx]
            else:
                word_to_highlight = DEFAULT_WORD
        else:
            words_to_highlight = [
                word.strip()
                for word in (word_input_value or "").split(",")
                if word.strip()
            ]
            words_not_in_map = [
                word for word in words_to_highlight if word not in word_embeddings_map
            ]
            if words_not_in_map:
                fetch_and_add_embeddings(words_not_in_map)
                # After adding the new words, update the PCA-reduced embeddings
                global reduced_embeddings
                reduced_embeddings = pca.fit_transform(np.array(embeddings))

            word_to_highlight = (
                words_to_highlight[0] if words_to_highlight else DEFAULT_WORD
            )

    # If after fetching the embeddings we still don't have the word_to_highlight, something failed.
    if word_to_highlight not in word_embeddings_map:
        return [
            html.Div(f"Failed to fetch embedding for '{word_to_highlight}'.")
        ], dash.no_update

    target_embedding = word_embeddings_map[word_to_highlight]
    closest_8 = closest_words(target_embedding, embeddings, words, 8)

    word_tiles = [
        dbc.Button(
            word,
            id={"type": "word-tile", "index": i},
            className="mx-2 my-1 rounded-pill",
            color="secondary",
            outline=True,
            n_clicks=0,
        )
        for i, word in enumerate(closest_8)
    ]

    if component_id and "word-tile" in component_id:
        return [
            html.Div(word_tiles)
        ], word_to_highlight  # update the input box with the clicked word
    else:
        return [html.Div(word_tiles)], dash.no_update  # no update for the input box


@dash_app.callback(
    Output("3d-plot", "figure"),
    [
        Input("word-input", "n_submit"),
        Input({"type": "word-tile", "index": ALL}, "n_clicks"),
    ],
    [
        State("word-input", "value"),
        State({"type": "word-tile", "index": ALL}, "children"),
    ],
)
def update_graph(n_submit, tile_clicks, input_value, tile_labels):
    ctx = dash.callback_context

    clicked_button_idx = next(
        (i for i, n in enumerate(tile_clicks) if n and n > 0), None
    )

    # If a word-tile button was clicked, update the words_to_highlight
    if clicked_button_idx is not None:
        words_to_highlight = [tile_labels[clicked_button_idx]]
    else:
        words_to_highlight = [word.strip() for word in input_value.split(",")]

    colors = ["blue" if word not in words_to_highlight else "red" for word in words]
    sizes = [20 if word in words_to_highlight else 5 for word in words]

    scatter = go.Scatter3d(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        z=reduced_embeddings[:, 2],
        mode="markers",
        marker=dict(color=colors, size=sizes),
        hovertext=words,
        hoverinfo="text",
    )

    layout = go.Layout(
        height=1000,
        margin=dict(l=0, r=0, b=0, t=0),
        uirevision="constant",
    )

    fig = go.Figure(data=[scatter], layout=layout)
    return fig


# Dash layout
dash_app.layout = dbc.Container(
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
                                        "paddingRight": "40px",  # to ensure text doesn't overlap with the button
                                    },
                                    n_submit=0,  # Initialize n_submit to 0
                                ),
                                dbc.Button(
                                    "?",
                                    id="instructions-button",
                                    className="position-absolute top-50 end-0 translate-middle-y border-0 rounded-circle",
                                    style={
                                        "background": "transparent",
                                        "color": "#808080",  # Grey color
                                        "fontSize": "1.2em",
                                        "right": "15px",  # adjust as needed for your design
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
                    className="position-fixed top-0 start-50 translate-middle-x p-2 rounded",
                    style={"zIndex": 1000},
                    width=5,
                )
            ],
        ),
        # 3D Plot
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="3d-plot", style={"height": "100vh"}), width=12
                )  # Set height to 100% of viewport height
            ],
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
    style={
        "paddingBottom": "50px"  # Add enough padding at the bottom to make space for the footer
    },
    fluid=True,
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=8050)
