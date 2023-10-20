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
from flask import Flask
from scipy.spatial import distance
from sklearn.decomposition import PCA

DEFAULT_WORD = "the"


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
    openai.api_key = "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"
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


app = Flask(__name__)
dash_app = dash.Dash(
    __name__,
    server=app,
    routes_pathname_prefix="/dash/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
dash_app.title = "3D Word Embedding Visualizer"


@dash_app.callback(
    [Output("closest-words", "children"), Output("word-input", "value")],
    [
        Input("word-input", "value"),
        Input({"type": "word-tile", "index": ALL}, "n_clicks"),
    ],
    [State({"type": "word-tile", "index": ALL}, "children")],
)
def display_closest_words(word_to_highlight, tile_clicks, tile_labels):
    ctx = dash.callback_context
    if not ctx.triggered:
        word_to_highlight = DEFAULT_WORD

    # Check if any button was clicked
    clicked_button_idx = next(
        (i for i, n in enumerate(tile_clicks) if n and n > 0), None
    )

    # If a word-tile button was clicked, update the word_to_highlight
    if clicked_button_idx is not None:
        word_to_highlight = tile_labels[clicked_button_idx]

    if word_to_highlight not in word_embeddings_map:
        return [], word_to_highlight

    target_embedding = word_embeddings_map[word_to_highlight]
    closest_8 = closest_words(target_embedding, embeddings, words, 8)

    word_tiles = [
        dbc.Button(
            word,
            id={"type": "word-tile", "index": i},
            className="mx-2 my-1 rounded-pill",  # Added rounded-pill for rounded buttons
            color="secondary",  # Changed color to secondary for neutral appearance
            outline=True,
            n_clicks=0,  # initialize with 0 clicks
        )
        for i, word in enumerate(closest_8)
    ]
    return word_tiles, word_to_highlight


@dash_app.callback(Output("3d-plot", "figure"), [Input("word-input", "value")])
def update_graph(word_to_highlight):
    if not word_to_highlight:
        word_to_highlight = DEFAULT_WORD
    colors = ["blue" for word in words]
    sizes = [5 for word in words]

    if word_to_highlight in words:
        idx = words.index(word_to_highlight)
        colors[idx] = "red"
        sizes[idx] = 20

    scatter = go.Scatter3d(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        z=reduced_embeddings[:, 2],
        mode="markers",
        marker=dict(color=colors, size=sizes),
        hovertext=words,
        hoverinfo="text",
    )

    layout = go.Layout(height=1000, margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=[scatter], layout=layout)
    return fig


# Dash layout
dash_app.layout = dbc.Container(
    [
        html.H2(
            "3D Word Embedding Visualizer",
            className="text-center my-5 font-weight-bold",
        ),  # Font-weight bold for header
        # Input Row
        dbc.Row(
            [
                dbc.Col(
                    dcc.Input(
                        id="word-input",
                        type="text",
                        placeholder="Enter a word...",
                        value=DEFAULT_WORD,
                        className="form-control mb-4 p-3",
                        style={"borderRadius": "25px", "fontSize": "1.2em"},
                    ),
                    width={
                        "size": 10,
                        "offset": 1,
                        "md": 6,
                        "mdOffset": 3,
                    },  # Adjusting width and offset for mobile and desktop
                ),
            ],
            justify="center",
        ),
        # Closest Words Row
        dbc.Row(
            [
                dbc.Col(
                    html.Div(id="closest-words", className="text-center my-4"), width=12
                )
            ]
        ),
        # 3D Plot Row
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="3d-plot", style={"marginTop": "20px"}), width=12)
            ],  # Increased margin-top
        ),
    ],
    fluid=True,
    className="p-2 p-md-5",  # Increased padding
)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
