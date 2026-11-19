
import logging
import sys
import random
from uuid import uuid4

import streamlit as st
import pandas as pd
from pathlib import Path

import app.spotify.spotify as spo
from app.web.blind_test.scoring import save_vote
from app.models.baseline_model import BaselineModel
from app.models.autoencoder_knn_model import Autoencoder_kNN_Model
from app.models.gmm_recommender import GMMModel



# ------------------------ Input track lists ------------------------

input_track_lists = [
    {
        "name": "Eclética",
        "filename": "Playlist_ECLETICA.csv"
    },
    {
        "name": "Homogênea",
        "filename": "Playlist_HOMOGENEA.csv"
    },
    {
        "name": "Intermediária",
        "filename": "Playlist_INTERMEDIARIA.csv"
    }
]

random.shuffle(input_track_lists)

# ------------------------ Logs ------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------ Creating (empty) playlists ------------------------

token = spo.get_access_token()


if "input_playlist_ids" not in st.session_state:
    st.session_state.input_playlist_ids = [
        spo.create_playlist(token)
        for _ in range(len(input_track_lists))
    ]

input_playlist_ids = st.session_state.input_playlist_ids


nb_models = 4
if "recommendation_playlist_ids" not in st.session_state:
    st.session_state.recommendation_playlist_ids = [
        spo.create_playlist(token)
        for _ in range(nb_models)
    ]

recommendation_playlist_ids = st.session_state.recommendation_playlist_ids


# ------------------------ Load input tracks ------------------------
# Temporary code until we have final test track lists


root = Path(__file__).resolve().parents[3]
path = root / "artifacts" / "input_track_lists"

# load input tracks
for tracklist in input_track_lists:
    tracklist["track_df"] = pd.read_csv(path / tracklist['filename'])

# show input track lists
st.markdown('# Input tracks')

columns = st.columns(len(input_track_lists))
for i in range(len(columns)):
    with columns[i]:

        label = chr(65 + i)  # A, B, C...
        input_track_lists[i]['label'] = label
        st.subheader(f"Playlist {label}")

        # show Spotify player
        spo.add_tracks_to_playlist(
            token,
            input_playlist_ids[i],
            input_track_lists[i]["track_df"]['track_id'])
        spo.display_player_streamlit(input_playlist_ids[i])


track_list_names = tuple(list["label"] for list in input_track_lists)
selected_input_list_label = st.radio('Select a playlist', track_list_names)


# ------------------------ Load models ------------------------
# Add here as many models as you want

models = []

# Load baseline
baseline_model = BaselineModel()
models.append(baseline_model)

# Load autoencoder
autoencoder_knn_model = Autoencoder_kNN_Model()
models.append(autoencoder_knn_model)

# Load gmm models
pca3_gmm_model = GMMModel(configuracao="pca3_tied15")
models.append(pca3_gmm_model)

pca5_gmm_model = GMMModel(configuracao="pca5_diag30")
models.append(pca5_gmm_model)

# Shuffling the list so that we don't know which model produce which recommendation
# We probably will have to change this when we will implement the user scoring
random.shuffle(models)



# ------------------------ Recommend ------------------------
# Call each model and ask for a recommendation
# Todo: display each reco in a Spotify player

st.markdown('# Recommend')
recommendations = []

if "trial" not in st.session_state:
    st.session_state.trial = None

disabled=st.session_state.trial is not None\
    and selected_input_list_label is not None

if st.button('Recommend', disabled):

    results = []
    input_tracks = next(
        (list for list in input_track_lists if list["label"] == selected_input_list_label),
        None
    )

    # calling all models to get recommendations
    for model in models:

        recommendations = model.recommend(
            input_track_ids=input_tracks['track_df']['track_id']
        )

        results.append({
            "model_id": model.name,
            "track_ids": recommendations
        })

    random.shuffle(results)

    st.session_state.trial = {
        "id": str(uuid4()),
        "input_track_ids": input_tracks['track_df']['track_id'].tolist(),
        "input_track_name": input_tracks['name'],
        "results": results
    }




# ------------------------ Show results and Score models ------------------------

trial = st.session_state.trial

if trial is not None:

    n = len(trial["results"])
    columns = st.columns(n)

    # Show recommendations
    for i, result in enumerate(trial["results"]):
        with columns[i]:

            label = chr(65 + i)  # A, B, C...
            st.subheader(f"Playlist {label}")

            # show Spotify player
            #playlist_id = spo.create_playlist(token)
            spo.add_tracks_to_playlist(token, recommendation_playlist_ids[i], result["track_ids"])
            spo.display_player_streamlit(recommendation_playlist_ids[i])


    # Scoring form
    with st.form("scoring_form"):

        name = st.text_input("Enter your name")
        ranks = []

        # scoring
        for i in range(n):

            label = chr(65 + i) # A, B, C...

            rank = st.selectbox(
                f"Score Playlist {label}",
                options=[None] + list(range(1, n + 1)),
                format_func=lambda x: (
                    "Choose position" if x is None else str(x)
                ),
                key=f"{trial['id']}_{i}"
            )

            ranks.append(rank)

        # saving score
        submitted = st.form_submit_button("Save scores")

        if submitted:

            # check form is not empty
            if not name.strip():
                st.error("Please enter your name")

            elif None in ranks or len(set(ranks)) != n:
                st.error("Please rank each playlist")

            else:

                # build the score json
                scored_results = []

                for i, result in enumerate(trial["results"]):
                    scored_results.append({
                        **result,
                        "blind_label": chr(65 + i),
                        "rank": ranks[i]
                    })

                # build the vote json
                vote = {
                    "id": trial["id"],
                    "voter": name.strip(),
                    "input_track_name": trial["input_track_name"],
                    "input_track_ids": trial["input_track_ids"],
                    "results": scored_results
                }

                # save vote json
                try:
                    save_vote(vote)

                except Exception as e:
                    st.error(f"Unable to save. Please try again. {e}")
                    st.exception(e)

                else:
                    st.session_state.trial = None
                    st.success("Vote saved !")
