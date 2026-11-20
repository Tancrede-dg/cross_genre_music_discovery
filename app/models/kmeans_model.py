from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from app.models.abstract_model import AbstractModel


class KMeansModel(AbstractModel):

    def __init__(self):

        # Loading artifacts
        root = Path(__file__).resolve().parents[2]
        path = root / "artifacts" / "models" / "kmeans"

        self.model = joblib.load(path / "kmeans_model.joblib")
        self.scaler = joblib.load(path / "kmeans_scaler.joblib")
        self.catalog_clean = pd.read_parquet(root / "artifacts" / "catalog_clean.parquet").set_index("track_id")

        # Scale the whole catalog and assign every track to a cluster (done once)
        self.catalog_scaled = np.asarray(self.scaler.transform(self.catalog_clean[self.audio_features]))
        self.labels = self.model.predict(self.catalog_scaled)


    @property
    def name(self):
        return "K-Means"


    def recommend(self, input_track_ids):

        # retrieve input tracks from catalog
        input_tracks = self.catalog_clean.loc[input_track_ids, self.audio_features]

        # scale input tracks
        input_tracks_scaled = np.asarray(self.scaler.transform(input_tracks))

        # calculate input tracks profile
        input_profile = input_tracks_scaled.mean(axis=0).reshape(1, -1)

        # find the cluster of the profile
        cluster = self.model.predict(input_profile)[0]

        # candidates: tracks in the same cluster, ignoring the input tracks
        is_candidate = (self.labels == cluster) & ~self.catalog_clean.index.isin(input_track_ids)
        candidates = np.where(is_candidate)[0]

        # sort candidates by euclidean distance to the profile
        distances = np.linalg.norm(self.catalog_scaled[candidates] - input_profile, axis=1)
        order = np.argsort(distances)[:self.output_size]

        output_track_ids = self.catalog_clean.index[candidates[order]].tolist()

        return output_track_ids
