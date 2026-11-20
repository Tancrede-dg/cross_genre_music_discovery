from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from app.models.abstract_model import AbstractModel


class DBSCANModel(AbstractModel):

    def __init__(self):

        # Loading artifacts
        root = Path(__file__).resolve().parents[2]
        path = root / "artifacts" / "models" / "dbscan"

        self.model = joblib.load(path / "dbscan_model.joblib")
        self.scaler = joblib.load(path / "dbscan_scaler.joblib")
        self.catalog_clean = pd.read_parquet(root / "artifacts" / "catalog_clean.parquet").set_index("track_id")

        # DBSCAN has no predict: index the core points to assign new tracks to clusters
        self.core_labels = self.model.labels_[self.model.core_sample_indices_]
        self.core_neighbors = NearestNeighbors(n_neighbors=1).fit(self.model.components_)

        # Scale the whole catalog and assign every track to a cluster (done once)
        self.catalog_scaled = np.asarray(self.scaler.transform(self.catalog_clean[self.audio_features]))
        self.labels = self._predict(self.catalog_scaled)


    @property
    def name(self):
        return "DBSCAN"


    def _predict(self, X):

        # nearest core point of each track
        distances, positions = self.core_neighbors.kneighbors(X)

        # inherit the cluster of the nearest core point, or noise (-1) if it is farther than eps
        labels = self.core_labels[positions[:, 0]]
        labels[distances[:, 0] > self.model.eps] = -1

        return labels


    def recommend(self, input_track_ids):

        # retrieve input tracks from catalog
        input_tracks = self.catalog_clean.loc[input_track_ids, self.audio_features]

        # scale input tracks
        input_tracks_scaled = np.asarray(self.scaler.transform(input_tracks))

        # calculate input tracks profile
        input_profile = input_tracks_scaled.mean(axis=0).reshape(1, -1)

        # find the cluster of the profile
        cluster = self._predict(input_profile)[0]

        # if the profile falls in noise, use the cluster of the closest non-noise track
        if cluster == -1:
            non_noise = np.where(self.labels != -1)[0]
            closest = non_noise[np.argmin(np.linalg.norm(self.catalog_scaled[non_noise] - input_profile, axis=1))]
            cluster = self.labels[closest]

        # candidates: tracks in the same cluster, ignoring the input tracks
        is_candidate = (self.labels == cluster) & ~self.catalog_clean.index.isin(input_track_ids)
        candidates = np.where(is_candidate)[0]

        # sort candidates by euclidean distance to the profile
        distances = np.linalg.norm(self.catalog_scaled[candidates] - input_profile, axis=1)
        order = np.argsort(distances)[:self.output_size]

        output_track_ids = self.catalog_clean.index[candidates[order]].tolist()

        return output_track_ids
