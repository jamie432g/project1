import numpy as np
from sklearn.cluster import KMeans
from utils import get_bottom_centre, measure_distance

class AssignColourTest:
    def __init__(self, random_state=45):
        self.random_state = random_state
        self.kmeans_model = None    
        self.team_centers = None     

    def get_top_half(self, img):
        h = img.shape[0]
        return img[: h // 2, :]

    def analyse_corners(self, labels):
        corners = [labels[0,0], labels[0,1], labels[1,0], labels[1,1]]
        counts  = {c: corners.count(c) for c in set(corners)}
        bg = max(counts, key=counts.get)
        return 1 if bg == 0 else 0

    def extract_shirt_colour(self, shirt):
        pixels = shirt.reshape(-1,3)
        km = KMeans(n_clusters=2, random_state=self.random_state)
        km.fit(pixels)
        lbls = km.labels_.reshape(shirt.shape[:2])
        player_cluster = self.analyse_corners(lbls)
        return km.cluster_centers_[player_cluster]

    def fit(self, frame0, detections0):
        #collect per-player shirt colours on frame 0
        cols = []
        for det in detections0:
            x1, y1, x2, y2 = map(int, det['bbox'])
            shirt = self.get_top_half(frame0[y1:y2, x1:x2])
            cols.append(self.extract_shirt_colour(shirt))

        cols = np.vstack(cols)  # RGB channel
        #cluster into two team‐centres
        self.kmeans_model = KMeans(n_clusters=2, random_state=self.random_state)
        self.kmeans_model.fit(cols)
        self.team_centers = self.kmeans_model.cluster_centers_

    def predict(self, frame, detections):
        #extract each shirt colour
        shirt_cols = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            shirt = self.get_top_half(frame[y1:y2, x1:x2])
            shirt_cols.append(self.extract_shirt_colour(shirt))
        shirt_cols = np.vstack(shirt_cols)  

        #initial colour‐cluster assignment
        pred_idxs = self.kmeans_model.predict(shirt_cols)
        #this is where I thought it was 1 and 2, can't be bothered changing since it works anyways
        teams0   = pred_idxs + 1         

        #write back into detections
        for i, det in enumerate(detections):
            team = teams0[i]
            det['team']        = team
            det['team_colour'] = self.team_centers[team-1].astype(int)

        return detections
