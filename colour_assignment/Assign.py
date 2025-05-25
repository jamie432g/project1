from sklearn.cluster import KMeans,MeanShift
import numpy as np
from utils import get_bottom_centre,measure_distance
class AssignColour:
    def __init__(self):
        self.colours = {}
        self.player_team = {}
        self.kmeans_model_instance = KMeans(n_clusters=2,init="k-means++",n_init="auto")

    def assign_teams_to_tracks(self, video_frames, tracks):
        #assign player tracks to a team
        #get initial team colours
        self.get_team_colour(video_frames[0], tracks['players'][0])

        for frame_num, player_track in enumerate(tracks['players']):
            positions = []
            player_ids = []
            for player_id, track in player_track.items():
                team = self.assign_player_to_team (video_frames[frame_num], track['bbox'], player_id)
                
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_colour'] = self.colours[team]

                positions.append(get_bottom_centre(track["bbox"]))
                player_ids.append(player_id)
            player_positions = np.array(positions)
            team_0_centroid = player_positions[np.array([self.player_team[id] == 1 for id in player_ids])].mean(axis=0)
            team_1_centroid = player_positions[np.array([self.player_team[id] == 2 for id in player_ids])].mean(axis=0)

            for gk_id, gk_track in tracks['goalkeepers'][frame_num].items():
                gk_bbox = gk_track['bbox']
                gk_center = get_bottom_centre(gk_bbox)

                dist_0 = measure_distance(gk_center, team_0_centroid)
                dist_1 = measure_distance(gk_center, team_1_centroid)

                gk_team = 1 if dist_0 < dist_1 else 2

                tracks['goalkeepers'][frame_num][gk_id]['team'] = gk_team
                tracks['goalkeepers'][frame_num][gk_id]['team_colour'] = self.colours[gk_team]

        return tracks  
    def kmeans_model(self,image):
        #return the k means image
        image_2d = image.reshape(-1,3)
        return self.kmeans_model_instance.fit(image_2d)
    
    def clustered_image(self,km,s_bbox):
        #takes kmeans and shirt bounding box to return clusted image of bbox
        labels = km.labels_
        return labels.reshape(s_bbox.shape[:2])
    
    def get_top_half(self,image):
        #get top half of image so we get player shirt colour
        return image[: image.shape[0] // 2, :]
    
    def analyse_corners(self,clustered_image):
        #analyse corner of images, determining background and player jersey colour
        get_colours_corner = [clustered_image[0,0],clustered_image[0,1],clustered_image[1,0],clustered_image[1,1]]
        cluster_counts = {cluster: get_colours_corner.count(cluster) for cluster in set(get_colours_corner)}
        non_player_cluster = None
        max_count = 0

        for cluster, count in cluster_counts.items():
            if count > max_count:
                non_player_cluster = cluster
                max_count = count
        if non_player_cluster == 0:
            player_cluster = 1
        else:
            player_cluster = 0

        return player_cluster
    
    def get_team_colour(self,frame,player_detections):
        #team colour determined by clustering colours of players jerseys
        player_colours = []
        for _,player_detection in player_detections.items():
            bbox = player_detection['bbox']
            x1,y1,x2,y2 = bbox

            shirt = self.get_top_half(frame[int(y1):int(y2),int(x1):int(x2)])
            k_shirt = self.kmeans_model(shirt)
            cluster = self.clustered_image(k_shirt,shirt)
            colour_shirt = self.analyse_corners(cluster)
            k_colour = k_shirt.cluster_centers_[colour_shirt]
            player_colours.append(k_colour)
        
        kmeans = KMeans(n_clusters=2,init="k-means++",n_init="auto").fit(player_colours)

        self.kmeans = kmeans

        self.colours[1] = kmeans.cluster_centers_[0]
        self.colours[2] = kmeans.cluster_centers_[1]

    def assign_player_to_team (self,frame,bbox,player_id):
        #new track detected, so find team
        if player_id in self.player_team:
            return self.player_team[player_id]
        
        x1,y1,x2,y2 = bbox
        shirt = self.get_top_half(frame[int(y1):int(y2),int(x1):int(x2)])
        k_shirt = self.kmeans_model(shirt)
        cluster = self.clustered_image(k_shirt,shirt)
        colour_shirt = self.analyse_corners(cluster)
        k_colour = k_shirt.cluster_centers_[colour_shirt]

        team_id = self.kmeans.predict(k_colour.reshape(1,-1))[0]
        team_id+=1

        self.player_team[player_id] = team_id

        return team_id