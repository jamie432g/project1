import sys
import numpy as np
sys.path.append('../')
from utils import get_centre_of_bbox, measure_distance, get_bottom_centre

class Possession:
    def __init__(self):
        self.max_player_ball_distance = 30  #can experiment with this value to find the most accurate value

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_centre_of_bbox(ball_bbox)

        min_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_position = get_bottom_centre(player['bbox'])

            #find distance between player and ball
            distance = measure_distance(player_position, ball_position)

            #assign closest player the ball if in the min distance
            if distance < self.max_player_ball_distance and distance < min_distance:
                min_distance = distance
                assigned_player = player_id

        return assigned_player

    def track_possession(self, tracks):
        team_possession = []
        #iterate over each frame, tracking over the frames which team has the ball
        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = self.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                team_possession.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                team_possession.append(team_possession[-1] if team_possession else None) 

        return np.array(team_possession) 
