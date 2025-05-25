import numpy as np
from utils import measure_distance, get_bottom_centre
from ViewTransform.ViewTransformer import ViewTransformer

class SpeedDistanceEstimation:
    def __init__(self):
        pass
    def calculate_speed_distance(self,tracks,rp_dict,frame_rp_dict):
        #speed and distance relies on consistent keypoint detection, fluctuations will harm results.
        frame_window = 10 #experimentally can find the best results
        frame_rate = 24
        speed_distance_dict={}

        for object,object_tracks in tracks.items():
            if object == "players":
                frame_num_length = len(object_tracks)

                for frame_num in range(0,frame_num_length,frame_window):
                    last_frame = min(frame_num + frame_window, frame_num_length-1)
                    for track_id, player in object_tracks[frame_num].items():
                        if track_id not in object_tracks[last_frame]:
                            continue

                        #get transformed positions of each player
                        start_position = get_bottom_centre(player["bbox"])
                        start_position_np = np.array(start_position,dtype=np.float32).reshape(-1,2)
                        end_position = get_bottom_centre(object_tracks[last_frame][track_id]["bbox"])
                        end_position_np = np.array(end_position,dtype=np.float32).reshape(-1,2)

                        #get team info for the csv output
                        team = player.get("team","?")

                        view_transform = ViewTransformer()

                        transformed_s_pos = view_transform.homography_transform(source=rp_dict.get(frame_num),
                                                                                target=frame_rp_dict.get(frame_num),
                                                                                points=start_position_np)

                        transformed_e_pos = view_transform.homography_transform(source=rp_dict.get(last_frame),
                                                                                target=frame_rp_dict.get(last_frame),
                                                                                points=end_position_np)
                        if transformed_s_pos is None or transformed_e_pos is None:
                            continue
                        
                        #calculate distance (/100 as it is in cm) and time, calculated with frame rate
                        distance_covered = measure_distance(transformed_s_pos[0],transformed_e_pos[0])/100
                        time = (last_frame-frame_num)/frame_rate
                        
                        speed_mps = distance_covered/time
                        speed_kph = speed_mps*3.6

                        #tracking data of players
                        if object not in speed_distance_dict:
                            speed_distance_dict[object] = {}

                        if track_id not in speed_distance_dict[object]:
                            speed_distance_dict[object][track_id] = {
                                "team": team,  
                                "total_distance": 0,  
                                "data": []  
                            }

                        speed_distance_dict[object][track_id]["total_distance"] += distance_covered

                        speed_distance_dict[object][track_id]["data"].append({
                            "frame": frame_num,
                            "speed": speed_kph,
                            "distance": distance_covered
                        })

                return speed_distance_dict


            else:
                continue