import cv2
from matplotlib import pyplot as plt
import numpy as np
from annotate.draw_shapes import draw_ellipse, draw_triangle
from utils import get_bottom_centre, convert_rgb_to_bgr,get_bottom_centre_float
import supervision as sv
from sports.configs.soccer import SoccerPitchConfiguration
from ViewTransform.ViewTransformer import ViewTransformer
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
class AnnotationDrawer:
    def __init__(self):
        self.CONFIG = SoccerPitchConfiguration()
        self.goalkeeper_colours = {}
        self.referee_ids = set()
        self.player_positions_hm = {}

    def draw_annotations(self, 
                        video_frames,
                        tracks,
                        edges_dict,
                        points_dict,
                        rp_dict,
                        frame_rp_dict,
                        team_ball_control,
                        show_keypoints = False):
        
        if show_keypoints:
            vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex("#FF1493"), radius=8)
            edge_annotator = sv.EdgeAnnotator(color=sv.Color.from_hex("00BFFF"),edges=self.CONFIG.edges)
        output_vf = []
        output_pitch = []
        
        for frame_num, frame in enumerate(video_frames):
            view_transform = ViewTransformer()
            pitch = draw_pitch(config=self.CONFIG)
            frame = frame.copy()
            if show_keypoints:
                frame = edge_annotator.annotate(frame, edges_dict.get(frame_num))
                frame = vertex_annotator.annotate(frame,points_dict.get(frame_num))
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            goalkeeper_dict = tracks["goalkeepers"][frame_num]

            #draw keepers, store their ids, store colour corresponding to id
            for track_id, keeper in goalkeeper_dict.items():
                colour = keeper.get('team_colour',(0,0,0))
                colour_store = tuple(map(int, keeper.get('team_colour', (0, 0, 0))))
                if track_id not in self.goalkeeper_colours:
                    self.goalkeeper_colours[track_id] = []
                self.goalkeeper_colours[track_id].append(colour_store)
                
                frame = draw_ellipse(frame,keeper["bbox"],colour,track_id)
                keeper_position = get_bottom_centre(keeper["bbox"])
                keeper_position_np = np.array(keeper_position,dtype=np.float32).reshape(-1,2)
                transformed_keeper_position_np = view_transform.homography_transform(source=rp_dict.get(frame_num),
                                                                                     target=frame_rp_dict.get(frame_num),
                                                                                     points=keeper_position_np)
                pitch = draw_points_on_pitch(config=self.CONFIG,
                                             xy=transformed_keeper_position_np,
                                             face_color=sv.Color(*convert_rgb_to_bgr(keeper.get('team_colour',(0,0,0)))),
                                             edge_color=sv.Color.BLACK,radius=16,
                                             pitch=pitch)
            
            #draw referee and their transformed position on the pitch
            for track_id, referee in referee_dict.items():
                self.referee_ids.add(track_id)
                frame = draw_ellipse(frame, referee["bbox"], (0, 255, 255))
                referee_position = get_bottom_centre(referee["bbox"])
                referee_position_np = np.array(referee_position,dtype=np.float32).reshape(-1,2)
                transformed_referee_position = view_transform.homography_transform(source=rp_dict.get(frame_num),
                                                                                   target=frame_rp_dict.get(frame_num),
                                                                                   points=referee_position_np)
                pitch = draw_points_on_pitch(config=self.CONFIG,
                                             xy=transformed_referee_position,
                                             face_color=sv.Color.GREEN,
                                             edge_color=sv.Color.BLACK,radius=16,
                                             pitch=pitch)

            #draw players and their transformed position on the pitch
            for track_id, player in player_dict.items():
                if track_id in self.goalkeeper_colours:
                    colour = max(set(self.goalkeeper_colours[track_id]), key=self.goalkeeper_colours[track_id].count)
                else:
                    colour = player.get('team_colour',(0,0,0))
                if track_id in self.referee_ids:
                    continue
                frame = draw_ellipse(frame,player["bbox"],colour,track_id)
                player_position = get_bottom_centre_float(player["bbox"])
                player_position_np = np.array(player_position,dtype=np.float32).reshape(-1,2)
                transformed_player_position_np = view_transform.homography_transform(source=rp_dict.get(frame_num),
                                                                                     target=frame_rp_dict.get(frame_num),
                                                                                     points=player_position_np)
                pitch = draw_points_on_pitch(config=self.CONFIG,
                                             xy=transformed_player_position_np,
                                             face_color=sv.Color(*convert_rgb_to_bgr(colour)),
                                             edge_color=sv.Color.BLACK,radius=16,
                                             pitch=pitch)
                #transformed position of each player is stored
                if track_id not in self.player_positions_hm:
                    self.player_positions_hm[track_id] = []  
                self.player_positions_hm[track_id].append(transformed_player_position_np[0])

            #draw ball and position on the h
            for _, ball in ball_dict.items():
                frame = draw_triangle(frame, ball["bbox"], (0, 255, 0))
                ball_position = get_bottom_centre(ball["bbox"])
                ball_position_np = np.array(ball_position,dtype=np.float32).reshape(-1,2)
                transformed_ball_position = view_transform.homography_transform(source=rp_dict.get(frame_num),
                                                                                target=frame_rp_dict.get(frame_num),
                                                                                points=ball_position_np)
                pitch = draw_points_on_pitch(config=self.CONFIG,
                                             xy=transformed_ball_position,
                                             face_color=sv.Color.WHITE,
                                             edge_color=sv.Color.BLACK,
                                             radius=10,
                                             pitch=pitch)
                
                frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            #draw pitch on frame
            pitch_height = int(frame.shape[0] * 0.25)  
            pitch_width = int(frame.shape[1] * 0.3)  
            pitch_resized = cv2.resize(pitch, (pitch_width, pitch_height))

            frame_h, frame_w, _ = frame.shape
            x_offset = (frame_w - pitch_width) // 2  
            y_offset = frame_h - pitch_height - 10  

            overlay = frame.copy()
            overlay[y_offset:y_offset + pitch_height, x_offset:x_offset + pitch_width] = pitch_resized
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)  


            output_pitch.append(pitch)
            output_vf.append(frame)

        return output_vf,output_pitch
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):

        #ensure frame validity
        if frame_num < 0 or frame_num >= len(team_ball_control):
            return frame

        #get possession of team up to said frame
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]

        #calculate possession frames
        team_1_num_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_num_frames = np.sum(team_ball_control_till_frame == 2)
        total_frames = team_1_num_frames + team_2_num_frames

        #no 0 division
        if total_frames == 0:
            team_1, team_2 = 0.5, 0.5  
        else:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames

        _, frame_width, _ = frame.shape

        #bar position, centred around the top
        bar_x_start = int(frame_width * 0.25)  
        bar_x_end = int(frame_width * 0.75)  
        bar_y_start = 50  
        bar_height = 30  

        #find the width of team 1 bar
        team_1_width = int((bar_x_end - bar_x_start) * team_1)

        overlay = frame.copy()

        #text background 
        text_box_y_start = bar_y_start + bar_height + 10
        text_box_y_end = text_box_y_start + 40
        cv2.rectangle(overlay, (bar_x_start, text_box_y_start), (bar_x_end, text_box_y_end), (255, 255, 255), -1)
        
        #possession bar drawn
        cv2.rectangle(overlay, (bar_x_start, bar_y_start), (bar_x_start + team_1_width, bar_y_start + bar_height), (0, 0, 255), -1)  
        cv2.rectangle(overlay, (bar_x_start + team_1_width, bar_y_start), (bar_x_end, bar_y_start + bar_height), (255, 0, 0), -1) 

        #transparency effect 
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        #positioning of text, changed to aesthetic
        text_y = text_box_y_start + 30  
        team_1_text_x = bar_x_start + 20
        team_2_text_x = bar_x_end - 240  

        #draw possession
        team_1_text = "Team 1: " + str(round(team_1 * 100, 2)) + "%"
        team_2_text = "Team 2: " + str(round(team_2 * 100, 2)) + "%"

        cv2.putText(frame, team_1_text, (team_1_text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, team_2_text, (team_2_text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return frame

    def generate_pitch_heatmap(self, save_path="heatmaps/pitch_heatmap.png"):
        counter = 0
        if not self.player_positions_hm:
            print("No player positions recorded. Heatmap cannot be generated.")
            return

        for track_id, positions in self.player_positions_hm.items():
            print("Heatmap generated for player", track_id)
            if counter == 5:
                break

            #since positions were stored as numpy arrays, convert to python arrays then store these in a numpy array
            positions = np.array([pos.tolist() if isinstance(pos, np.ndarray) else pos for pos in positions], dtype=np.float32)

            x_coords, y_coords = positions[:, 0], positions[:, 1]

            pitch = draw_pitch(config=self.CONFIG)
            pitch_height, pitch_width, _ = pitch.shape

            #coords according to the config dimensions 12000cm x 7000cm
            x_coords = np.clip(x_coords, 0, self.CONFIG.length - 1) 
            y_coords = np.clip(y_coords, 0, self.CONFIG.width - 1)   

            #histogram generated, bins with corresponding sizes
            heatmap, _, _ = np.histogram2d(
                x_coords, y_coords, 
                bins=[120, 70],  
                range=[[0, self.CONFIG.length], [0, self.CONFIG.width]]
            )

            #log scaling for when there are a lot of values
            heatmap = np.log1p(heatmap)  
            max_value = np.max(heatmap)

            if max_value == 0:
                heatmap += 1  
            else:
                heatmap /= max_value 

            #transposed after normalisation
            heatmap_resized = cv2.resize(heatmap.T, (pitch_width, pitch_height)) 

            heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

            #heatmap blended onto pitch
            heatmap_final = cv2.addWeighted(heatmap_colored, 0.7, pitch, 0.3, 0)  

            #save
            file_path = save_path.replace(".png", "") + "_player" + str(track_id) + ".png"
            cv2.imwrite(file_path, heatmap_final)

            counter += 1  #currently only processing one player, remove to do all players

