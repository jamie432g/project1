import cv2
from matplotlib import pyplot as plt
import numpy as np
from annotate.draw_shapes import draw_ellipse, draw_triangle
from utils import get_bottom_centre, convert_rgb_to_bgr, get_bottom_centre_float
import supervision as sv
from sports.configs.soccer import SoccerPitchConfiguration
from ViewTransform.ViewTransformer import ViewTransformer
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

class AnnotationDrawerTest:
    def __init__(self, show_2d_view: bool = True, show_possession: bool = False):
        self.CONFIG = SoccerPitchConfiguration()
        self.goalkeeper_colours = {}
        self.referee_ids = set()
        self.player_positions_hm = {}
        self.show_2d_view = show_2d_view
        self.show_possession = show_possession

    def draw_annotations(self, 
                         video_frames,
                         tracks,
                         edges_dict,
                         points_dict,
                         rp_dict,
                         frame_rp_dict,
                         team_ball_control,
                         show_keypoints: bool = True):
        """
        Draw annotations on video frames and optional 2D pitch overlay.

        Args:
            video_frames: list of frames (BGR images)
            tracks: dict with 'players', 'ball', 'referees', 'goalkeepers'
            edges_dict, points_dict: keypoint edge/vertex data
            rp_dict, frame_rp_dict: reference point homography data
            team_ball_control: array indicating possession per frame (1 or 2)
            show_keypoints: toggle keypoint drawing
        """
        # Initialize annotators if needed
        if show_keypoints:
            vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex("#FF1493"), radius=8)
            edge_annotator = sv.EdgeAnnotator(color=sv.Color.from_hex("00BFFF"), edges=self.CONFIG.edges)

        output_vf = []
        output_pitch = []

        for frame_num, frame in enumerate(video_frames):
            view_transform = ViewTransformer()
            pitch = draw_pitch(config=self.CONFIG)
            frame = frame.copy()

            # Keypoints
            if show_keypoints:
                frame = edge_annotator.annotate(frame, edges_dict.get(frame_num))
                frame = vertex_annotator.annotate(frame, points_dict.get(frame_num))

            # Retrieve detections
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            goalkeeper_dict = tracks["goalkeepers"][frame_num]

            # Draw goalkeepers
            for track_id, keeper in goalkeeper_dict.items():
                colour = keeper.get('team_colour', (0, 0, 0))
                colour_store = tuple(map(int, colour))
                self.goalkeeper_colours.setdefault(track_id, []).append(colour_store)

                frame = draw_ellipse(frame, keeper["bbox"], colour, track_id)
                pos = get_bottom_centre(keeper["bbox"])
                src_pt = np.array(pos, dtype=np.float32).reshape(-1, 2)
                tgt_pt = view_transform.homography_transform(
                    source=rp_dict.get(frame_num),
                    target=frame_rp_dict.get(frame_num),
                    points=src_pt
                )
                pitch = draw_points_on_pitch(
                    config=self.CONFIG,
                    xy=tgt_pt,
                    face_color=sv.Color(*convert_rgb_to_bgr(colour)),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=pitch
                )

            # Draw referees
            for track_id, referee in referee_dict.items():
                self.referee_ids.add(track_id)
                frame = draw_ellipse(frame, referee["bbox"], (0, 255, 255))
                pos = get_bottom_centre(referee["bbox"])
                src_pt = np.array(pos, dtype=np.float32).reshape(-1, 2)
                tgt_pt = view_transform.homography_transform(
                    source=rp_dict.get(frame_num),
                    target=frame_rp_dict.get(frame_num),
                    points=src_pt
                )
                pitch = draw_points_on_pitch(
                    config=self.CONFIG,
                    xy=tgt_pt,
                    face_color=sv.Color.GREEN,
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=pitch
                )

            # Draw players
            for track_id, player in player_dict.items():
                if track_id in self.referee_ids:
                    continue

                colour = (player.get('team_colour', (0, 0, 0))
                          if track_id not in self.goalkeeper_colours
                          else max(set(self.goalkeeper_colours[track_id]),
                                   key=self.goalkeeper_colours[track_id].count))
                frame = draw_ellipse(frame, player["bbox"], colour, track_id)
                pos = get_bottom_centre_float(player["bbox"])
                src_pt = np.array(pos, dtype=np.float32).reshape(-1, 2)
                tgt_pt = view_transform.homography_transform(
                    source=rp_dict.get(frame_num),
                    target=frame_rp_dict.get(frame_num),
                    points=src_pt
                )
                pitch = draw_points_on_pitch(
                    config=self.CONFIG,
                    xy=tgt_pt,
                    face_color=sv.Color(*convert_rgb_to_bgr(colour)),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=pitch
                )
                self.player_positions_hm.setdefault(track_id, []).append(tgt_pt[0])

            # Draw ball
            for ball in ball_dict.values():
                frame = draw_triangle(frame, ball["bbox"], (0, 255, 0))
                pos = get_bottom_centre(ball["bbox"])
                src_pt = np.array(pos, dtype=np.float32).reshape(-1, 2)
                tgt_pt = view_transform.homography_transform(
                    source=rp_dict.get(frame_num),
                    target=frame_rp_dict.get(frame_num),
                    points=src_pt
                )
                pitch = draw_points_on_pitch(
                    config=self.CONFIG,
                    xy=tgt_pt,
                    face_color=sv.Color.WHITE,
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    pitch=pitch
                )

                # Draw possession bar only if enabled
                if self.show_possession:
                    frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            # Overlay 2D pitch view if enabled
            if self.show_2d_view:
                pitch_h, pitch_w = pitch.shape[:2]
                frame_h, frame_w = frame.shape[:2]
                # Resize overlay to 30% width and 25% height
                target_w = int(frame_w * 0.3)
                target_h = int(frame_h * 0.25)
                resized = cv2.resize(pitch, (target_w, target_h))
                x_off = (frame_w - target_w) // 2
                y_off = frame_h - target_h - 10
                overlay = frame.copy()
                overlay[y_off:y_off+target_h, x_off:x_off+target_w] = resized
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            output_vf.append(frame)
            output_pitch.append(pitch)

        return output_vf, output_pitch

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        if frame_num < 0 or frame_num >= len(team_ball_control):
            return frame
        # Compute possession stats
        control = np.array(team_ball_control[:frame_num+1])
        count1 = np.sum(control == 1)
        count2 = np.sum(control == 2)
        total = count1 + count2
        if total == 0:
            pct1 = pct2 = 0.5
        else:
            pct1, pct2 = count1/total, count2/total

        h, w = frame.shape[:2]
        x_start = int(w * 0.25)
        x_end = int(w * 0.75)
        y_start = 50
        bar_h = 30
        width1 = int((x_end - x_start) * pct1)

        overlay = frame.copy()
        # Background box for text
        txt_y0 = y_start + bar_h + 10
        txt_y1 = txt_y0 + 40
        cv2.rectangle(overlay, (x_start, txt_y0), (x_end, txt_y1), (255, 255, 255), -1)
        # Bars
        cv2.rectangle(overlay, (x_start, y_start), (x_start+width1, y_start+bar_h), (0, 0, 255), -1)
        cv2.rectangle(overlay, (x_start+width1, y_start), (x_end, y_start+bar_h), (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Text
        text_y = txt_y0 + 30
        cv2.putText(frame, f"Team 1: {pct1*100:.2f}%", (x_start+20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 2: {pct2*100:.2f}%", (x_end-240, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return frame

    def generate_pitch_heatmap(self, save_path="heatmaps/pitch_heatmap.png"):
        if not self.player_positions_hm:
            print("No player positions recorded. Heatmap cannot be generated.")
            return
        count = 0
        for track_id, positions in self.player_positions_hm.items():
            if count >= 5:
                break
            # Prepare data
            pts = np.array([pos.tolist() if isinstance(pos, np.ndarray) else pos for pos in positions], dtype=np.float32)
            x, y = pts[:,0], pts[:,1]
            pitch = draw_pitch(config=self.CONFIG)
            h, w, _ = pitch.shape
            x = np.clip(x, 0, self.CONFIG.length-1)
            y = np.clip(y, 0, self.CONFIG.width-1)
            heatmap, _, _ = np.histogram2d(x, y, bins=[120,70], range=[[0,self.CONFIG.length],[0,self.CONFIG.width]])
            heatmap = np.log1p(heatmap)
            mx = heatmap.max()
            heatmap = heatmap/mx if mx>0 else heatmap+1
            hm_resized = cv2.resize(heatmap.T, (w, h))
            hm_colored = cv2.applyColorMap((hm_resized*255).astype(np.uint8), cv2.COLORMAP_JET)
            final = cv2.addWeighted(hm_colored, 0.7, pitch, 0.3, 0)
            path = save_path.replace('.png','') + f"_player{track_id}.png"
            cv2.imwrite(path, final)
            count += 1
