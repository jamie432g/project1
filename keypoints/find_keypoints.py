from ultralytics import YOLO
import supervision as sv
import numpy as np
from sports.configs.soccer import SoccerPitchConfiguration

from ViewTransform.ViewTransformer import ViewTransformer

class Keypoints:
    def __init__(self,model_path):
        self.model_kp = YOLO(model_path)
        self.CONFIG = SoccerPitchConfiguration()
        self.prev_kp = []

    def smooth_kp(self, current_points, window_size = 5):
        if len(self.prev_kp) > 0 and self.prev_kp[-1].shape != current_points.shape:
            self.prev_kp = []
        self.prev_kp.append(current_points)
        if len(self.prev_kp) > window_size:
            self.prev_kp.pop(0)
        return np.mean(self.prev_kp, axis=0)

    def find_points(self,frame):
        #detect keypoints using the key points model and apply homography based on that
        result = self.model_kp.predict(frame,conf=0.5)

        #detected keypoints + confidence scores are detected
        pitch_keypoints = result[0].keypoints.cpu().numpy()
        pitch_conf_levels = pitch_keypoints.conf.astype(np.float32)

        #filter keypoint confidence, so only key points > 0.5 are used.
        filtered = pitch_conf_levels[0] > 0.5

        #reference points = keypoints without the key points below 0.5
        reference_points = pitch_keypoints.xy[0][filtered]

        if reference_points.shape[0] >= 4:
            smoothed_rp = self.smooth_kp(reference_points)
        else:
            smoothed_rp = reference_points

        #supervision rp = keypoints in supervision format
        supervision_rp = sv.KeyPoints(xy=smoothed_rp[np.newaxis, ...])

        #frame reference points from the imported pitch view
        frame_rp = np.array(self.CONFIG.vertices)[filtered]

        #homography applied for key points.
        view_transformer = ViewTransformer()

        pitch_all_points = np.array(self.CONFIG.vertices)
        frame_all_points = view_transformer.homography_transform(source=frame_rp,target=reference_points,points=pitch_all_points)

        #returns transformed keypoints and reference points
        return sv.KeyPoints(xy=frame_all_points[np.newaxis, ...]),supervision_rp, reference_points,frame_rp
    
    def video_points(self,video_frames):
        output_edges = {}
        output_points = {}
        output_rp = {}
        output_frame_rp = {}
        for frame_num,frame in enumerate(video_frames):
            edges,points,rp,frame_rp = self.find_points(frame)
            #output_edges relates to the transformed keypoints from the detected key points with the predefined pitch model
            output_edges[frame_num] = edges

            #output points relates to the YOLO detected points in supervision format
            output_points[frame_num] = points

            #output rp are the actual reference points from YOLO detected points
            output_rp[frame_num] = rp
            
            #the keypoints from the pitch model
            output_frame_rp[frame_num] = frame_rp
            
        return output_edges,output_points,output_rp,output_frame_rp


