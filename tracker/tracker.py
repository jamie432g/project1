from ultralytics import YOLO
import supervision as sv
import pickle
import os

class Tracker:       
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        detections = []
        for frame in frames:  
            detection = self.model.predict(frame, conf=0.3)  
            detections.append(detection[0])  
        return detections
    
    def process_detections(self, detections):
        tracks = {"players": [], "referees": [], "ball": [], "goalkeepers":[]}
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            class_name_to_id = {v: k for k, v in cls_names.items()}
            sv_detect = sv.Detections.from_ultralytics(detection)
            
            sv_detect = sv_detect.with_nms(threshold=0.3,class_agnostic=True)
            
            tracks_detection = self.tracker.update_with_detections(sv_detect)
            self.store_tracks(tracks, frame_num, tracks_detection, sv_detect, class_name_to_id)
        
        return tracks

    
    def store_tracks(self, tracks, frame_num, tracks_detection, sv_detect, class_name_to_id):
        tracks["players"].append({})
        tracks["referees"].append({})
        tracks["ball"].append({})
        tracks["goalkeepers"].append({})

        player_frame, referee_frame, goalkeeper_frame, ball_frame = (
            tracks["players"][frame_num],
            tracks["referees"][frame_num],
            tracks["goalkeepers"][frame_num],
            tracks["ball"][frame_num],
        )

        for i in range(len(tracks_detection.xyxy)):  # Iterate using index
            bbox = tracks_detection.xyxy[i].tolist()
            cls_id = tracks_detection.class_id[i]
            track_id = tracks_detection.tracker_id[i]

            if cls_id == class_name_to_id["Player"]:
                player_frame[track_id] = {"bbox": bbox}
            elif cls_id == class_name_to_id["Referee"]:
                referee_frame[track_id] = {"bbox": bbox}
            elif cls_id == class_name_to_id["Goalkeeper"]:
                goalkeeper_frame[track_id] = {"bbox": bbox}

        for i in range(len(sv_detect.xyxy)):  # Iterate using index
            bbox = sv_detect.xyxy[i].tolist()
            cls_id = int(sv_detect.class_id[i])

            if cls_id == class_name_to_id["Ball"]:
                ball_frame[1] = {"bbox": bbox}

    
    def generate_tracking_data(self, frames, read_from_path=False, stub_path=None):
        if read_from_path and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)
        
        detections = self.detect_frames(frames)
        tracks = self.process_detections(detections)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks
