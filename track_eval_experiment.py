import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv

import cv2
from utils import read_video,save_video, save_speed_distance_to_csv
from tracker import Tracker

def read_frames(path):
    if os.path.isdir(path):
        imgs = sorted(f for f in os.listdir(path)
                      if f.lower().endswith((".jpg","png","jpeg")))
        return [cv2.imread(os.path.join(path, f)) for f in imgs]
    else:
        return read_video(path)

def save_trackeval_csv(tracks, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # iterate per‐frame
        for frame_idx, (p_frame, g_frame) in enumerate(
                zip(tracks["players"], tracks["goalkeepers"]), start=1):
            
            #players + keeper
            for cls_frame in (p_frame, g_frame):
                for track_id, obj in cls_frame.items():
                    x1, y1, x2, y2 = obj["bbox"]
                    w, h = x2 - x1, y2 - y1
                    # conf doesn't matter here so we set 0, and -1,-1,-1 for 3D coords
                    writer.writerow([
                        frame_idx,
                        track_id,
                        int(x1), int(y1),
                        int(w),  int(h),
                        0,   
                        -1, -1, -1
                    ])

def main():
    #read video, done for each video
    input_video = "experiments/img1"
    output_csv = "v_2QhNRucNC7E_c017.csv"


    video_frames = read_frames(input_video)

    #initialise tracker
    tracker = Tracker("models/best.pt")
    tracks  = tracker.generate_tracking_data(
                  video_frames,
                  read_from_path=False,
                  stub_path="stubs/track_stubs_6.pkl"
              )

    #save in correct format
    save_trackeval_csv(tracks, output_csv)


if __name__ == "__main__":
    main()