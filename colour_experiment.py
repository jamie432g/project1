import os
import glob
import cv2
import pandas as pd
import numpy as np
from colour_assignment.Assign_test import AssignColourTest
from sklearn.metrics import precision_recall_fscore_support

def load_annotations(txt_path):
    #convert txt to CSV-like format
    df = pd.read_csv(
        txt_path,
        header=None,
        sep=',',
        names=['frame','player_id','x','y','w','h','a','b','c','team']
    )
    #on the zero-index frame 
    df['frame'] = df['frame'].astype(int) - 1
    #convert x,y,w,h -> x1,y1,x2,y2
    df['x2'] = df['x'] + df['w']
    df['y2'] = df['y'] + df['h']
    #didn't read the team labels thought they were 1/2, but were 0/1 -> just shifted them
    df['team'] = df['team'].astype(int) + 1
    return df[['frame','player_id','x','y','x2','y2','team']]

def group_detections(df):
    #put things into a detection like format (no confidence since YOLO wasn't run)
    grouped = {}
    for _, row in df.iterrows():
        f = int(row.frame)
        det = {'player_id': int(row.player_id), 'bbox': [row.x, row.y, row.x2, row.y2]}
        grouped.setdefault(f, []).append(det)
    return grouped

def load_frames(img_folder):
    #loaded with RGB, but didn't change anything, thought there was a bug but this changing to rgb doesn't really do anything
    paths = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
    frames = []
    for p in paths:
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        # convert BGR (OpenCV default) to RGB for consistent colour
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb)
    return frames

def evaluate_overall(all_assigned, df):
    #comparing ground truth and predicted labels
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        f, pid, true = int(row.frame), int(row.player_id), int(row.team)
        pred = next((d['team'] for d in all_assigned.get(f, []) if d['player_id'] == pid), None)
        if pred is not None:
            y_true.append(true)
            y_pred.append(pred)
    #metrics
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1,2], average='macro'
    )
    print("\n=== Overall Metrics ===")
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
def main(gt_path, img_folder):
    #gt and "detections" calculated per frame
    df = load_annotations(gt_path)
    dets = group_detections(df)

    #framess loaded
    frames = load_frames(img_folder)

    # clustering fit on the first frame
    assigner = AssignColourTest(random_state=43)
    assigner.fit(frames[0], dets[0])


    #predict on each frame from the clusters calculated in the first frame
    all_assigned = {}
    for f_idx, frame in enumerate(frames):
        det_list = dets.get(f_idx, [])
        assigned = assigner.predict(frame, det_list)
        all_assigned[f_idx] = assigned

    #evaluation function run
    evaluate_overall(all_assigned, df)

if __name__ == '__main__':
    GT = 'experiments/v_ITo3sCnpw_k_c010/gt/gt.txt'
    IMG_FOLDER = 'experiments/img5'
    main(GT, IMG_FOLDER)
