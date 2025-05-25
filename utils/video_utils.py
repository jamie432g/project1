import cv2

def read_video(video_path):
    #take in file path and return a list of frames
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened:
        print ("Error: Camera not opened")
        exit()
    frames = []
    while True:
        ret,frame = vidcap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames,output_video_path):
    #save a video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (width,height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
