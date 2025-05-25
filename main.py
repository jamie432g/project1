from utils import read_video,save_video, save_speed_distance_to_csv
from tracker import Tracker
from colour_assignment.Assign import AssignColour
from annotate.annotate_draw import AnnotationDrawer
from keypoints import Keypoints
from SpeedDistance import SpeedDistanceEstimation
from ball_interpolation import Ball_Interpolate
from possession import Possession
from action_detection import ActionDection
import cv2
def main():
    #read video
    input_video = "input_videos/0bfacc_0.mp4"
    video_frames = read_video(input_video)

    #initialise tracker
    tracker = Tracker("models/best.pt")
    tracks = tracker.generate_tracking_data(video_frames, read_from_path= True, stub_path="stubs/track_stubs_78.pkl")

    #initialise ball interpolation
    interpolate = Ball_Interpolate()
    tracks["ball"] = interpolate.interpolate_ball_positions(tracks["ball"])

    #find keypoints
    keypoints = Keypoints("models/best_kp_augments.pt")
    edges,points,pitch_rp,frame_rp = keypoints.video_points(video_frames)

    #assign teams
    team_assigner = AssignColour()
    team_assigner.assign_teams_to_tracks(video_frames, tracks)


    #possession
    possession = Possession()
    control_array = possession.track_possession(tracks=tracks)

    #find speed and distance using transformed points
    speed_distance = SpeedDistanceEstimation()
    speed_dist_dict = speed_distance.calculate_speed_distance(tracks,pitch_rp,frame_rp)

    #draw annotations
    annotation_drawer = AnnotationDrawer()
    output_vf,output_pitch = annotation_drawer.draw_annotations(video_frames,tracks,edges,points,pitch_rp,frame_rp,control_array,show_keypoints = True)
    annotation_drawer.generate_pitch_heatmap()

    #save video
    save_video(output_vf,"output_videos/outputDemo.avi")
    save_video(output_pitch,"output_videos/pitch.avi")
    save_speed_distance_to_csv(speed_dist_dict)

    #action detection
    detectAction = ActionDection()
    #takes a start time and finds the actions from two seconds of that clip
    detectAction.action_detect(start_time=0,video_path=input_video)

if __name__ == "__main__":
    main()