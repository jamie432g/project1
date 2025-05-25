import torch
from torch import nn
from torchvision.transforms  import Compose, Lambda, CenterCrop
from pytorchvideo.transforms import (ApplyTransformToKey, UniformTemporalSubsample, Normalize, ShortSideScale)
from pytorchvideo.data.encoded_video import EncodedVideo
import random

from action_detection.pack_pathway import PackPathway

class ActionDection:
    def __init__(self):
        #annotations for multisport dataset
        self.football_action_labels = {
            0: "football shoot",
            1: "football long pass",
            2: "football short pass",
            3: "football through pass",
            4: "football cross",
            5: "football dribble",
            6: "football trap",
            7: "football throw",
            8: "football diving",
            9: "football tackle",
            10: "football steal",
            11: "football clearance",
            12: "football block",
            13: "football press",
            14: "football aerial duels"
        }
    def action_detect(self,start_time,video_path):
        if video_path is None:
            print ("No Video Path detected")
            return
        #Will detect actions 2 seconds from start time
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "slowfast_r50"
        model = torch.hub.load("facebookresearch/pytorchvideo",model=model_name, pretrained=True)
        num_classes = 15  
        #classifier layer changed
        model.blocks[6].proj = nn.Linear(2304, num_classes)  
        model.load_state_dict(torch.load("models/trained_model_epoch_3.pth",map_location=device))

        model = model.to(device).eval()

        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        num_frames = 32

        #inference trransformations
        inference_transform = ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(num_frames),  
                Lambda(lambda x: x / 255.0),  
                Normalize(mean, std),  
                ShortSideScale(size=side_size),  
                CenterCrop(side_size),  
            ])
        )
        video = EncodedVideo.from_path(video_path)

        #calculate end sec to get clip in question
        clip_duration = 2  
        end_sec = clip_duration + start_time
        video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)

        #apply transformations, as done in training
        video_data = inference_transform(video_data)

        #GPU ideal for this
        inputs = video_data["video"].unsqueeze(0).to(device)  

        #slowfast architecture processing (need a list of inputs, one for slow pathway and one for fast)
        pack_path = PackPathway()
        inputs = pack_path(inputs)

        #ensure inference only
        with torch.no_grad():
            outputs = model(inputs)

        #get top 3 predicted classes
        top_k = 3
        _, top_indices = torch.topk(outputs, top_k, dim=1)

        #using the action labels, convert the predictions into labels
        top_actions = [(self.football_action_labels[idx.item()]) for idx in top_indices[0]]

        #want it shuffled since we just want the actions, not a ranking of what's more present since that's subjective
        random.shuffle(top_actions)

        #display results
        print("Predicted Actions:")
        for _, action in enumerate(top_actions, 1):
            print("- ", action)
