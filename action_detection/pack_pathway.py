import torch
from torch import nn
#PackPathway method from pytorch documentation on slowfast, edited for GPU purposes and inference
slowfast_alpha = 4
class PackPathway(torch.nn.Module):
    def __init__(self, alpha=4):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames):

        device = frames.device  
        fast_pathway = frames  

        slow_indices = torch.linspace(0, frames.shape[2] - 1, frames.shape[2] // self.alpha, device=device).long()

        slow_pathway = torch.index_select(frames, 2, slow_indices)  


        return [slow_pathway, fast_pathway]