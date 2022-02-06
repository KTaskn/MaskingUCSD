import torch
import torch.nn as nn
from i3d import InceptionI3d

I3D_PRETRAINED_PATH = "/workspace/MILForVideos/extractors/i3d/rgb_i3d_pretrained.pt"
class WrapperI3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.i3d = InceptionI3d()
        self.i3d.load_state_dict(torch.load(I3D_PRETRAINED_PATH))

    def forward(self, video):
        video = video.transpose_(1, 2)
        return self.i3d(video).squeeze(2).unsqueeze(1)

class WrapperResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

    def forward(self, images):
        return self.resnet(images).unsqueeze(1)