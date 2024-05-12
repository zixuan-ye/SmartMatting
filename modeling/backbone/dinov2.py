from .backbone import Backbone
import torch

class DINOv2(Backbone):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    def forward(self, image):
        return self.model(image)