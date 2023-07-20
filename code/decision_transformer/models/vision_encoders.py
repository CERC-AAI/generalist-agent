import torch
import clip

class VisionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device)

    def forward(self, observations):
        image_tensor = observations.to(self.device)
        image_features = self.model.encode_image(image_tensor)
        return image_features