import timm
from torch import nn


def get_vision_embedder(name: str, embed_size: int = 256):
    model = timm.create_model(name, pretrained=True)
    model.head = nn.Sequential(nn.Linear(384, embed_size))
    return model
