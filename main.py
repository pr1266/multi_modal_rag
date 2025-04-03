import clip
import torch
from PIL import Image

model, preprocess = clip.load("ViT-B/32", device = "gpu")