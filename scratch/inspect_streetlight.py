import os
import sys
sys.path.insert(0, ".")
from PIL import Image
import torch
import scratch.test_relevance_scenarios as t

img = Image.open("test_fixtures/real_streetlight.jpg")
print("Image size:", img.size)

prompts = [
    "a street light pole",
    "a street light",
    "a lamp post",
    "an outdoor street lantern",
    "a night photo of city street light",
    "a city street light pole against the sky",
    "a building",
    "a tree",
    "a person"
]
inputs = t.clip_processor(text=prompts, images=img, return_tensors="pt", padding=True)
with torch.no_grad():
    probs = t.clip_model(**inputs).logits_per_image.softmax(dim=1)[0]

for p, prob in zip(prompts, probs):
    print(f"{p}: {prob.item()*100:.2f}%")
