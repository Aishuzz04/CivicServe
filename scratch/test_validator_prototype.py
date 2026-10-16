import os
import io
import torch
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel, pipeline

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Loading AI Image Detector...")
ai_pipe = pipeline("image-classification", model="umm-maybe/AI-image-detector")

def create_sample_road_image():
    # Gray asphalt with dark irregular patch (pothole)
    img = Image.new("RGB", (300, 300), color=(80, 80, 80))
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 100, 220, 200], fill=(30, 30, 30), outline=(20, 20, 20), width=3)
    draw.line([50, 150, 250, 160], fill=(20, 20, 20), width=2)
    return img

def create_sample_unrelated_image():
    # Bright rainbow pattern resembling graphic artwork
    img = Image.new("RGB", (300, 300), color=(255, 100, 100))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 250, 250], fill=(100, 255, 100))
    draw.text((80, 140), "CARTOON MEME", fill=(0, 0, 0))
    return img

print("\n--- Testing Sample Road Image ---")
road_img = create_sample_road_image()
candidate_labels = [
    "a photo of a road with potholes, cracked asphalt, or road surface damage",
    "a photo of overflowing garbage or waste dump",
    "a photo of a broken street light pole",
    "a photo of an indoor living room",
    "a selfie photo of a person",
    "a cartoon, graphic illustration, or meme"
]
inputs = clip_processor(text=candidate_labels, images=road_img, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]

for l, p in zip(candidate_labels, probs):
    print(f"  {p.item()*100:.2f}% : {l}")

ai_res = ai_pipe(road_img)
print("AI Detector result:", ai_res)

print("\n--- Testing Sample Unrelated Image ---")
unrel_img = create_sample_unrelated_image()
inputs = clip_processor(text=candidate_labels, images=unrel_img, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]

for l, p in zip(candidate_labels, probs):
    print(f"  {p.item()*100:.2f}% : {l}")
