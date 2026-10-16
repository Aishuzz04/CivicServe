from transformers import pipeline
from PIL import Image

pipe = pipeline("image-classification", model="umm-maybe/AI-image-detector")

test_paths = [
    "test_fixtures/real_road_pothole.jpg",
    "test_fixtures/real_garbage_waste.jpg",
    "test_fixtures/drawn_streetlight.jpg",
    "static/images/cities.jpg"
]

for p in test_paths:
    img = Image.open(p)
    res = pipe(img)
    print(f"\nImage: {p}")
    for item in res:
        print(f"  {item['label']}: {item['score']*100:.2f}%")
