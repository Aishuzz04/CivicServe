import os
import sys
sys.path.insert(0, ".")
from PIL import Image
import scratch.test_relevance_scenarios as t

fixtures = [
    ("test_fixtures/real_road_pothole.jpg", "Roads", "Deep pothole on main road", "Hazardous pothole cracking asphalt"),
    ("test_fixtures/real_garbage_waste.jpg", "Waste Management", "Overflowing garbage dump", "Trash pile accumulating on sidewalk"),
    ("test_fixtures/real_streetlight.jpg", "Street Lights", "Broken streetlight fixture", "Streetlight pole lamp fixture"),
    ("test_fixtures/unrelated_dog_pet.jpg", "Roads", "Road problem", "Road broken here"),
    ("test_fixtures/unrelated_food.jpg", "Waste Management", "Waste cleanup needed", "Please clean this up"),
    # Mismatch category: real pothole image submitted for Street Lights
    ("test_fixtures/real_road_pothole.jpg", "Street Lights", "Light not working", "Pole is dark")
]

for img_path, cat, title, desc in fixtures:
    if os.path.exists(img_path):
        img = Image.open(img_path)
        res = t.evaluate_image_relevance(img, cat, title, desc)
        print(f"\nImage: {os.path.basename(img_path)} | Category: {cat}")
        print(f"  Decision: {res['status']} (Score: {res['score']}%)")
        print(f"  Top Category: {res['top_cat']} ({res['top_cat_prob']}%) | Target Prob: {res['target_prob']}% | Top Neg: {res['top_neg']} ({res['top_neg_prob']}%)")
        print(f"  Message: {res['message']}")
