import os
import urllib.request

os.makedirs("test_fixtures", exist_ok=True)

test_images = {
    # Unsplash public demo images
    "real_road_pothole.jpg": "https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?w=500&auto=format&fit=crop&q=60",
    "real_garbage_waste.jpg": "https://images.unsplash.com/photo-1605600659908-0ef719419d41?w=500&auto=format&fit=crop&q=60",
    "real_streetlight.jpg": "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=60",
    "unrelated_dog_pet.jpg": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=500&auto=format&fit=crop&q=60",
    "unrelated_food.jpg": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=500&auto=format&fit=crop&q=60"
}

headers = {"User-Agent": "Mozilla/5.0"}

for fname, url in test_images.items():
    fpath = os.path.join("test_fixtures", fname)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            with open(fpath, "wb") as f:
                f.write(data)
        print(f"Downloaded: {fname} ({len(data)} bytes)")
    except Exception as e:
        print(f"Failed to download {fname}: {e}")
