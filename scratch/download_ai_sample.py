import urllib.request
import os

# Wikimedia commons has known AI generated images (tagged category: AI-generated images)
# For example: Midjourney/Stable Diffusion generated images uploaded to Wikimedia
ai_image_url = "https://upload.wikimedia.org/wikipedia/commons/4/4b/An_astronaut_riding_a_green_horse_-_Stable_Diffusion.jpg"
headers = {"User-Agent": "Mozilla/5.0"}

try:
    req = urllib.request.Request(ai_image_url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        with open("test_fixtures/ai_generated_sample.jpg", "wb") as f:
            f.write(resp.read())
    print("Downloaded AI image successfully!")
except Exception as e:
    print("Failed to download:", e)
