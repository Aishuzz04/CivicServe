import urllib.request
import json

# Fetch a sample from HuggingFace dataset viewer API or raw repo file
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.png"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        with open("test_fixtures/ai_generated_astronaut.png", "wb") as f:
            f.write(resp.read())
    print("Downloaded Stable Diffusion sample: ai_generated_astronaut.png")
except Exception as e:
    print("Error:", e)
