import urllib.request
import os

url = "https://images.unsplash.com/photo-1563245372-f21724e3856d?w=500&auto=format&fit=crop&q=60"
headers = {"User-Agent": "Mozilla/5.0"}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=10) as resp:
    with open("test_fixtures/real_streetlight_actual.jpg", "wb") as f:
        f.write(resp.read())
print("Downloaded actual streetlight photo!")
