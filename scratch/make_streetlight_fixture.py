from PIL import Image, ImageDraw

def draw_streetlight_image():
    # 400x400 night outdoor scene
    img = Image.new("RGB", (400, 400), color=(15, 25, 45)) # Dark night sky
    draw = ImageDraw.Draw(img)
    # Dark road at bottom
    draw.rectangle([0, 320, 400, 400], fill=(40, 40, 40))
    # Curb / sidewalk
    draw.line([0, 320, 400, 320], fill=(90, 90, 90), width=4)
    # Tall metal streetlight pole
    draw.rectangle([185, 70, 195, 330], fill=(130, 140, 150))
    # Street lamp curved arm
    draw.line([190, 80, 230, 60], fill=(130, 140, 150), width=6)
    # Street lamp fixture
    draw.ellipse([220, 50, 260, 75], fill=(200, 200, 200), outline=(100, 100, 100))
    # Street lamp light glow
    draw.ellipse([215, 55, 265, 85], fill=(255, 255, 210))
    return img

img = draw_streetlight_image()
img.save("test_fixtures/drawn_streetlight.jpg")
print("Saved drawn streetlight fixture!")
