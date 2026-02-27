import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Test multiple preprocessing approaches on the cracked image
img = Image.open(r'D:\Orb\data\finance\screenshots\20260225_223801_1000074760.jpg')

print(f"Image size: {img.size}")
print(f"Image mode: {img.mode}")

# Approach 1: Basic grayscale
gray = img.convert("L")
text1 = pytesseract.image_to_string(gray, lang="eng")
print(f"\n=== Approach 1: Grayscale ({len(text1)} chars) ===")
print(text1[:400])

# Approach 2: Threshold (binarize) - push noise to white
gray2 = img.convert("L")
thresh = gray2.point(lambda x: 255 if x > 150 else 0)
text2 = pytesseract.image_to_string(thresh, lang="eng")
print(f"\n=== Approach 2: Threshold 150 ({len(text2)} chars) ===")
print(text2[:400])

# Approach 3: High contrast + sharpen
enhancer = ImageEnhance.Contrast(gray)
high_contrast = enhancer.enhance(2.0)
sharp = high_contrast.filter(ImageFilter.SHARPEN)
text3 = pytesseract.image_to_string(sharp, lang="eng")
print(f"\n=== Approach 3: High contrast + sharpen ({len(text3)} chars) ===")
print(text3[:400])

# Approach 4: PSM 6 (assume single block of text)
custom_config = r'--oem 3 --psm 6'
text4 = pytesseract.image_to_string(thresh, lang="eng", config=custom_config)
print(f"\n=== Approach 4: Threshold + PSM 6 ({len(text4)} chars) ===")
print(text4[:400])
