import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Test on a CLEAN screenshot (not cracked)
img = Image.open(r'D:\Orb\data\finance\screenshots\20260226_182808_Screenshot_20251118-203509.png')
print(f"Clean screenshot size: {img.size}")

# High contrast approach
gray = img.convert("L")
enhancer = ImageEnhance.Contrast(gray)
hc = enhancer.enhance(2.0)
sharp = hc.filter(ImageFilter.SHARPEN)
text = pytesseract.image_to_string(sharp, lang="eng")
print(f"\n=== High contrast + sharpen ({len(text)} chars) ===")
print(text[:600])
