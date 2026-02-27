import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = Image.open(r'D:\Orb\data\test_screenshot.jpg')
# Copy the uploaded file first
