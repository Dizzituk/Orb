import sys
sys.path.insert(0, r'D:\Orb')

from app.finance.services.screenshot_ocr_deterministic import extract_from_image

# Test on the cracked screen image
result = extract_from_image(r'D:\Orb\data\finance\screenshots\20260225_223801_1000074760.jpg')

print(f"Method: {result.method}")
print(f"Valid: {result.is_valid}")
print(f"Confidence: {result.confidence}%")
print(f"Fields extracted: {result.fields_extracted}")
print(f"Fields missing: {result.fields_missing}")
print()
print(f"Date: {result.work_date}")
print(f"User ID: {result.user_id}")
print(f"Tour ID: {result.tour_id}")
print(f"Start: {result.start_time}")
print(f"End: {result.end_time}")
print(f"Duration: {result.duration_hours}h")
print(f"Delivered Parcels: {result.delivery_count}")
print(f"Attempted Stops: {result.attempted_stops}")
print(f"Collections: {result.collections}")
print(f"Deliveries: {result.deliveries}")
print(f"Stores: {result.stores}")
print(f"Lockers: {result.lockers}")
print(f"Not Attempted: {result.not_attempted}")
print()
print("=== RAW TEXT ===")
print(result.raw_text[:500])
