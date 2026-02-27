import sys
sys.path.insert(0, r'D:\Orb')

from app.finance.services.screenshot_ocr_deterministic import extract_from_image

# Test 1: Cracked screen
print("=" * 50)
print("TEST 1: CRACKED SCREEN")
print("=" * 50)
r1 = extract_from_image(r'D:\Orb\data\finance\screenshots\20260225_223801_1000074760.jpg')
print(f"Valid: {r1.is_valid} | Confidence: {r1.confidence}%")
print(f"Extracted: {r1.fields_extracted}")
print(f"Missing: {r1.fields_missing}")
print(f"Date: {r1.work_date} | User: {r1.user_id} | Tour: {r1.tour_id}")
print(f"Start: {r1.start_time} | End: {r1.end_time} | Duration: {r1.duration_hours}h")
print(f"Parcels: {r1.delivery_count} | Stops: {r1.attempted_stops}")
print(f"Deliveries: {r1.deliveries} | Collections: {r1.collections}")
print(f"Lockers: {r1.lockers} | Not Attempted: {r1.not_attempted}")

# Test 2: Clean screenshot
print()
print("=" * 50)
print("TEST 2: CLEAN SCREENSHOT")
print("=" * 50)
r2 = extract_from_image(r'D:\Orb\data\finance\screenshots\20260226_182808_Screenshot_20251118-203509.png')
print(f"Valid: {r2.is_valid} | Confidence: {r2.confidence}%")
print(f"Extracted: {r2.fields_extracted}")
print(f"Missing: {r2.fields_missing}")
print(f"Date: {r2.work_date} | User: {r2.user_id} | Tour: {r2.tour_id}")
print(f"Start: {r2.start_time} | End: {r2.end_time} | Duration: {r2.duration_hours}h")
print(f"Parcels: {r2.delivery_count} | Stops: {r2.attempted_stops}")
print(f"Deliveries: {r2.deliveries} | Collections: {r2.collections}")
print(f"Lockers: {r2.lockers} | Not Attempted: {r2.not_attempted}")

# Test 3: Another screenshot
print()
print("=" * 50)
print("TEST 3: ANOTHER SCREENSHOT")
print("=" * 50)
r3 = extract_from_image(r'D:\Orb\data\finance\screenshots\20260226_191115_Screenshot_20251120-212529.png')
print(f"Valid: {r3.is_valid} | Confidence: {r3.confidence}%")
print(f"Extracted: {r3.fields_extracted}")
print(f"Missing: {r3.fields_missing}")
print(f"Date: {r3.work_date} | User: {r3.user_id} | Tour: {r3.tour_id}")
print(f"Start: {r3.start_time} | End: {r3.end_time} | Duration: {r3.duration_hours}h")
print(f"Parcels: {r3.delivery_count} | Stops: {r3.attempted_stops}")
