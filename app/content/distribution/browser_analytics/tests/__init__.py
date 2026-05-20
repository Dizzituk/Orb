# Test harness for the browser_analytics module.
#
# Run from D:\Orb:
#   .\.venv\Scripts\python.exe app\content\distribution\browser_analytics\tests\test_scrape.py
#
# Covers:
#   - Number parsers (23 edge cases)
#   - TikTok parser on synthetic + real recon files
#   - ChannelAnalytics DB round-trip (insert/query/delete)
#   - Router endpoint inventory
#   - Parser registry dispatch
