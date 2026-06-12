# Purpose: End-to-end tests for the browser_analytics module.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.content.distribution.browser_analytics, app.content.distribution.browser_analytics.models, app.content.distribution.browser_analytics.parsers, app.content.distribution.browser_analytics.parsers.common (+7 more)
# Last-renovated: 2026-06-11
"""
End-to-end tests for the browser_analytics module.

Runs without needing a backend restart:
  - Creates the channel_analytics table in the live DB if absent
  - Tests number parsers on edge cases
  - Tests the TikTok parser on the ACTUAL recon file on disk
  - Round-trips a ChannelAnalytics row (insert, query, verify)
  - Exercises the scrape-all dispatch logic without actually scraping

Run:  .\.venv\Scripts\python.exe _tmp_test_scrape.py
"""
import sys
import ast
import traceback
from pathlib import Path

sys.path.insert(0, 'D:/Orb')


# --- Counters -------------------------------------------------------

passed = 0
failed = 0
errors = []

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [OK]   {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        errors.append(f"{name}: {detail}")
        print(f"  [FAIL] {name}  ({detail})")


def section(title):
    print()
    print(f"--- {title} " + "-" * (60 - len(title)))


# ====================================================================
# 1. Parser edge cases
# ====================================================================
section("1. Number parser edge cases")

from app.content.distribution.browser_analytics.parsers.common import (
    parse_number, parse_int, parse_percent,
)

check("parse 0",         parse_number("0") == 0.0,        f"got {parse_number('0')}")
check("parse plain int", parse_number("42") == 42.0)
check("parse comma",     parse_number("1,234") == 1234.0)
check("parse large",     parse_number("12,345,678") == 12345678.0)
check("parse K suffix",  parse_number("1.5K") == 1500.0)
check("parse M suffix",  parse_number("3.2M") == 3_200_000.0)
check("parse B suffix",  parse_number("1B") == 1_000_000_000.0)
check("parse dollar",    parse_number("$42") == 42.0)
check("parse negative",  parse_number("-5") == -5.0)
check("parse decimal",   parse_number("3.14") == 3.14)
check("parse (--) null", parse_number("(--)") is None)
check("parse -- null",   parse_number("--") is None)
check("parse empty",     parse_number("") is None)
check("parse whitespace", parse_number("   ") is None)

check("parse_int rounds", parse_int("3.7") == 4)
check("parse_int negative", parse_int("-5") == -5)
check("parse_int K", parse_int("2.5K") == 2500)

check("percent simple", parse_percent("42%") == 42.0)
check("percent parens", parse_percent("(42%)") == 42.0)
check("percent negative", parse_percent("(-100.0%)") == -100.0)
check("percent plus", parse_percent("(+5.3%)") == 5.3)
check("percent (--) null", parse_percent("(--)") is None)
check("percent zero", parse_percent("(0.0%)") == 0.0)


# ====================================================================
# 2. TikTok parser on synthetic data
# ====================================================================
section("2. TikTok parser - synthetic data")

from app.content.distribution.browser_analytics.parsers.tiktok import (
    parse_tiktok_overview,
)

synthetic = {"elements": [
    {"tag": "button", "text": "Last 28 days", "x": 868, "y": 18, "w": 121, "h": 32},
    {"tag": "button", "text": "Video views\n12,345\n1,234\n(+11.1%)", "x": 113, "y": 101},
    {"tag": "button", "text": "Profile views\n678\n-5\n(-0.7%)", "x": 302, "y": 101},
    {"tag": "button", "text": "Likes\n890\n45\n(+5.3%)", "x": 491, "y": 101},
    {"tag": "button", "text": "Comments\n23\n0\n(--)", "x": 680, "y": 101},
    {"tag": "button", "text": "Shares\n42\n3\n(+7.7%)", "x": 869, "y": 101},
    {"tag": "button", "text": "Est. rewards\n$1.50K\n$120\n(+8.7%)", "x": 1058, "y": 101},
]}
r = parse_tiktok_overview(synthetic)
check("period parsed",       r.get("period") == "28d",         f"got {r.get('period')}")
check("views=12345",         r.get("views") == 12345,          f"got {r.get('views')}")
check("profile_views=678",   r.get("profile_views") == 678,    f"got {r.get('profile_views')}")
check("likes=890",           r.get("likes") == 890)
check("comments=23",         r.get("comments") == 23)
check("shares=42",           r.get("shares") == 42)
check("rewards=1500",        r.get("estimated_earnings") == 1500.0,
                             f"got {r.get('estimated_earnings')}")
check("raw_cards all 6",     len(r["metrics_json"]["raw_cards"]) == 6)
check("delta pct captured",  r["metrics_json"]["raw_cards"]["Video views"]["delta_pct_numeric"] == 11.1)


# ====================================================================
# 3. TikTok parser on REAL recon file
# ====================================================================
section("3. TikTok parser - real recon file from disk")

recon_dir = Path("D:/Orb/data/browser_recon")
tiktok_files = sorted(recon_dir.glob("tiktok_astraukai-*.txt"))
check("recon file exists", bool(tiktok_files),
      f"{len(tiktok_files)} file(s)" if tiktok_files else "no files in browser_recon/")

if tiktok_files:
    latest = tiktok_files[-1]
    text = latest.read_text(encoding="utf-8")
    print(f"         reading: {latest.name}")

    # The DOM snapshot in the file is Python repr (single quotes).
    # Extract the {'elements': [...]} block and literal_eval it.
    marker = "## DOM / ACCESSIBILITY TREE"
    idx = text.find(marker)
    check("found DOM marker", idx > 0)
    if idx > 0:
        body = text[idx:]
        start = body.find("{")
        snapshot_str = body[start:].strip()
        try:
            snapshot = ast.literal_eval(snapshot_str)
            check("DOM parsed",     isinstance(snapshot, dict))
            n_elements = len(snapshot.get("elements", []))
            check("has elements",   "elements" in snapshot and isinstance(snapshot["elements"], list),
                                    f"{n_elements} elements")

            parsed = parse_tiktok_overview(snapshot)
            # Real page had all zeros (no posts). Verify we extracted them.
            check("real: views=0",         parsed.get("views") == 0,
                                           f"got {parsed.get('views')}")
            check("real: likes=0",         parsed.get("likes") == 0)
            check("real: comments=0",      parsed.get("comments") == 0)
            check("real: shares=0",        parsed.get("shares") == 0)
            check("real: profile_views=0", parsed.get("profile_views") == 0)
            check("real: rewards=0",       parsed.get("estimated_earnings") == 0.0)
            check("real: period=7d",       parsed.get("period") == "7d",
                                           f"got {parsed.get('period')}")
            check("real: 6 raw cards",     len(parsed["metrics_json"]["raw_cards"]) == 6,
                                           f"got {list(parsed['metrics_json']['raw_cards'].keys())}")
        except Exception as e:
            check(f"DOM parse exception", False, str(e))
            traceback.print_exc()


# ====================================================================
# 3b. Meta parser on synthetic aggregate-card data
# ====================================================================
section("3b. Meta parser - synthetic aggregate cards")

from app.content.distribution.browser_analytics.parsers.meta import (
    parse_meta_overview,
)

synthetic_meta_aggregate = {"elements": [
    {"tag": "div", "role": "button",
     "text": "Last 28 days: 25 Mar 2026 - 21 Apr 2026", "x": 518, "y": 20},
    {"tag": "div", "text": "Views\n755\n767.8%"},
    {"tag": "div", "text": "Reach\n172\n911.8%"},
    {"tag": "div", "text": "Content interactions\n20\n400%"},
    {"tag": "div", "text": "Follows\n3\n"},
    {"tag": "div", "text": "Facebook visits\n41\n"},
]}
mr = parse_meta_overview(synthetic_meta_aggregate)
check("meta period",        mr.get("period") == "28d", f"got {mr.get('period')}")
check("meta views=755",     mr.get("views") == 755, f"got {mr.get('views')}")
check("meta reach=172",     mr.get("reach") == 172, f"got {mr.get('reach')}")
check("meta interactions=20", mr.get("content_interactions") == 20)
check("meta follows=3",     mr.get("followers_delta") == 3)
check("meta profile_views=41", mr.get("profile_views") == 41)
check("meta used aggregates",
      mr["metrics_json"].get("source_strategy") == "aggregates")


# ====================================================================
# 3c. Meta parser fallback on REAL recon file (per-post sum path)
# ====================================================================
section("3c. Meta parser - real recon file")

meta_files = sorted(recon_dir.glob("meta_business-*.txt"))
check("meta recon file exists", bool(meta_files),
      f"{len(meta_files)} file(s)" if meta_files else "none found")

if meta_files:
    latest_meta = meta_files[-1]
    print(f"         reading: {latest_meta.name}")
    text = latest_meta.read_text(encoding="utf-8")
    idx = text.find("## DOM / ACCESSIBILITY TREE")
    if idx > 0:
        body = text[idx:]
        start = body.find("{")
        import json
        snapshot = None
        try:
            snapshot = json.loads(body[start:].strip())
            check("meta DOM JSON-parsed", isinstance(snapshot, dict))
        except Exception:
            try:
                snapshot = ast.literal_eval(body[start:].strip())
                check("meta DOM repr-parsed", isinstance(snapshot, dict))
            except Exception as e:
                check("meta DOM parse", False, str(e))

        if snapshot:
            n = len(snapshot.get("elements", []))
            check("meta has elements", n > 0, f"{n} elements")
            parsed = parse_meta_overview(snapshot)
            # Recon files vary depending on which Meta URL was scraped
            # and when. A real dump might contain:
            #   - full overview page w/ post cards (strategy=per_post_fallback)
            #   - results page w/ aggregate labels (strategy=aggregates)
            #   - a nav-chrome-only snapshot with nothing useful (strategy=None)
            # All three are valid outcomes - the parser should never crash.
            check("meta parser returns dict", isinstance(parsed, dict))
            check("meta parser has metrics_json",
                  "metrics_json" in parsed and isinstance(parsed["metrics_json"], dict))
            strategy = parsed["metrics_json"].get("source_strategy")
            # Report which strategy (if any) ran so the test output is informative
            print(f"         strategy: {strategy or 'none (chrome-only dump)'}")


# ====================================================================
# 3d. Meta parser - synthetic per-post fallback
# ====================================================================
section("3d. Meta parser - synthetic per-post fallback")

# Simulates the /overview page's "Recent content" grid. Each post card
# is a link to /object_insights with a text shape:
#   title\ntitle\ntimestamp\nviews\nreactions\ncomments\nshares
synthetic_meta_fallback = {"elements": [
    {"tag": "div", "role": "button", "text": "Last 28 days"},
    {"tag": "a",
     "href": "/latest/insights/object_insights/?content_id=1",
     "text": "Post one\nPost one\n9 April 14:58\n81\n2\n0\n1"},
    {"tag": "a",
     "href": "/latest/insights/object_insights/?content_id=2",
     "text": "Post two\nPost two\n8 April 10:30\n45\n5\n2\n0"},
    {"tag": "a",
     "href": "/latest/insights/object_insights/?content_id=3",
     "text": "Post three\nPost three\n7 April 09:15\n12\n1\n0\n0"},
]}
fb = parse_meta_overview(synthetic_meta_fallback)
check("fallback strategy set",
      fb["metrics_json"].get("source_strategy") == "per_post_fallback",
      f"got {fb['metrics_json'].get('source_strategy')}")
check("fallback views summed", fb.get("views") == 81 + 45 + 12,
      f"got {fb.get('views')}")
check("fallback likes summed", fb.get("likes") == 2 + 5 + 1)
check("fallback posts_summed=3",
      fb["metrics_json"].get("posts_summed") == 3)


# ====================================================================
# 3e. Strategy module - per-platform attempt lists
# ====================================================================
section("3e. Strategy module")

from app.content.distribution.browser_analytics.strategies import (
    STRATEGIES, get_strategy,
)

check("meta has 3 attempts",     len(STRATEGIES["meta_business"]) == 3,
                                  f"got {len(STRATEGIES['meta_business'])}")
check("tiktok has 1 attempt",    len(STRATEGIES["tiktok_astraukai"]) == 1)
check("youtube has 1 attempt",   len(STRATEGIES["youtube_studio"]) == 1)
check("unknown platform",        get_strategy("nonexistent") == [])
check("meta attempt 1 is content_summary",
      "content_summary" in STRATEGIES["meta_business"][0]["url"])
check("meta attempt 2 is scroll",
      STRATEGIES["meta_business"][1].get("scroll") is True)
check("meta attempt 3 is overview",
      "overview" in STRATEGIES["meta_business"][2]["url"])
# Every attempt has required keys
for plat, attempts in STRATEGIES.items():
    for att in attempts:
        check(f"{plat}/{att.get('label')} has url", "url" in att)
        check(f"{plat}/{att.get('label')} has label", "label" in att)


# ====================================================================
# 3f. _is_meaningful logic
# ====================================================================
section("3f. _is_meaningful signal detection")

from app.content.distribution.browser_analytics.scrape import _is_meaningful

check("empty metrics rejected",
      not _is_meaningful({"metrics_json": {}}))
check("all-None fields rejected",
      not _is_meaningful({"views": None, "likes": None, "metrics_json": {}}))
check("all-zero Meta-style rejected",
      not _is_meaningful({"views": 0, "likes": 0, "reach": 0,
                          "metrics_json": {}}))
check("raw_cards accepts TikTok zeros",
      _is_meaningful({"views": 0, "likes": 0,
                      "metrics_json": {"raw_cards": {"Views": {"value": "0"}}}}))
check("raw_aggregates accepts Meta hit",
      _is_meaningful({"views": 755, "metrics_json":
                       {"source_strategy": "aggregates",
                        "raw_aggregates": {"Views": "755"}}}))
check("posts_summed accepts fallback",
      _is_meaningful({"views": 81, "metrics_json":
                       {"source_strategy": "per_post_fallback",
                        "posts_summed": 1}}))
check("non-zero field accepts even without strategy",
      _is_meaningful({"views": 100, "metrics_json": {}}))


# ====================================================================
# 4. DB round-trip
# ====================================================================
section("4. ChannelAnalytics DB round-trip")

from app.db import SessionLocal, engine, Base
from app.content.distribution.browser_analytics.models import ChannelAnalytics

# Pre-create the table (in case the running backend predates its registration).
# create_all is idempotent - safe to call when the table already exists.
try:
    Base.metadata.create_all(engine, tables=[ChannelAnalytics.__table__])
    check("create_all idempotent", True)
except Exception as e:
    check("create_all", False, str(e))

db = SessionLocal()
try:
    # Insert a fake scrape row
    fake = ChannelAnalytics(
        platform="tiktok_astraukai",
        period="7d",
        views=123,
        likes=45,
        comments=7,
        shares=3,
        profile_views=89,
        estimated_earnings=12.50,
        source="test_harness",
        source_url="https://www.tiktok.com/tiktokstudio/analytics/overview",
        metrics_json={"raw_cards": {"Video views": {"value": "123", "delta": "10"}}},
    )
    db.add(fake)
    db.commit()
    db.refresh(fake)
    check("insert committed", fake.id is not None, f"id={fake.id[:8]}...")
    check("captured_at set",  fake.captured_at is not None)

    # Query it back
    found = (
        db.query(ChannelAnalytics)
        .filter(ChannelAnalytics.source == "test_harness")
        .order_by(ChannelAnalytics.captured_at.desc())
        .first()
    )
    check("query found",      found is not None)
    check("views round-trip", found.views == 123)
    check("likes round-trip", found.likes == 45)
    check("period round-trip", found.period == "7d")
    check("json round-trip",
          found.metrics_json["raw_cards"]["Video views"]["value"] == "123")

    # Clean up the test row
    db.delete(found)
    db.commit()
    remaining = db.query(ChannelAnalytics).filter(
        ChannelAnalytics.source == "test_harness"
    ).count()
    check("cleanup deleted test row", remaining == 0)
finally:
    db.close()


# ====================================================================
# 5. Scrape dispatch logic (parser registry + INSIGHTS_URLS sanity)
# ====================================================================
section("5. Scrape dispatch logic")

from app.content.distribution.browser_analytics.parsers import (
    PARSERS, get_parser,
)
from app.content.distribution.browser_analytics.urls import INSIGHTS_URLS

check("tiktok parser registered",   "tiktok_astraukai" in PARSERS)
check("meta parser registered",     "meta_business" in PARSERS)
check("youtube_studio no parser",   "youtube_studio" not in PARSERS,
                                    "skipping - API covers this")

check("all parsers have URLs",
      all(p in INSIGHTS_URLS for p in PARSERS),
      f"parsers: {list(PARSERS.keys())}, urls: {list(INSIGHTS_URLS.keys())}")

check("get_parser works",     get_parser("tiktok_astraukai") is not None)
check("get_parser meta",      get_parser("meta_business") is not None)
check("get_parser fallback",  get_parser("nonexistent_platform") is None)


# ====================================================================
# 6. Router endpoint inventory
# ====================================================================
section("6. Router endpoint inventory")

from app.content.distribution.browser_analytics import router

expected = {
    "/content/distribution/browser_analytics/platforms",
    "/content/distribution/browser_analytics/recon/{platform}",
    "/content/distribution/browser_analytics/recon-all",
    "/content/distribution/browser_analytics/scrape/{platform}",
    "/content/distribution/browser_analytics/scrape-all",
    "/content/distribution/browser_analytics/channel-summary/{platform}",
    "/content/distribution/browser_analytics/channel-history/{platform}",
}
actual = {r.path for r in router.routes}
missing = expected - actual
extra = actual - expected
check("no missing routes", not missing, f"missing: {missing}" if missing else "")
check("no unexpected routes", not extra, f"extra: {extra}" if extra else "")


# ====================================================================
# SUMMARY
# ====================================================================
print()
print("=" * 70)
print(f"RESULTS:  {passed} passed, {failed} failed")
if failed:
    print()
    print("FAILURES:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("All tests passed.")
    sys.exit(0)
