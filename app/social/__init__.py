# FILE: app/social/__init__.py
"""
ASTRA social media APIs.

First-party HTTP integrations with social platforms (Meta Graph, TikTok
Content Posting API, YouTube Data API, WordPress REST). Sits alongside
app/web_automation (browser-driven) — the two are *complementary*, not
substitutes:

  * APIs win for narrow, deterministic publish/read/schedule actions
  * Browser automation wins for engagement reading, comment drafting,
    content discovery, novel-UI navigation
  * Verification crosses channels: API actions verified by browser/vision,
    browser actions verified by API reads.

Currently shipped:
  - meta_client / meta_post : Facebook Page publishing via Graph API.

Planned:
  - instagram_post  : pending public image hosting decision
  - tiktok_post     : Content Posting API
  - youtube_upload  : Data API v3 resumable uploads
  - wordpress_post  : REST API
"""
