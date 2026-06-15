# FILE: app/podcasts/__init__.py
# Purpose: Podcast module — discovery (PodcastIndex + iTunes fallback),
#          RSS subscriptions, and playback dispatch to desktop displays.
# Called-by: app.tools.podcast_tools
# Depends-on: app.podcasts.podcast_index_client, app.podcasts.subscriptions, app.podcasts.recent
# Last-renovated: 2026-06-12
"""
Podcast module.

No platform gatekeeper: discovery via the PodcastIndex API (free key) with
the keyless iTunes Search API as fallback, feeds via open RSS (feedparser),
playback handed to the desktop display manager (minimal window) until a
dedicated audio pane exists. Phone playback is a follow-up Bridge task.

Secrets: PODCASTINDEX_API_KEY + PODCASTINDEX_API_SECRET env vars (register
free at api.podcastindex.org). Without them, search degrades to iTunes and
trending is unavailable — nothing crashes.
"""
