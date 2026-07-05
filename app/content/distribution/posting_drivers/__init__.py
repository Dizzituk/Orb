# FILE: app/content/distribution/posting_drivers/__init__.py
# Purpose: Composite browser posting drivers (the posting layer).
# Called-by: app.tools.social_posting_tools
# Depends-on: submodules (results, driver_runner, self_heal, meta_driver)
# Last-renovated: 2026-07-02
"""
Posting drivers — deterministic, wait-gated browser composites that
publish through platform composers and return a PostResult.

The live posting surface is the Meta Business Suite composer
(meta_driver), which drives Facebook AND Instagram from one login
(IG->FB auto-share). See selector_maps/meta_business.json for the
decision record and selector evidence.
"""
from app.content.distribution.posting_drivers.results import PostResult, failure

__all__ = ["PostResult", "failure", "post_image", "post_reel"]


def __getattr__(name):
    # Lazy re-export so importing the package doesn't drag in the bridge/DB.
    if name in ("post_image", "post_reel"):
        from app.content.distribution.posting_drivers import meta_driver
        return getattr(meta_driver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
