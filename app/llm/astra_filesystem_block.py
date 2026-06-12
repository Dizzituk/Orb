# FILE: app/llm/astra_filesystem_block.py
# Purpose: ASTRA filesystem facts — static knowledge block injected into chat context.
# Called-by: app.llm.routing.prompt_builders
# Depends-on: app.llm.image_output_dir
# Last-renovated: 2026-06-11
"""
ASTRA filesystem facts — static knowledge block injected into chat context.

Single responsibility: tell the chat agent where ASTRA's own generated
outputs live on disk, so it never wastes tool calls searching for a file
that a previous turn just created.

Solves the failure mode observed 2026-04-25: agent generates an image,
later needs to upload it to Meta, runs search_my_files → search_disk_live,
both miss, then asks the user for help — four wasted round-trips when the
filename was right there in chat history all along.

The path is read fresh from the IMAGE_OUTPUT_DIR env var via the shared
helper, so changes to .env automatically propagate to this block.

v1.0 (2026-04-25): Initial implementation.
v1.1 (2026-04-25): Tightened social-upload guidance — vision_check is
    now mandatory between dom_snapshot and upload_file. The DOM tree
    alone cannot disambiguate the upload trigger from Recents thumbnails
    in Meta/IG/TikTok photo-pickers, and silent timeouts break the
    hands-free workflow.
v1.2 (2026-04-26): Native-driven upload flow promoted to PRIMARY.
    web_upload_file's CDP-intercept path was failing on Meta even with
    correct coords (the page wraps the input in a way the intercept
    misses). New flow: web_click → native dialog opens → system_keys
    types path + Enter. Production-tested.
v1.3 (2026-04-26): Added Meta-specific TWO-STEP photo flow + post-click
    verification rule. Observed failure mode: agent clicks 'Add Photo'
    expecting the OS dialog to open, instead a small dropdown appears
    with 'Upload from computer' / 'From your device' / 'Search stock'
    sub-options. Without a dom_snapshot the agent doesn't see the
    dropdown and declares the task done. The playbook now requires a
    dom_snapshot after EVERY click in the upload flow, with the
    explicit instruction: keep clicking through dropdowns / menus
    until a thumbnail is visible OR a native file dialog has opened.
"""
from __future__ import annotations

import os
from typing import Optional

from app.llm.image_output_dir import get_image_output_dir


def build_filesystem_block() -> Optional[str]:
    """Return the ASTRA filesystem knowledge block.

    Always returns a string — this is canonical static knowledge, not
    derived from any data store. Cheap to call (one env-var read).
    """
    img_dir = str(get_image_output_dir())
    sep = os.sep  # '\\' on Windows, '/' elsewhere

    return (
        "[ASTRA FILESYSTEM]\n"
        "You generate images that save to a known directory. When a previous turn\n"
        "(yours or earlier) produced an image and you now need its filepath — to\n"
        "upload it, attach it to a post, or otherwise reference it — construct the\n"
        "path from the filename. DO NOT run search_my_files, search_disk_live, or\n"
        "any filesystem search to locate it.\n"
        "\n"
        f"Image directory: {img_dir}\n"
        "Filename patterns: gpt-*.png (OpenAI), nano-*.png (Gemini), chart-*.png (Plotly).\n"
        "\n"
        "Example workflow:\n"
        "  Earlier turn says: \"Generated with openai/gpt-image-2: gpt-92e9feda-194024.png\"\n"
        f"  Full path is:    {img_dir}{sep}gpt-92e9feda-194024.png\n"
        "  Pass that path directly to your upload / attach / post tool.\n"
        "\n"
        "Posting an image to social media (Facebook, Instagram, TikTok, etc.):\n"
        "  Use the NATIVE-DRIVEN flow — it works on Meta where the in-page\n"
        "  intercept does not. Sequence:\n"
        "    1. web_click on Photo/Video button → EITHER a picker modal opens\n"
        "       OR a small dropdown appears with sub-options like\n"
        "       'Upload from computer' / 'From your device' / 'Search stock'.\n"
        "       You CANNOT tell which from the click result alone.\n"
        "    2. web_dom_snapshot → ALWAYS run this. Look at what appeared.\n"
        "       → If you see a dropdown / menu with 'Upload from computer',\n"
        "         'From your device', 'From this computer', etc.: that is\n"
        "         a DIFFERENT button. Click THAT one next (still under the\n"
        "         same flow). The native file dialog opens after THIS click,\n"
        "         not the previous one. This is the Meta pattern.\n"
        "       → If you see a full picker modal (thumbnails, tabs, an\n"
        "         upload trigger button): you need vision_check to find\n"
        "         the upload trigger inside the modal.\n"
        "    3. web_vision_check → ONLY needed for the modal-style picker.\n"
        "       Skip this for the dropdown pattern — dom_snapshot already\n"
        "       has the 'Upload from computer' button by name.\n"
        "    4. web_click(x, y, snap_to_button=true if from vision) →\n"
        "       native OS file dialog opens. snap_to_button corrects for\n"
        "       vision's ±50px imprecision. Skip snap_to_button if coords\n"
        "       came from dom_snapshot.\n"
        "    5. system_keys(text=ABS_PATH, press_enter_after=true)\n"
        "       → types path, presses Enter, dialog closes, file is selected.\n"
        "    6. web_dom_snapshot → confirm thumbnail / preview is now in\n"
        "       the composer. If no thumbnail visible, the upload did NOT\n"
        "       succeed — do NOT report success. Re-attempt from step 1.\n"
        "  HARD RULE: do NOT report 'image attached' or 'upload done' until\n"
        "  step 6 confirms a visible thumbnail. The flow has 2-3 clicks on\n"
        "  Meta and at least one is non-obvious (the dropdown sub-option).\n"
        "  See web_upload_file's full docs for the worked example. NEVER use\n"
        "  read_file to attach images — that just dumps bytes into chat.\n"
        "\n"
        "(Legacy note: images generated before 2026-04-25 may still live at\n"
        " D:\\Orb\\output\\images. The current location above is the canonical one\n"
        " going forward.)\n"
        "[/ASTRA FILESYSTEM]"
    )


__all__ = ["build_filesystem_block"]
