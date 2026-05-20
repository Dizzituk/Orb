# FILE: app/debug/web_tool_playbooks.py
"""
Long-form playbook strings used by web_tool_definitions.

Kept in their own module so web_tool_definitions.py stays under the
20 KB target and so playbooks can be edited without touching schema
definitions. Each constant is plain prose embedded into the relevant
tool description via f-string at definition time.

v1.0 (2026-04-26): extracted from web_tool_definitions.py + new
sections for portal-aware click interpretation, bounded retry, and
wrong-page recovery.
"""
from __future__ import annotations


# ── Text input targeting ──────────────────────────────────────────────
# Empty contenteditable text inputs (Meta caption box, IG comment field,
# WordPress post body, YouTube description) have a counter-intuitive
# focus zone: only the TOP LINE of the visible rectangle accepts focus
# clicks. The rest of the empty whitespace inside the bbox shows the
# arrow cursor instead of the text I-beam, so a click there does NOT
# focus the editable — system_keys then types into nothing. Vision
# routinely returns the visual CENTRE of the box, which is dead space.
# This rule fires before every click into a text input; it's the most
# common silent-failure mode in social-platform composers.
TEXT_INPUT_TARGETING = """\
TEXT INPUT TARGETING — clicking into a text box / caption / comment
field / search box / any contenteditable to start typing:

  • AIM TOP, NOT CENTRE. Click `y_top + 14` (inside the top line),
    NOT the bbox centre. Empty contenteditables in modern composers
    only accept focus on their first line — the rest of the visible
    rectangle is dead space (arrow cursor, no text I-beam, click
    does nothing). Hit-testing CSS often differs from the visible
    border by 30–80 px; aiming high stays safely inside the live
    strip.

  • Worked numbers: bbox at (x=240, y=820, w=420, h=180) → click at
    (240 + 210, 820 + 14) = (450, 834). NOT (450, 910) — that's the
    centre and won't focus.

  • snap_to_button does NOT help here. Snap looks for interactive
    BUTTONS within radius; text inputs aren't buttons, so snap
    either does nothing or pulls you to a nearby icon (emoji
    picker, format toggle, hashtag button) and you click that
    instead. Pass snap_to_button=false / unset for text inputs.

  • VERIFY FOCUS before typing. Run web_dom_snapshot after the
    click — the focused element will be marked. If focus didn't
    land, the textbox bbox you targeted will be back to its
    pre-click state with no caret indicator. RE-CLICK closer to
    the top edge (y_top + 6) and re-verify before retrying type.

  • If two top-edge clicks both fail to focus, switch tools:
    web_type with a CSS selector targets the element directly
    via Electron's typing API and bypasses click-focus entirely.
    Derive the selector from dom_snapshot — usually role=textbox
    + aria-label, or a contenteditable=true div near the visible
    label ("Text", "Caption", "Comment", "Description").

  • Vision is poor at locating empty text inputs (no visual
    landmark inside the box for the model to anchor on). If you
    need vision to find one, ask for the TOP-LEFT corner
    explicitly: "Return pixel coords (x, y) of the TOP-LEFT
    CORNER of the empty caption text box, where the cursor will
    appear when I click. Add 14 to y when reporting." Don't ask
    for the "centre" of an empty box.
"""


# ── Click result interpretation ────────────────────────────────────────
# Tells the agent how to read the four *_changed flags vs the `snapped`
# field. The big trap is React portals: dropdowns / popovers / modals
# are rendered outside the click target's DOM subtree, so element_changed
# is false; if the URL also doesn't change, all four flags read false
# even though the click hit the correct button. Without this guidance,
# the agent treats `changed=false` as "click missed" and silently gives
# up halfway through a flow.
CLICK_RESULT_INTERPRETATION = """\
INTERPRETING THE RESULT — read `snapped` AND `changed` together, not
either one alone:

  • snapped is populated AND snapped.text matches your target →
    the click LANDED on the right element, regardless of `changed`.
    Example: you wanted "Add Photo", snapped.text is "Add Photo",
    snapped.search_distance < 50 → the click hit the button.

  • changed is FALSE but snapped landed on the right target →
    DO NOT treat this as a miss. Many UI side effects happen in
    React portals (dropdown menus, popovers, picker modals) that
    are rendered outside the clicked element's subtree, so none
    of url/title/heading/element_changed catches them. The click
    worked; the visible effect is somewhere else on the page.
    NEXT STEP: run web_dom_snapshot — the new menu items / modal
    will be in the fresh tree even though the change flags missed
    them. Only after dom_snapshot confirms NOTHING new appeared
    should you treat the click as a real miss.

  • changed is FALSE and snapped is null (or wrong target) →
    real miss. Re-read web_dom_snapshot, pick a different target.
    When multiple elements share text, prefer LARGEST w×h.

  • changed is TRUE → click worked, page visibly updated. Proceed.
"""


# ── Retry policy ───────────────────────────────────────────────────────
# Bounded persistence. Without this the model either gives up after one
# failure or loops forever. Three attempts on the same target is the
# right balance — enough to recover from transient layout shifts
# (e.g. lazy-loaded composer), few enough to fail fast on real bugs.
RETRY_POLICY = """\
RETRY POLICY — when an action doesn't produce the expected effect:

  • Up to THREE attempts on the same target before pivoting.
    Between attempts, run web_dom_snapshot or web_vision_check to
    confirm the page state actually matches what you expect. The
    composer can take an extra second to mount; a single retry
    after a 1-second wait often resolves transient layout misses.

  • If three attempts on the same target all fail, PIVOT — don't
    keep clicking the same coordinates. Pivot options:
      - Switch from web_dom_snapshot coords to web_vision_check coords
        (vision sees what the user sees; DOM sometimes hides things).
      - Switch from web_upload_file's CDP intercept to the
        web_click + system_keys flow (the native dialog path).
      - Re-read the playbook for the platform — Meta needs a
        dropdown step, Instagram needs a modal step, etc.

  • Never silently report success after a failed action. If the
    expected outcome (file dialog opening, thumbnail appearing,
    post showing on the wall) hasn't been verified, say so plainly
    and either retry or hand back to the user.
"""


# ── Wrong-page recovery ───────────────────────────────────────────────
# The Meta sidebar in particular — Insights, Ads Manager, Planner —
# is one careless click away from the composer. Without explicit
# recovery guidance the agent surfaces "I'm on the wrong page" and
# stops. With it, the agent navigates back and continues.
WRONG_PAGE_RECOVERY = """\
WRONG-PAGE RECOVERY — if web_current_state or web_dom_snapshot shows
you've landed somewhere unexpected (Insights instead of composer,
Ads Manager instead of Posts, a post-detail view instead of the
feed):

  • Don't stop and ask the user. Recover and continue.

  • Preferred: web_navigate to the known landing URL for the
    session. For Meta Business Suite the composer lives under
    https://business.facebook.com/latest/home or
    https://business.facebook.com/latest/posts/published_posts —
    navigate there and call web_dom_snapshot to find the
    "Create Post" entry point.

  • If you don't know the URL: web_dom_snapshot the current page,
    look for navigation links labelled Home / Posts / Composer /
    Create / "Back to feed", click whichever returns to the place
    you needed to be, then resume the original task.

  • Only escalate to the user if recovery itself fails twice.
"""


# ── Meta upload flow (full worked example) ─────────────────────────────
# Production-tested pattern. The upload is a TWO-CALL sequence:
# web_click on "Add Photo" to open the dropdown, then web_upload_file
# against the upload-button coordinates. web_upload_file uses Chrome
# DevTools Protocol to intercept the file-chooser request BEFORE the
# native OS dialog can spawn — the file is injected directly into the
# hidden <input type=file> behind the button. No OS dialog, no
# system_keys typing, no vision-check on a dialog the webview
# screenshot can't see anyway. The native-dialog fallback is
# documented at the bottom for the rare cases CDP misses (closed
# shadow DOM, custom picker widgets, A/B test variants).
META_UPLOAD_PLAYBOOK = """\
WORKED EXAMPLE — Posting an image to Meta Business (PRIMARY FLOW)

User: "post that image to Facebook with this caption ..."

  1. web_open_session(session='meta_business')
  2. web_dom_snapshot                # see the composer
  3. web_click on "Add Photo"        # opens the upload dropdown
                                     # (use snap_to_button=true if
                                     #  coords come from vision)

  4. web_dom_snapshot                # MANDATORY — even if click
                                     # returned changed=false. The
                                     # dropdown is in a React portal;
                                     # the change-detection flags
                                     # miss it. Snapshot anyway.

     Two possible outcomes after step 4:

     (a) DROPDOWN PATTERN (Meta default): the new snapshot shows
         buttons like "Upload from computer", "From your device",
         "Browse files". DO NOT click these with web_click — that
         spawns the native OS dialog and forces the brittle
         fallback path. Read the (x, y) of the upload-from-computer
         button from dom_snapshot and continue at step 5.

     (b) MODAL PATTERN (Instagram, TikTok, some YouTube flows):
         the snapshot shows a full picker modal with Recents
         thumbnails and an upload trigger. The DOM can't always
         distinguish the trigger from thumbnails — use vision to
         find the coords:
             web_vision_check question: "Return the pixel
               coordinates (x, y) of the centre of the button
               that opens a file picker to upload a NEW photo
               from my computer. Ignore thumbnails of existing
               photos in any Recents grid. Format: x=NUMBER,
               y=NUMBER followed by a one-sentence description."
         Then continue at step 5.

  5. PRIMARY UPLOAD — web_upload_file using the coords from step 4:

         web_upload_file(
             session='meta_business',
             x=X, y=Y,                # coords of the
                                      # "Upload from computer" button
             path='C:\\\\absolute\\\\path\\\\image.png'
         )

     This uses Chrome DevTools Protocol to intercept the
     file-chooser request Meta's JS dispatches when the upload
     button is activated. CDP catches it BEFORE the native OS
     dialog can spawn and injects the file directly into the
     hidden <input type=file> behind the button. The result
     dict tells you which path actually ran:

       • mode='cdp' AND attached=<filename>
            → upload succeeded. Continue at step 6.
       • mode='os_dialog' (or attached='')
            → CDP missed the chooser; the native dialog has
              opened instead. See FALLBACK section below.
       • ok=False with an error
            → likely bad coords (clicked the wrong element, no
              chooser was triggered) or the input is inside a
              closed shadow DOM CDP can't reach. Retry once
              with vision-derived coords; if still failing,
              go to FALLBACK.

  6. POLL for the thumbnail to appear — do NOT single-shot.
     Meta takes 3–15 seconds to render an uploaded image into the
     composer DOM (longer for large files, 30+ seconds for videos).
     A web_dom_snapshot taken 1–2 seconds after the upload call
     will show the page in pre-render state and falsely report no
     thumbnail. The image is uploading; the DOM just hasn't caught
     up yet. Pattern:

         for attempt in 1..6:               # for video: 1..20
             web_dom_snapshot
             # success markers in the snapshot:
             #   - file dimensions like "1024 x 1024"
             #   - alt text containing the filename or "image preview"
             #   - role=img elements that weren't there before
             #   - "Edit photo" / "Edit media" controls
             #   - a delete/remove button next to a media slot
             if any of those present:
                 break — upload confirmed
             # else loop; one snapshot is ~500ms so 6 iterations
             # span ~3–5 seconds, plenty for image rendering

     ONLY after the full attempt budget shows no thumbnail should
     you declare upload failure. A common silent-failure mode is
     reporting "no thumbnail" after a SINGLE snapshot taken too
     quickly — the image is actually attached and the user can see
     it on screen, but the agent's verification snapshot beat the
     render. That's a false negative, not a real failure. Don't do
     that. If unsure, keep polling.

     For VIDEO uploads, raise the cap to 20 attempts and accept
     that processing can take 30–60+ seconds before the thumbnail
     finalises in the composer.

  7. Caption box — PRIMARY path is web_type with a selector, NOT
     click-then-type. Meta's caption box is a stable contenteditable
     div that web_type can target directly, bypassing the entire
     click-focus race that wastes turns and lands clicks in dead
     space between the "Customise post" toggle and the textbox.

         web_type(
             session='meta_business',
             selector="div[contenteditable='true']",
             text='<your caption>',
             clear_first=true
         )

     This avoids the click-focus problem entirely. No need to find
     the textbox's exact (x, y), no y_top + 14 calculation, no risk
     of clicking dead space, no system_keys focus race. The
     selector resolves to the right element regardless of viewport
     position or what other UI is open.

     VERIFY via dom_snapshot, NOT vision_check:
         web_dom_snapshot
         # look in the snapshot for the contenteditable element's
         # text content / innerText — it should now contain your
         # caption. The DOM holds the truth here.
     CRITICAL: do NOT use web_vision_check to read filled text
     inputs. Gemini Flash regularly reports "empty" for boxes
     that visibly contain text — small fonts, low contrast, and
     placeholder-like rendering all defeat OCR. This is a class
     of false negative we have hit repeatedly. Trust the DOM
     snapshot's text content, not vision.

     FALLBACK (only if web_type with selector returns ok=false or
     the dom_snapshot shows the contenteditable still empty after
     the call): get the textbox bbox from dom_snapshot, click at
     (x_centre, y_top + 14) with snap_to_button=false, verify focus
     via dom_snapshot, then system_keys the caption. This was the
     original approach but is strictly brittler than the selector
     path — do not start here.

  8. Click Schedule (if scheduling) → handle date/time → click
     Publish.

PERSISTENCE RULE: the upload is always a TWO-CALL sequence — first
web_click on "Add Photo", then web_upload_file on the upload-button
coords. ONE call is never enough. Never call web_upload_file with
the Add Photo coords — that opens the dropdown but doesn't trigger
a file chooser, so CDP has nothing to intercept and the result is
meaningless.

────────────────────────────────────────────────────────────────────
FALLBACK — native OS dialog path (use only when CDP intercept misses)

Trigger conditions: web_upload_file returned mode='os_dialog', OR
returned attached='' with no mode set. Either means Meta's JS opened
the native Windows file dialog before CDP could catch the chooser
request. Recover with these steps. Do NOT use this flow as the
primary — every attempt should start at step 5 of the PRIMARY FLOW.

  F1. Verify the dialog is actually open before typing anything.
      web_vision_check CANNOT see OS-level dialogs through the
      browser-view screenshot — vision will return 'no dialog'
      even when the dialog is plainly on screen. Use this DOM
      proxy instead: run web_dom_snapshot. If Meta's underlying
      UI is now blurred or marked inert (the page DOM goes
      inactive while a modal OS dialog has focus), the dialog
      is up and waiting for input. If the page DOM looks live
      and interactive, the chooser click probably missed —
      retry with corrected coords rather than typing into
      nothing.

  F2. system_keys(text='C:\\\\absolute\\\\path\\\\image.png',
        press_enter_after=false,
        pre_delay_ms=1500)
      Type the path WITHOUT pressing Enter. The atomic
      type+Enter pattern hides focus failures (if keystrokes
      go to the wrong window, the path silently leaks and
      Enter does nothing useful).

  F3. Commit — send Enter alone:
        system_keys(text='', press_enter_after=true,
                    pre_delay_ms=300)

  F4. POLL for thumbnail — same pattern as step 6 of the PRIMARY
      flow (do NOT single-shot dom_snapshot; Meta needs 3–15 seconds
      to render). If after the full attempt budget no thumbnail
      appears, the dialog ate the keystrokes (focus stolen by
      another OS window mid-type) — pivot via RETRY POLICY, and on
      the next attempt go back to PRIMARY FLOW step 5 rather than
      retrying the fallback.

Why fallback exists: CDP intercept is not 100% reliable across all
Meta variants. A/B test buckets, closed Shadow DOM, and custom
file-picker widgets can all defeat the intercept. The PRIMARY flow
should still be the first attempt every time; only fall through to
FALLBACK when the primary explicitly reports it didn't catch the
chooser (mode='os_dialog' in the return value).
"""
