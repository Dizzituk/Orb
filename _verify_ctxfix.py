# Verification suite for the 2026-06-10 context/image fixes. Read-only.
import sys
sys.path.insert(0, r"D:\Orb")
PASS, FAIL = 0, 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS  {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}")

print("== 1. image gen intent (real failure messages from convos 270/271) ==")
from app.llm.routing.chat_intent_detection import (
    detect_image_gen_intent as gen,
    detect_image_refinement as ref,
    _IMAGE_MSG_MARKERS,
)
check("make this quote into an Instagram Square -> True",
      gen("I want to make this quote into an Instagram Square"))
check("make that quote into an Instagram Square with a Gradient background -> True",
      gen("I want to make that quote into an Instagram Square with a Gradient background"))
check("make it into a Instagram Square -> True",
      gen("I want to make it into a Instagram Square"))
check("make the image please -> True", gen("Yep sounds good to me make the image please"))
check("make me a bar chart still True", gen("make me a bar chart of sales"))
check("generate an image of a duck still True", gen("generate an image of a duck"))
check("turn this into an infographic still True", gen("turn this into an infographic"))
check("create a quote card -> True", gen("create a quote card for me"))
check("design me a poster -> True", gen("design me a poster"))
check("negative: make this work -> False", not gen("can you make this work"))
check("negative: I need a plan for tomorrow -> False", not gen("I need a plan for tomorrow"))
check("negative: make me a cup of tea -> False", not gen("make me a cup of tea"))
check("negative: make it shorter -> False", not gen("can you make it shorter"))

print("== 2. image refinement ==")
check("That isnt the image... Redesign it please and send it again -> True",
      ref("That isn't the image that I got Redesign it please and send it again"))
check("redesign it -> True", ref("redesign it"))
check("make it darker still True", ref("make it darker"))
check("wrong image -> True", ref("that's the wrong image"))
check("negative: what do you think about this -> False",
      not ref("what do you think about this"))
check("bridge marker present", "[ASTRA_ARTIFACT:image:" in _IMAGE_MSG_MARKERS)

print("== 3. aspect ratio ==")
from app.llm.image_prompt_synth import (
    _detect_aspect_ratio as ar, _looks_truncated, _build_fallback_prompt,
)
check("real 271 case: synth says Instagram square...split vertically -> 1:1",
      ar("Yep sounds good to me make the image please with the changes you suggested",
         "A high-contrast activist-poster style Instagram square. The image is split vertically") == "1:1")
check("user square beats synth vertical -> 1:1",
      ar("make me an instagram square", "the image is split vertically") == "1:1")
check("youtube banner -> 16:9", ar("make a youtube banner", "") == "16:9")
check("phone wallpaper -> 9:16", ar("phone wallpaper of a forest", "") == "9:16")
check("history of rome does NOT match story -> None",
      ar("tell me about the history of rome", "") is None)
check("iconic does NOT match icon -> None", ar("", "an iconic scene at dusk") is None)
check("vertical poster: user says vertical... banner absent, poster not AR word -> 9:16",
      ar("a vertical poster of mountains", "") == "9:16")

print("== 4. truncation guard ==")
frag = "A high-contrast activist-poster style Instagram square. The image is split vertically"
check("85-char fragment + MAX_TOKENS -> truncated", _looks_truncated(frag, "FinishReason.MAX_TOKENS"))
check("85-char fragment + STOP -> still truncated (no terminal punct)", _looks_truncated(frag, "STOP"))
long_ok = ("Instagram square, 1:1 ratio, minimalist editorial quote card with a deep "
           "charcoal gradient background, bold white sans-serif headline reading exactly: "
           "'It isn't about punishing the rich.' with a smaller grey subline beneath. "
           "Clean margins, high contrast, professional social media aesthetic.")
check("full prompt + STOP -> not truncated", not _looks_truncated(long_ok, "STOP"))
check("empty -> truncated", _looks_truncated("", "STOP"))

print("== 5. fallback prompt carries the quote ==")
hist = [
    {"role": "user", "content": "I want to make this quote into an Instagram Square"},
    {"role": "assistant", "content": "Paste the quote text here."},
    {"role": "user", "content": "It Isn't about punishing the rich..."},
    {"role": "assistant", "content": ("I'd make the square with: 'It isn't about punishing "
        "the rich. It's about whether the productivity of a civilisation belongs to everyone "
        "born into it - or only to those who own the machines.'")},
    {"role": "user", "content": "Yep sounds good to me make the image please"},
]
fb = _build_fallback_prompt("Yep sounds good to me make the image please", hist)
check("fallback contains the quote text", "punishing the rich" in fb and "own the machines" in fb)
check("fallback contains the user request", "make the image please" in fb)

print("== 6. list_messages returns the LAST N (live DB, read-only) ==")
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.memory._service_utils_2 import list_messages
    eng = create_engine(r"sqlite:///D:\Orb\data\orb_memory.db")
    S = sessionmaker(bind=eng)
    s = S()
    ids = [m.id for m in list_messages(s, 270, limit=20)]
    check(f"project 270 limit=20 -> 3580..3599 ascending (got {ids[0]}..{ids[-1]}, n={len(ids)})",
          ids == list(range(3580, 3600)))
    ids5 = [m.id for m in list_messages(s, 271, limit=5)]
    check(f"project 271 limit=5 -> 3605..3609 (got {ids5})", ids5 == list(range(3605, 3610)))
    s.close()
except Exception as e:
    import traceback; traceback.print_exc()
    check(f"live DB test ran ({e})", False)

print("== 7. import smoke tests ==")
try:
    from app.memory.integration import record_session_activity
    check("record_session_activity importable", callable(record_session_activity))
except Exception as e:
    check(f"record_session_activity import ({e})", False)
try:
    import app.bridge.router as _br
    check("app.bridge.router imports cleanly", True)
except Exception as e:
    check(f"app.bridge.router import ({e})", False)

print(f"\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
