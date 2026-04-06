from pathlib import Path

path = Path(r"D:/Orb/app/debug/system_prompt.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "- You have direct read/write access to all project codebases on the host filesystem.",
    "- You can read host project codebases, but host repos are read-only by default.",
)
text = text.replace(
    "- **Write files**, **edit files**, and **run commands** directly on the host filesystem.",
    "- **Write files**, **edit files**, and **run commands** in the sandbox environment.",
)
text = text.replace(
    "- You have FULL read/write access to project files on the host filesystem.\n- Allowed write paths: D:/Orb/, D:/orb-desktop/, D:/Astra Android Folder/, C:/Users/dizzi/Documents/.",
    "- Host project repositories are read-only by default.\n- Only the sandbox mirror may be used for code edits, file writes, and project commands.\n- If the sandbox is unavailable or disconnected, do not edit host code. Ask the user to start the sandbox.\n- Host-direct writes are allowed only for explicit user-requested personal files via the dedicated user file tools.\n- Do not write to D:/Orb/, D:/orb-desktop/, D:/orb-electron/, or other host project roots unless the user explicitly overrides this safety policy for that task.",
)
path.write_text(text, encoding="utf-8")
print("patched system_prompt.py")
