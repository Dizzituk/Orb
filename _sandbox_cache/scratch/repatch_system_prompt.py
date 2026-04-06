from pathlib import Path
path = Path("D:/Orb/app/debug/system_prompt.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    '- You have direct read/write access to all project codebases on the host filesystem."""',
    '- You can read host project codebases, but host repos are read-only by default.\n- You must make code edits only in the sandbox mirror unless Taz explicitly authorises a host write for a specific task."""'
)
text = text.replace(
    '- **Write files**, **edit files**, and **run commands** directly on the host filesystem."""',
    '- **Write files** and **edit files** in the sandbox mirror only unless Taz explicitly overrides that boundary for a specific task.\n- **Run commands** in the sandbox when changing code. Use host-side commands only for host-only diagnostics or user-approved deployment actions."""'
)
text = text.replace(
    '- You have FULL read/write access to project files on the host filesystem.\n- Allowed write paths: D:/Orb/, D:/orb-desktop/, D:/Astra Android Folder/, C:/Users/dizzi/Documents/.\n- Use your tools directly — do NOT paste code and ask the user to copy it.',
    '- Host project repos are READ-ONLY by default. Treat `D:/Orb/`, `D:/orb-desktop/`, `D:/orb-electron/`, and other live project roots as protected unless Taz explicitly approves a host write for that task.\n- Sandbox mirrors are the default and only approved location for code edits.\n- Before any code edit, verify sandbox availability. If the sandbox is unavailable or unreachable, STOP and ask Taz to start the sandbox.\n- Never silently fall back to editing the live host repo when sandbox access is missing.\n- User-document folders like `C:/Users/dizzi/Documents/` may still be written when the user explicitly asks for file output there.\n- Use your tools directly — do NOT paste code and ask the user to copy it.'
)
path.write_text(text, encoding="utf-8")
print("repatched system_prompt.py")
