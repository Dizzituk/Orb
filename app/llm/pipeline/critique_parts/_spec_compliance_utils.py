import re


_STACK_KEYWORDS = {
    "python": "Python", "pygame": "Python+Pygame", "tkinter": "Python+Tkinter",
    "pyqt": "Python+PyQt", "flask": "Python+Flask", "fastapi": "Python+FastAPI",
    "django": "Python+Django", "javascript": "JavaScript", "typescript": "TypeScript",
    "react": "TypeScript/React", "electron": "TypeScript/Electron",
    "node.js": "Node.js", "nodejs": "Node.js", "next.js": "TypeScript/Next.js",
    "vue": "TypeScript/Vue", "rust": "Rust", "golang": "Go",
    "c++": "C++", "c#": "C#", "java": "Java",
}

_SCOPE_INFLATION_KEYWORDS = [
    "electron-builder", "packaging", "installer", ".exe", ".msi",
    "telemetry", "analytics", "crash-report", "remote",
    "vite", "webpack", "bundler",
    "playwright", "e2e", "end-to-end",
    "authentication", "auth", "oauth",
    "database", "sqlite", "persistence",
    "%appdata%", "local storage",
    "settings ui", "menus", "overlays",
]

_INLINE_EXCLUSION_PATTERNS = [
    re.compile(r'^\s*[\u274c\u2717\u2718\u2573\u00d7]', re.UNICODE),
    re.compile(r'no\s+mobile', re.IGNORECASE),
    re.compile(r'not\s+(?:in\s+)?(?:this|phase|scope|v1|mvp)', re.IGNORECASE),
    re.compile(r'phase\s+[2-9]', re.IGNORECASE),
    re.compile(r'must\s+not\s+block', re.IGNORECASE),
    re.compile(r'future\s+phase', re.IGNORECASE),
    re.compile(r'explicitly\s+not', re.IGNORECASE),
    re.compile(r'\*\*Reviewer\s+(?:Claim|Suggestion)', re.IGNORECASE),
    re.compile(r'\*\*DECISION\*\*\s*:\s*\*\*REJECT', re.IGNORECASE),
    re.compile(r'false\s+positive', re.IGNORECASE),
    re.compile(r'allowing\s+future\s+\w+', re.IGNORECASE),
]

_EXCLUSION_SECTION_HEADERS = [
    re.compile(r'^#+\s*.*(?:out\s+of\s+scope|future\s+consideration|not\s+in\s+(?:scope|phase)|excluded|deferred|limitation)', re.IGNORECASE),
    re.compile(r'^#+\s*.*(?:revision\s+(?:notes?|log|history))', re.IGNORECASE),
    re.compile(r'^\d+\.\s*(?:future\s+consideration|out\s+of\s+scope|revision\s+(?:notes?|log))', re.IGNORECASE),
    re.compile(r'^\*\*(?:out\s+of\s+scope|future|excluded|not\s+in\s+phase)', re.IGNORECASE),
    re.compile(r'^#+\s*.*(?:reviewer\s+suggestion|revision\s+response|critique\s+rebuttal|spec[_-]?compliance)', re.IGNORECASE),
    re.compile(r'^#+\s*.*(?:platform\s+mismatch\s+(?:claim|analysis))', re.IGNORECASE),
]

_SECTION_HEADER_RE = re.compile(r'^(?:#{1,6}\s|\d+\.\s)')

_DESKTOP_DEFINITIVE_STACKS = {'TypeScript/Electron', 'Python+PyQt', 'Python+Tkinter', 'C#'}

_MOBILE_DEFINITIVE_STACKS = {'React Native', 'Flutter', 'Kotlin', 'Swift'}

_WORD_BOUNDARY_KEYWORDS = {"rust", "java", "vue"}
