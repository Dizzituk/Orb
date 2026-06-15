import re
from typing import List, Set

# ---- EXACT COPY from app/llm/routing/rag_fallback.py ----
_ARCHITECTURE_QUERY_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"^[Ww]here\s+is\s+(?:the\s+)?(?:main\s+)?(?:\w+\s+){0,6}"
        r"(?:entrypoint|entry\s*point|router|stream|handler|function|class|module|file|config|constant|routing|implementation|trigger|pipeline|gate)[s]?"
    ),
    re.compile(
        r"^[Ss]how\s+(?:me\s+)?where\s+.+"
        r"(?:is\s+)?(?:implemented|defined|located|triggered|called|used|loaded|handled|routed|processed)"
    ),
    re.compile(
        r"^[Ff]ind\s+(?:where|the\s+file\s+where)\s+.+"
        r"(?:is\s+)?(?:implemented|defined|located|triggered|called|used|loaded|handled|routed|processed|routes)"
    ),
    re.compile(
        r"^[Ff]ind\s+(?:the\s+)?(?:file|module|class|function)\s+(?:where|that)\s+.+"
    ),
    re.compile(
        r"^(?:[Ff]ind|[Ll]ist|[Ss]how)\s+(?:the\s+)?(?:call\s*sites?|callers?)\s+(?:of|for)\s+.+"
    ),
    re.compile(r"^[Ww]ho\s+calls\s+.+"),
    re.compile(
        r"^(?:[Ww]here|[Hh]ow|[Ww]hat)\s+.+"
        r"(?:[Ss]pec\s*[Gg]ate|[Oo]verwatcher|[Ww]eaver|[Cc]ritical\s*[Pp]ipeline|"
        r"[Ss]tream\s*[Rr]outer|[Tt]ranslation\s*[Ll]ayer|[Rr][Aa][Gg]|[Ee]mbedding)"
        r"\s*.+[?.!]?$"
    ),
]

_NATURAL_CODEBASE_PATTERNS_FB: List[re.Pattern] = [
    re.compile(r"^[Hh]ow\s+many\s+(?:times?\s+)?(?:does|do|is|are)\s+.+", re.IGNORECASE),
    re.compile(
        r"^[Hh]ow\s+(?:hard|difficult|easy|complex)\s+would\s+it\s+be\s+to\s+"
        r"(?:refactor|rename|change|replace|update|migrate|remove|extract|split|merge|consolidate)\s+.+",
        re.IGNORECASE,
    ),
    re.compile(r"^[Ww]hat\s+does?\s+(?:the\s+)?(?:\w+\s+){0,3}(?:do|handle|manage|control|process|route)\s*\??", re.IGNORECASE),
    re.compile(r"^[Hh]ow\s+does\s+(?:the\s+)?(?:\w+\s+){0,3}(?:work|function|operate|run)\s*\??", re.IGNORECASE),
    re.compile(r"^[Tt]ell\s+me\s+about\s+(?:the\s+)?.+", re.IGNORECASE),
    re.compile(r"^[Ee]xplain\s+(?:the\s+)?(?:how\s+)?.+", re.IGNORECASE),
    re.compile(r"^[Ww]hat\s+happens\s+(?:when|if)\s+.+", re.IGNORECASE),
    re.compile(r"^[Ww]here\s+(?:does|do|is|are)\s+.+", re.IGNORECASE),
    re.compile(r"^[Ww]hich\s+(?:files?|modules?|functions?|classes?|components?)\s+.+", re.IGNORECASE),
]

_CODEBASE_GUARD_KEYWORDS: Set[str] = {
    "weaver", "specgate", "spec gate", "overwatcher", "sandbox",
    "translation", "pipeline", "implementer",
    "memory", "rag", "confidence", "preference", "context",
    "knowledge", "embedding", "store", "domain store",
    "codebase", "module", "file", "function", "class", "method",
    "import", "dependency", "endpoint", "handler", "router",
    "provider", "registry", "stream", "hook", "tier",
    "refactor", "rename", "migrate", "extract", "decompose",
    "architecture", "subsystem", "component",
    "database", "table", "sqlite", "orb.db",
}

def has_guard(text):
    tl = text.lower()
    hits = []
    for kw in _CODEBASE_GUARD_KEYWORDS:
        if " " in kw:
            if kw in tl:
                hits.append(kw)
        else:
            if re.search(r'\b' + re.escape(kw) + r'\b', tl):
                hits.append(kw)
    return hits

_COMMAND_PREFIX_PATTERN = re.compile(r"^[Aa]stra[,:]?\s*command:\s*", re.IGNORECASE)

def is_architecture_query(message):
    text = message.strip()
    if _COMMAND_PREFIX_PATTERN.match(text):
        return ("CMD-skip", None)
    for i, p in enumerate(_ARCHITECTURE_QUERY_PATTERNS):
        if p.match(text):
            return ("FB-stage1", "arch_pattern[%d]" % i)
    for i, p in enumerate(_NATURAL_CODEBASE_PATTERNS_FB):
        if p.match(text):
            g = has_guard(text)
            if g:
                return ("FB-stage2", "nat_pattern[%d] guard=%s" % (i, g))
            return (False, "nat_pattern[%d] matched but NO guard -> break" % i)
    return (False, "no FB pattern")

# ---- EXACT COPY from app/translation/_tier0_codebase_questions.py ----
def t0_rigid(text):
    text_stripped = text.strip()
    text_lower = text_stripped.lower()
    text_clean = text_lower.rstrip('?.!')
    index_patterns = [r"^index\s+(?:the\s+)?(?:architecture|codebase|rag)$", r"^run\s+rag\s+index$"]
    for p in index_patterns:
        if re.match(p, text_clean):
            return ("T0-rigid", "rag_index")
    explicit_search_patterns = [
        r"^search\s+(?:the\s+)?codebase:\s*.+",
        r"^ask\s+about\s+(?:the\s+)?codebase:\s*.+",
        r"^codebase\s+(?:search|query):\s*.+",
        r"^in\s+(?:the|this)\s+codebase,?\s+.+",
    ]
    for p in explicit_search_patterns:
        if re.match(p, text_lower):
            return ("T0-rigid", "rag_explicit_search")
    architecture_question_patterns = [
        r"^what\s+(?:are|is)\s+(?:the\s+)?(?:main\s+)?entry\s*points?$",
        r"^where\s+(?:are|is)\s+(?:the\s+)?(?:main\s+)?entry\s*points?$",
        r"^show\s+(?:me\s+)?(?:the\s+)?entry\s*points?$",
        r"^(?:list|find)\s+(?:the\s+)?entry\s*points?$",
        r"^what\s+(?:are|is)\s+(?:the\s+)?(?:potential\s+)?bottlenecks?$",
        r"^where\s+(?:are|is)\s+(?:the\s+)?bottlenecks?$",
        r"^(?:find|identify|show)\s+(?:the\s+)?bottlenecks?$",
        r"^bottlenecks$",
        r"^(?:which|what)\s+modules?\s+(?:are|is)\s+(?:tightly\s+)?coupled",
        r"^where\s+(?:is|are)\s+(?:the\s+)?tight\s+coupling",
        r"^show\s+(?:me\s+)?(?:the\s+)?dependencies",
        r"^where\s+should\s+(?:a\s+)?(?:new\s+)?(?:feature|functionality|code|module|component)\s+.+\s+(?:live|go)",
        r"^where\s+should\s+I\s+(?:put|add|place|implement)\s+.+",
        r"^where\s+does\s+.+\s+(?:belong|go|fit)",
        r"^what\s+(?:function|class|method|module|file)s?\s+(?:handle|process|manage|control)s?\s+.+",
        r"^(?:which|what)\s+(?:function|class|method|module|file)s?\s+(?:are\s+)?responsible\s+for\s+.+",
        r"^(?:which|what)\s+(?:function|class|method|module|file)s?\s+(?:do|does)\s+.+",
        r"^where\s+is\s+.+\s+(?:implemented|defined|located|declared|found)",
        r"^find\s+(?:the\s+)?(?:implementation|definition|location)\s+of\s+.+",
        r"^(?:locate|find)\s+.+\s+(?:in\s+the\s+codebase|code)",
        r"^how\s+does\s+(?:the\s+)?(?:routing|streaming|pipeline|job|auth|memory|rag)\s+(?:work|function|flow)",
        r"^explain\s+(?:the\s+)?(?:routing|streaming|pipeline|job|auth|memory|rag)\s+(?:flow|system|architecture)",
        r"^how\s+(?:would|should|could)\s+(?:you|I|we)\s+optimize\s+.+",
        r"^(?:suggest|recommend)\s+(?:optimizations?|improvements?)\s+(?:for|to)\s+.+",
        r"^what\s+is\s+the\s+(?:purpose|role|responsibility)\s+of\s+.+",
        r"^what\s+does\s+(?:the\s+)?(?:file|module|class|function)\s+.+\s+do",
        r"^describe\s+(?:the\s+)?(?:file|module|class|function|component)\s+.+",
        r"^(?:list|show|display)\s+(?:all\s+)?(?:the\s+)?(?:modules?|components?|services?|handlers?|routers?)$",
        r"^what\s+(?:modules?|components?|services?)\s+(?:exist|are\s+there)",
    ]
    for idx, p in enumerate(architecture_question_patterns):
        if re.match(p, text_clean):
            return ("T0-rigid", "rag_architecture_question[%d]" % idx)
    return (False, None)

_NAT_T0 = [
    re.compile(r"^[Hh]ow\s+many\s+(?:times?\s+)?(?:does|do|is|are)\s+.+", re.IGNORECASE),
    re.compile(r"^[Hh]ow\s+(?:hard|difficult|easy|complex)\s+would\s+it\s+be\s+to\s+(?:refactor|rename|change|replace|update|migrate|remove|extract|split|merge|consolidate)\s+.+", re.IGNORECASE),
    re.compile(r"^[Ww]hat\s+does?\s+(?:the\s+)?(?:\w+\s+){0,3}(?:do|handle|manage|control|process|route)\s*\??", re.IGNORECASE),
    re.compile(r"^[Hh]ow\s+does\s+(?:the\s+)?(?:\w+\s+){0,3}(?:work|function|operate|run)\s*\??", re.IGNORECASE),
    re.compile(r"^[Tt]ell\s+me\s+about\s+(?:the\s+)?.+", re.IGNORECASE),
    re.compile(r"^[Ee]xplain\s+(?:the\s+)?(?:how\s+)?.+", re.IGNORECASE),
    re.compile(r"^[Cc]an\s+you\s+(?:show|tell|explain)\s+.+", re.IGNORECASE),
    re.compile(r"^[Ww]hat\s+happens\s+(?:when|if)\s+.+", re.IGNORECASE),
    re.compile(r"^[Ww]here\s+(?:does|do|is|are)\s+.+", re.IGNORECASE),
    re.compile(r"^[Ww]hich\s+(?:files?|modules?|functions?|classes?|components?)\s+.+", re.IGNORECASE),
]

def t0_natural(text):
    ts = text.strip()
    mp = None
    for i, p in enumerate(_NAT_T0):
        if p.match(ts):
            mp = i
            break
    if mp is None:
        return (False, "no T0 natural pattern")
    g = has_guard(ts)
    if not g:
        return (False, "T0-nat_pattern[%d] matched but NO guard -> defer Tier1" % mp)
    return ("T0-natural", "nat_pattern[%d] guard=%s" % (mp, g))

def evaluate(q):
    r1 = t0_rigid(q)
    r2 = t0_natural(q)
    r3 = is_architecture_query(q)
    routes = bool(r1[0]) or bool(r2[0]) or bool(r3[0])
    return routes, r1, r2, r3

questions = [
 "How does the memory system work?",
 "What data does the chat window have?",
 "What data does the work table show?",
 "What can the finance panel do?",
 "Tell me about your memory.",
 "How does the meal planner work?",
 "What are your abilities?",
 "Explain the build pipeline.",
 "What does the orchestrator do?",
 "What data does the sidebar have?",
 "How does the Weaver work?",
 "Where does the food log store its data?",
]

for q in questions:
    routes, r1, r2, r3 = evaluate(q)
    print("=" * 90)
    print("Q: %s" % q)
    print("  guard hits   : %s" % has_guard(q))
    print("  T0 rigid     : %s" % (r1,))
    print("  T0 natural   : %s" % (r2,))
    print("  RAG fallback : %s" % (r3,))
    print("  >>> ROUTES_TO_RAG = %s" % routes)
