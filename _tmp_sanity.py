import sys
sys.path.insert(0, 'D:/Orb')

from app.db import Base
import app.learning.models
from app.debug.tool_definitions import get_phase1_tools
from app.debug.action_executor import TOOL_HANDLERS
from app.web_automation.tool_schemas import TOOL_SCHEMAS

# Check learning tables
learning_tables = [t for t in Base.metadata.tables if t.startswith("course_")]
print(f"learning tables: {learning_tables}")

# Check chat LLM sees web tools
tools = get_phase1_tools()
web = [t["name"] for t in tools if t["name"].startswith("web_")]
print(f"LLM sees {len(web)} web_* tools")

# Check handlers
web_handlers = [k for k in TOOL_HANDLERS if k.startswith("web_") and k != "web_search"]
print(f"TOOL_HANDLERS has {len(web_handlers)} web browsing handlers")

# web_dom_snapshot specifically
has_dom = "web_dom_snapshot" in TOOL_SCHEMAS
print(f"web_dom_snapshot schema exists: {has_dom}")
has_dom_handler = "web_dom_snapshot" in TOOL_HANDLERS
print(f"web_dom_snapshot handler exists: {has_dom_handler}")
