from app.llm.chat_tool_loop import get_chat_tools
import json
tools = get_chat_tools()
with open(r'D:\Orb\logs\tool_test.txt', 'w') as f:
    f.write(f'Total: {len(tools)}\n')
    f.write(json.dumps(tools[0], indent=2))
