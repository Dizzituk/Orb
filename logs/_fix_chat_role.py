import ast

with open(r'D:\Orb\app\llm\routing\chat_routing.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = '            _chat_tools = get_chat_tools()\n            print(f"[CHAT_MODE] Tool access ENABLED for {provider}/{model} ({len(_chat_tools)} tools)")'

new_block = '''            _chat_tools = get_chat_tools()
            print(f"[CHAT_MODE] Tool access ENABLED for {provider}/{model} ({len(_chat_tools)} tools)")
            # v8.1: Inject research-only role into system prompt for tool-enabled chat
            _TOOL_ROLE_BLOCK = (
                "\\n\\n## TOOL ACCESS -- RESEARCH MODE\\n"
                "You have READ-ONLY tool access (read_file, list_files, search_files, read_logs).\\n"
                "Use these tools to explore the codebase and gather information.\\n\\n"
                "YOUR ROLE: You are a RESEARCHER, not a builder.\\n"
                "- Explore files, read code, understand patterns, discover design tokens\\n"
                "- Report your findings as text in the chat -- describe what you found\\n"
                "- Present component structures, CSS variables, layout patterns, file paths\\n"
                "- This research will be picked up by the Weaver to create accurate build specs\\n\\n"
                "DO NOT:\\n"
                "- Generate code blocks, full file contents, or implementation files\\n"
                "- Try to create, write, or modify any files\\n"
                "- Produce implementation plans or architecture documents\\n"
                "- Dump raw file contents -- summarise and highlight the relevant patterns\\n\\n"
                "GOOD OUTPUT: Describe patterns, tokens, and structures you found.\\n"
                "BAD OUTPUT: Producing hundreds of lines of implementation code.\\n"
            )
            system_prompt += _TOOL_ROLE_BLOCK'''

if old in content:
    content = content.replace(old, new_block, 1)
    print('CHAT_MODE fix: applied')
else:
    print('CHAT_MODE fix: old text NOT FOUND')
    # Debug: show what's actually there
    idx = content.find('_chat_tools = get_chat_tools()')
    if idx >= 0:
        print(f'Found at index {idx}')
        print(repr(content[idx:idx+200]))

with open(r'D:\Orb\app\llm\routing\chat_routing.py', 'w', encoding='utf-8') as f:
    f.write(content)

ast.parse(content)
print('AST: VALID')
print(f'Size: {len(content)/1024:.1f} KB')
