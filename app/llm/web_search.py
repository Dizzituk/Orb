# app/llm/web_search.py
"""
Web search grounding using Gemini.
Uses the new google-genai SDK with google_search tool.
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, List

# Try new SDK first (google-genai), fall back to old SDK (google-generativeai)
HAS_NEW_SDK = False
HAS_OLD_SDK = False

try:
    from google import genai
    from google.genai import types
    HAS_NEW_SDK = True
    print("[web_search] Using new google-genai SDK")
except ImportError:
    try:
        import google.generativeai as genai_old
        HAS_OLD_SDK = True
        print("[web_search] Using old google-generativeai SDK")
    except ImportError:
        print("[web_search] No Gemini SDK installed")


def is_web_search_available() -> bool:
    """Check if web search is available."""
    return (HAS_NEW_SDK or HAS_OLD_SDK) and bool(os.getenv("GOOGLE_API_KEY"))


def get_current_datetime() -> str:
    """Get current date and time formatted for prompt."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


def search_and_answer(
    query: str,
    context: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Perform a web-grounded search query using Gemini.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {
            "answer": "Web search unavailable: GOOGLE_API_KEY not set",
            "sources": [],
            "search_queries": [],
            "provider": "none",
            "error": "missing_api_key",
        }
    
    if HAS_NEW_SDK:
        return _search_with_new_sdk(query, context, history, api_key)
    elif HAS_OLD_SDK:
        return _search_with_old_sdk(query, context, history, api_key)
    else:
        return {
            "answer": "Web search unavailable: No Gemini SDK installed. Run: pip install google-genai",
            "sources": [],
            "search_queries": [],
            "provider": "none",
            "error": "missing_dependency",
        }


def _search_with_new_sdk(
    query: str,
    context: Optional[str],
    history: Optional[List[Dict[str, str]]],
    api_key: str,
) -> Dict[str, Any]:
    """Use new google-genai SDK with google_search tool."""
    try:
        from google import genai
        from google.genai import types
        
        # Initialize client with API key
        client = genai.Client(api_key=api_key)
        
        # Create Google Search tool
        google_search_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        # Get current date/time
        current_datetime = get_current_datetime()
        
        # Build prompt with current date/time
        prompt = f"""Current date and time: {current_datetime}

Search the web for the most up-to-date information and answer the following question.
Make sure to find results from today or the most recent available data.

Question: {query}"""
        
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        print(f"[web_search] Querying with google_search tool: {query[:80]}...")
        print(f"[web_search] Current datetime: {current_datetime}")
        
        # Generate with search grounding
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[google_search_tool],
            ),
        )
        
        # Extract answer
        answer = response.text if response.text else "No response generated"
        
        # Extract grounding metadata
        sources = []
        search_queries = []
        
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                metadata = candidate.grounding_metadata
                
                # Get search queries
                if hasattr(metadata, 'web_search_queries'):
                    search_queries = list(metadata.web_search_queries or [])
                    print(f"[web_search] Search queries: {search_queries}")
                
                # Get sources from grounding_chunks
                if hasattr(metadata, 'grounding_chunks'):
                    for chunk in (metadata.grounding_chunks or []):
                        if hasattr(chunk, 'web') and chunk.web:
                            sources.append({
                                "title": getattr(chunk.web, 'title', 'Source'),
                                "uri": getattr(chunk.web, 'uri', ''),
                            })
        
        print(f"[web_search] Success: {len(sources)} sources found")
        
        return {
            "answer": answer,
            "sources": sources,
            "search_queries": search_queries,
            "provider": "gemini-grounded",
        }
        
    except Exception as e:
        print(f"[web_search] New SDK error: {type(e).__name__}: {e}")
        return {
            "answer": f"Web search failed: {str(e)}",
            "sources": [],
            "search_queries": [],
            "provider": "gemini",
            "error": str(e),
        }


def _search_with_old_sdk(
    query: str,
    context: Optional[str],
    history: Optional[List[Dict[str, str]]],
    api_key: str,
) -> Dict[str, Any]:
    """Use old google-generativeai SDK."""
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
        )
        
        current_datetime = get_current_datetime()
        
        prompt = f"""Current date and time: {current_datetime}

Search the web for the most up-to-date information and answer: {query}"""
        
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        print(f"[web_search] Using old SDK (no grounding): {query[:80]}...")
        
        response = model.generate_content(prompt)
        
        return {
            "answer": response.text if response.text else "No response",
            "sources": [],
            "search_queries": [],
            "provider": "gemini-basic",
            "note": "Grounding requires google-genai SDK. Install with: pip install google-genai",
        }
        
    except Exception as e:
        print(f"[web_search] Old SDK error: {e}")
        return {
            "answer": f"Web search failed: {str(e)}",
            "sources": [],
            "search_queries": [],
            "provider": "none",
            "error": str(e),
        }


def format_sources_markdown(sources: List[Dict[str, str]]) -> str:
    """Format sources as markdown links."""
    if not sources:
        return ""
    
    lines = ["\n\n**Sources:**"]
    for src in sources[:5]:
        title = src.get("title", "Link")
        uri = src.get("uri", "")
        if uri:
            lines.append(f"- [{title}]({uri})")
        else:
            lines.append(f"- {title}")
    
    return "\n".join(lines)
