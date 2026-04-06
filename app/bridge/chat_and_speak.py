"""
Bridge chat-and-speak pipeline.

Single endpoint that:
  1. Streams LLM tokens as they generate
  2. Accumulates into complete sentences
  3. Feeds each sentence to Chatterbox /tts/sentence immediately
  4. Yields MP3 bytes as a single chunked audio/mpeg stream
  5. Appends the full text as a trailing metadata chunk

The phone receives one HTTP response:
  - Content-Type: audio/mpeg
  - Transfer-Encoding: chunked
  - X-Full-Text header with the complete reply (URL-encoded)
  - Body: growing MP3 stream that can be played progressively

This eliminates the sequential bottleneck of:
  old: LLM (5-12s) → full text → TTS (20-45s) → buffer → play
  new: LLM sentence 1 (1-2s) → TTS sentence 1 (3-5s) → play
       while LLM generates sentence 2, 3, ... in parallel
"""

import asyncio
import logging
import re
import time
from urllib.parse import quote as url_quote

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

TTS_BASE = "http://127.0.0.1:8002"

# Sentence boundary pattern: split after . ? ! followed by space or EOL
# Also splits on newlines (paragraph breaks).
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+|(?<=\n)')


class ChatAndSpeakRequest(BaseModel):
    message: str
    project_id: int | None = None


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences at . ? ! boundaries.

    Returns a list of non-empty sentences. If text has no sentence
    boundaries, returns the whole text as one sentence.
    """
    parts = _SENTENCE_END.split(text)
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences


async def _stream_llm_tokens(provider: str, model: str, messages: list):
    """Yield text chunks as the LLM generates them.

    Uses the streaming API for each provider so we get tokens
    as they're produced rather than waiting for the full response.
    """
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=2048,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg += m["content"] + "\n"
            else:
                chat_messages.append(m)
        with client.messages.stream(
            model=model,
            max_tokens=2048,
            system=system_msg.strip(),
            messages=chat_messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    elif provider == "google":
        import google.generativeai as genai
        gmodel = genai.GenerativeModel(model)
        history = []
        system_text = ""
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            elif m["role"] == "user":
                history.append({"role": "user", "parts": [m["content"]]})
            elif m["role"] == "assistant":
                history.append({"role": "model", "parts": [m["content"]]})
        if system_text and history:
            history[0]["parts"].insert(0, system_text.strip() + "\n\n")
        chat = gmodel.start_chat(history=history[:-1] if history else [])
        last_msg = history[-1]["parts"][0] if history else "Hello"
        response = chat.send_message(last_msg, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def _synthesise_sentence(sentence: str) -> bytes:
    """Send a single sentence to Chatterbox and return the MP3 bytes."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{TTS_BASE}/tts/sentence",
            json={"text": sentence},
        )
        resp.raise_for_status()
        return resp.content


async def _pipeline_generate(
    provider: str,
    model: str,
    messages: list,
):
    """Core pipeline generator.

    Yields MP3 bytes as sentences complete. The LLM streams tokens,
    we accumulate into sentences, and each sentence is synthesised
    immediately. Callers receive a growing MP3 stream.

    Also yields a final metadata dict with the full text after all
    audio is done (signalled by yielding a bytes object vs dict).
    """
    full_text = ""
    buffer = ""
    sentence_count = 0
    total_audio_bytes = 0
    t_start = time.perf_counter()

    logger.info("[pipeline] Starting LLM stream (%s/%s)", provider, model)

    async for token in _stream_llm_tokens(provider, model, messages):
        full_text += token
        buffer += token

        # Check if we have complete sentences in the buffer
        sentences = _split_into_sentences(buffer)
        if len(sentences) > 1:
            # All but the last are complete sentences
            for sentence in sentences[:-1]:
                sentence_count += 1
                logger.info("[pipeline] Sentence %d: %s...",
                           sentence_count, sentence[:60])
                try:
                    mp3_bytes = await _synthesise_sentence(sentence)
                    total_audio_bytes += len(mp3_bytes)
                    logger.info("[pipeline] Sentence %d: %dKB audio",
                               sentence_count, len(mp3_bytes) // 1024)
                    yield mp3_bytes
                except Exception as e:
                    logger.error("[pipeline] TTS failed for sentence %d: %s",
                                sentence_count, e)
            # Keep the incomplete last part
            buffer = sentences[-1]

    # Flush remaining buffer (last sentence without trailing punctuation)
    if buffer.strip():
        sentence_count += 1
        logger.info("[pipeline] Final sentence %d: %s...",
                   sentence_count, buffer.strip()[:60])
        try:
            mp3_bytes = await _synthesise_sentence(buffer.strip())
            total_audio_bytes += len(mp3_bytes)
            logger.info("[pipeline] Final sentence %d: %dKB audio",
                       sentence_count, len(mp3_bytes) // 1024)
            yield mp3_bytes
        except Exception as e:
            logger.error("[pipeline] TTS failed for final sentence: %s", e)

    elapsed = time.perf_counter() - t_start
    logger.info("[pipeline] Complete: %d sentences, %dKB audio, %.1fs total",
               sentence_count, total_audio_bytes // 1024, elapsed)

    # Signal the full text for the caller to use in headers/metadata
    yield {"full_text": full_text, "sentences": sentence_count}
