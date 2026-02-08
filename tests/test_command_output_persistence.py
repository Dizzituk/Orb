import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from app.llm.routing.command_output_persistence import wrap_sse_and_persist
from app.memory import schemas


@pytest.fixture
def mock_db():
    """Mock database session."""
    return Mock()


@pytest.fixture
def mock_create_message():
    """Mock create_message function."""
    with patch('app.llm.routing.command_output_persistence.create_message') as mock:
        yield mock


async def mock_sse_generator(events):
    """Helper to create mock SSE generators."""
    for event in events:
        yield f"data: {json.dumps(event)}\n\n"


@pytest.mark.asyncio
async def test_wrap_sse_and_persist_basic(mock_db, mock_create_message):
    """Test basic wrapping and persistence."""
    events = [
        {"type": "token", "text": "Hello "},
        {"type": "token", "text": "world"},
        {"type": "done"}
    ]
    
    gen = mock_sse_generator(events)
    wrapped = wrap_sse_and_persist(
        gen=gen,
        db=mock_db,
        project_id=123,
        provider="openai",
        model="gpt-4"
    )
    
    # Consume wrapped generator
    chunks = [chunk async for chunk in wrapped]
    
    # Verify chunks forwarded unchanged
    assert len(chunks) == 3
    
    # Verify create_message called once
    assert mock_create_message.call_count == 1
    
    # Verify message content
    call_args = mock_create_message.call_args[0]
    msg_create = call_args[1]
    assert msg_create.project_id == 123
    assert msg_create.role == "assistant"
    assert msg_create.content == "Hello world"
    assert msg_create.provider == "openai"
    assert msg_create.model == "gpt-4"


@pytest.mark.asyncio
async def test_wrap_sse_and_persist_with_metadata(mock_db, mock_create_message):
    """Test persistence with provider and model metadata."""
    events = [
        {"type": "token", "text": "Test ", "provider": "anthropic", "model": "claude-3"},
        {"type": "token", "text": "output"},
        {"type": "done"}
    ]
    
    gen = mock_sse_generator(events)
    wrapped = wrap_sse_and_persist(
        gen=gen,
        db=mock_db,
        project_id=456,
        provider="openai",
        model="gpt-4"
    )
    
    chunks = [chunk async for chunk in wrapped]
    
    # Verify message persisted with metadata
    call_args = mock_create_message.call_args[0]
    msg_create = call_args[1]
    assert msg_create.provider == "openai"
    assert msg_create.model == "gpt-4"


@pytest.mark.asyncio
async def test_wrap_sse_and_persist_empty_output(mock_db, mock_create_message):
    """Test persistence with no token events."""
    events = [
        {"type": "done"}
    ]
    
    gen = mock_sse_generator(events)
    wrapped = wrap_sse_and_persist(
        gen=gen,
        db=mock_db,
        project_id=123,
        provider="openai",
        model="gpt-4"
    )
    
    chunks = [chunk async for chunk in wrapped]
    
    # Verify message persisted with empty content
    call_args = mock_create_message.call_args[0]
    msg_create = call_args[1]
    assert msg_create.content == ""


@pytest.mark.asyncio
async def test_wrap_sse_and_persist_on_error(mock_db, mock_create_message):
    """Test persistence when generator raises error."""
    async def error_generator():
        yield f"data: {json.dumps({'type': 'token', 'text': 'Partial '})}\n\n"
        raise ValueError("Stream error")
    
    gen = error_generator()
    wrapped = wrap_sse_and_persist(
        gen=gen,
        db=mock_db,
        project_id=123,
        provider="openai",
        model="gpt-4"
    )
    
    # Consume generator, expect error
    with pytest.raises(ValueError):
        chunks = [chunk async for chunk in wrapped]
    
    # Verify partial content persisted
    assert mock_create_message.call_count == 1
    call_args = mock_create_message.call_args[0]
    msg_create = call_args[1]
    assert msg_create.content == "Partial "


@pytest.mark.asyncio
async def test_wrap_sse_and_persist_non_json_chunks(mock_db, mock_create_message):
    """Test handling of non-JSON SSE chunks."""
    async def mixed_generator():
        yield f"data: {json.dumps({'type': 'token', 'text': 'Valid '})}\n\n"
        yield "invalid line without data prefix\n\n"
        yield f"data: {json.dumps({'type': 'token', 'text': 'token'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    gen = mixed_generator()
    wrapped = wrap_sse_and_persist(
        gen=gen,
        db=mock_db,
        project_id=123,
        provider="openai",
        model="gpt-4"
    )
    
    chunks = [chunk async for chunk in wrapped]
    
    # Verify all chunks forwarded
    assert len(chunks) == 4
    
    # Verify only valid tokens accumulated
    call_args = mock_create_message.call_args[0]
    msg_create = call_args[1]
    assert msg_create.content == "Valid token"


@pytest.mark.asyncio
async def test_wrap_sse_and_persist_malformed_json(mock_db, mock_create_message):
    """Test handling of malformed JSON in data chunks."""
    async def malformed_generator():
        yield f"data: {json.dumps({'type': 'token', 'text': 'Good'})}\n\n"
        yield "data: {invalid json}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'text': ' text'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    gen = malformed_generator()
    wrapped = wrap_sse_and_persist(
        gen=gen,
        db=mock_db,
        project_id=123,
        provider="openai",
        model="gpt-4"
    )
    
    chunks = [chunk async for chunk in wrapped]
    
    # Verify malformed chunk skipped but others processed
    call_args = mock_create_message.call_args[0]
    msg_create = call_args[1]
    assert msg_create.content == "Good text"


@pytest.mark.asyncio
async def test_wrap_sse_and_persist_fallback_metadata(mock_db, mock_create_message):
    """Test fallback to default provider/model when not provided."""
    events = [
        {"type": "token", "text": "Output"},
        {"type": "done"}
    ]
    
    gen = mock_sse_generator(events)
    wrapped = wrap_sse_and_persist(
        gen=gen,
        db=mock_db,
        project_id=123
    )
    
    chunks = [chunk async for chunk in wrapped]
    
    # Verify fallback metadata used
    call_args = mock_create_message.call_args[0]
    msg_create = call_args[1]
    assert msg_create.provider == "local"
    assert msg_create.model == "command_router"


@pytest.mark.asyncio
async def test_wrap_sse_and_persist_db_failure(mock_db, mock_create_message):
    """Test that DB failures don't crash stream."""
    mock_create_message.side_effect = Exception("DB error")
    
    events = [
        {"type": "token", "text": "Test"},
        {"type": "done"}
    ]
    
    gen = mock_sse_generator(events)
    wrapped = wrap_sse_and_persist(
        gen=gen,
        db=mock_db,
        project_id=123,
        provider="openai",
        model="gpt-4"
    )
    
    # Stream should complete despite DB error
    chunks = [chunk async for chunk in wrapped]
    assert len(chunks) == 2
    
    # Verify persistence was attempted
    assert mock_create_message.call_count == 1


@pytest.mark.asyncio
async def test_wrap_sse_and_persist_multiple_token_types(mock_db, mock_create_message):
    """Test accumulation of only 'token' type events."""
    events = [
        {"type": "status", "message": "Starting..."},
        {"type": "token", "text": "Hello "},
        {"type": "metadata", "key": "value"},
        {"type": "token", "text": "world"},
        {"type": "done"}
    ]
    
    gen = mock_sse_generator(events)
    wrapped = wrap_sse_and_persist(
        gen=gen,
        db=mock_db,
        project_id=123,
        provider="openai",
        model="gpt-4"
    )
    
    chunks = [chunk async for chunk in wrapped]
    
    # All chunks forwarded
    assert len(chunks) == 5
    
    # Only token events accumulated
    call_args = mock_create_message.call_args[0]
    msg_create = call_args[1]
    assert msg_create.content == "Hello world"


@pytest.mark.asyncio
async def test_wrap_sse_and_persist_large_output(mock_db, mock_create_message):
    """Test persistence of large command output."""
    large_text = "x" * 15000
    events = [
        {"type": "token", "text": large_text},
        {"type": "done"}
    ]
    
    gen = mock_sse_generator(events)
    wrapped = wrap_sse_and_persist(
        gen=gen,
        db=mock_db,
        project_id=123,
        provider="openai",
        model="gpt-4"
    )
    
    chunks = [chunk async for chunk in wrapped]
    
    # Verify full content persisted (no truncation at persistence layer)
    call_args = mock_create_message.call_args[0]
    msg_create = call_args[1]
    assert len(msg_create.content) == 15000


@pytest.mark.asyncio
async def test_wrap_sse_and_persist_unicode_content(mock_db, mock_create_message):
    """Test handling of Unicode characters in output."""
    events = [
        {"type": "token", "text": "Hello 世界 🌍"},
        {"type": "done"}
    ]
    
    gen = mock_sse_generator(events)
    wrapped = wrap_sse_and_persist(
        gen=gen,
        db=mock_db,
        project_id=123,
        provider="openai",
        model="gpt-4"
    )
    
    chunks = [chunk async for chunk in wrapped]
    
    # Verify Unicode preserved
    call_args = mock_create_message.call_args[0]
    msg_create = call_args[1]
    assert msg_create.content == "Hello 世界 🌍"


@pytest.mark.asyncio
async def test_wrap_sse_and_persist_generator_disconnect(mock_db, mock_create_message):
    """Test persistence when generator disconnects without done event."""
    async def disconnect_generator():
        yield f"data: {json.dumps({'type': 'token', 'text': 'Incomplete'})}\n\n"
        # Generator ends without 'done' event
    
    gen = disconnect_generator()
    wrapped = wrap_sse_and_persist(
        gen=gen,
        db=mock_db,
        project_id=123,
        provider="openai",
        model="gpt-4"
    )
    
    chunks = [chunk async for chunk in wrapped]
    
    # Verify partial content persisted even without done event
    assert mock_create_message.call_count == 1
    call_args = mock_create_message.call_args[0]
    msg_create = call_args[1]
    assert msg_create.content == "Incomplete"