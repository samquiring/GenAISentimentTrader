"""Unit tests for DataSink."""

import asyncio
from uuid import uuid4

import pytest

from genaisentimenttrader.data_sink.sink import DataSink
from genaisentimenttrader.models.types import ContentItem, ContentType


class TestDataSink:
    """Test the DataSink class."""

    @pytest.fixture
    def data_sink(self):
        """Create a DataSink instance for testing."""
        return DataSink(max_buffer_size=5)

    @pytest.fixture
    def sample_content_items(self):
        """Create sample ContentItem objects for testing."""
        return [
            ContentItem(
                content_type=ContentType.TWEET,
                text=f"Sample tweet content {i}",
                source_url=f"https://example.com/tweet/{i}",
                author=f"user_{i}",
            )
            for i in range(3)
        ]

    def test_init(self):
        """Test DataSink initialization."""
        sink = DataSink(max_buffer_size=100)
        assert sink._max_buffer_size == 100
        assert len(sink._buffer) == 0
        assert sink._lock is not None
        assert sink._new_item_event is not None

    def test_init_default_buffer_size(self):
        """Test DataSink initialization with default buffer size."""
        sink = DataSink()
        assert sink._max_buffer_size == 1000

    @pytest.mark.asyncio
    async def test_add_item(self, data_sink, sample_content_items):
        """Test adding a single item to the sink."""
        item = sample_content_items[0]

        await data_sink.add_item(item)

        assert await data_sink.size() == 1
        assert not await data_sink.is_empty()

        # Verify the item is in the buffer
        peeked_items = await data_sink.peek_items(1)
        assert len(peeked_items) == 1
        assert peeked_items[0].id == item.id

    @pytest.mark.asyncio
    async def test_add_items(self, data_sink, sample_content_items):
        """Test adding multiple items to the sink."""
        await data_sink.add_items(sample_content_items)

        assert await data_sink.size() == len(sample_content_items)
        assert not await data_sink.is_empty()

    @pytest.mark.asyncio
    async def test_get_item(self, data_sink, sample_content_items):
        """Test getting a single item from the sink."""
        item = sample_content_items[0]
        await data_sink.add_item(item)

        retrieved_item = await data_sink.get_item()

        assert retrieved_item is not None
        assert retrieved_item.id == item.id
        assert retrieved_item.text == item.text
        assert await data_sink.size() == 0
        assert await data_sink.is_empty()

    @pytest.mark.asyncio
    async def test_get_item_empty_sink(self, data_sink):
        """Test getting an item from an empty sink."""
        result = await data_sink.get_item()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_items(self, data_sink, sample_content_items):
        """Test getting multiple items from the sink."""
        await data_sink.add_items(sample_content_items)

        retrieved_items = await data_sink.get_items(2)

        assert len(retrieved_items) == 2
        assert retrieved_items[0].id == sample_content_items[0].id
        assert retrieved_items[1].id == sample_content_items[1].id
        assert await data_sink.size() == 1  # One item remaining

    @pytest.mark.asyncio
    async def test_get_items_more_than_available(self, data_sink, sample_content_items):
        """Test getting more items than available in the sink."""
        await data_sink.add_items(sample_content_items)

        retrieved_items = await data_sink.get_items(10)  # More than available

        assert len(retrieved_items) == len(sample_content_items)
        assert await data_sink.size() == 0
        assert await data_sink.is_empty()

    @pytest.mark.asyncio
    async def test_get_items_empty_sink(self, data_sink):
        """Test getting items from an empty sink."""
        result = await data_sink.get_items(5)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_item_by_id(self, data_sink, sample_content_items):
        """Test getting a specific item by ID."""
        await data_sink.add_items(sample_content_items)
        target_item = sample_content_items[1]

        retrieved_item = await data_sink.get_item_by_id(target_item.id)

        assert retrieved_item is not None
        assert retrieved_item.id == target_item.id
        assert await data_sink.size() == 2  # One item removed

    @pytest.mark.asyncio
    async def test_get_item_by_id_not_found(self, data_sink, sample_content_items):
        """Test getting an item by non-existent ID."""
        await data_sink.add_items(sample_content_items)
        non_existent_id = uuid4()

        result = await data_sink.get_item_by_id(non_existent_id)

        assert result is None
        assert await data_sink.size() == len(sample_content_items)  # No items removed

    @pytest.mark.asyncio
    async def test_peek_items(self, data_sink, sample_content_items):
        """Test peeking at items without removing them."""
        await data_sink.add_items(sample_content_items)

        peeked_items = await data_sink.peek_items(2)

        assert len(peeked_items) == 2
        assert peeked_items[0].id == sample_content_items[0].id
        assert peeked_items[1].id == sample_content_items[1].id
        assert await data_sink.size() == len(sample_content_items)  # No items removed

    @pytest.mark.asyncio
    async def test_peek_items_more_than_available(
        self, data_sink, sample_content_items
    ):
        """Test peeking at more items than available."""
        await data_sink.add_items(sample_content_items)

        peeked_items = await data_sink.peek_items(10)

        assert len(peeked_items) == len(sample_content_items)
        assert await data_sink.size() == len(sample_content_items)

    @pytest.mark.asyncio
    async def test_peek_items_default_count(self, data_sink, sample_content_items):
        """Test peeking with default count (1)."""
        await data_sink.add_items(sample_content_items)

        peeked_items = await data_sink.peek_items()

        assert len(peeked_items) == 1
        assert peeked_items[0].id == sample_content_items[0].id

    @pytest.mark.asyncio
    async def test_buffer_size_limit(self, data_sink, sample_content_items):
        """Test that buffer respects max_buffer_size limit."""
        # data_sink has max_buffer_size=5
        extra_items = [
            ContentItem(
                content_type=ContentType.NEWS,
                text=f"Extra content {i}",
                source_url=f"https://example.com/news/{i}",
                author=f"author_{i}",
            )
            for i in range(5)
        ]

        all_items = sample_content_items + extra_items  # 8 items total
        await data_sink.add_items(all_items)

        # Should only have 5 items (max_buffer_size)
        assert await data_sink.size() == 5

        # Should have the last 5 items (oldest removed)
        peeked_items = await data_sink.peek_items(5)
        assert len(peeked_items) == 5
        # First item should be from the extra_items (oldest 3 were removed)
        assert peeked_items[0].text == "Extra content 0"

    @pytest.mark.asyncio
    async def test_clear(self, data_sink, sample_content_items):
        """Test clearing all items from the sink."""
        await data_sink.add_items(sample_content_items)
        assert await data_sink.size() == len(sample_content_items)

        await data_sink.clear()

        assert await data_sink.size() == 0
        assert await data_sink.is_empty()

    @pytest.mark.asyncio
    async def test_size(self, data_sink, sample_content_items):
        """Test size method returns correct count."""
        assert await data_sink.size() == 0

        await data_sink.add_item(sample_content_items[0])
        assert await data_sink.size() == 1

        await data_sink.add_items(sample_content_items[1:])
        assert await data_sink.size() == len(sample_content_items)

    @pytest.mark.asyncio
    async def test_is_empty(self, data_sink, sample_content_items):
        """Test is_empty method."""
        assert await data_sink.is_empty()

        await data_sink.add_item(sample_content_items[0])
        assert not await data_sink.is_empty()

        await data_sink.get_item()
        assert await data_sink.is_empty()

    @pytest.mark.asyncio
    async def test_wait_for_item_timeout(self, data_sink):
        """Test wait_for_item with timeout when no items are added."""
        result = await data_sink.wait_for_item(timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_item_success(self, data_sink, sample_content_items):
        """Test wait_for_item when an item is added."""

        async def add_item_later():
            await asyncio.sleep(0.1)
            await data_sink.add_item(sample_content_items[0])

        # Start adding item in background
        add_task = asyncio.create_task(add_item_later())

        # Wait for item (should succeed)
        result = await data_sink.wait_for_item(timeout=1.0)

        await add_task
        assert result is True

    @pytest.mark.asyncio
    async def test_fifo_order(self, data_sink, sample_content_items):
        """Test that items are retrieved in FIFO (first-in-first-out) order."""
        await data_sink.add_items(sample_content_items)

        # Get items one by one and verify order
        for expected_item in sample_content_items:
            retrieved_item = await data_sink.get_item()
            assert retrieved_item.id == expected_item.id

    @pytest.mark.asyncio
    async def test_concurrent_access(self, data_sink, sample_content_items):
        """Test concurrent access to the sink."""

        async def add_items():
            for item in sample_content_items:
                await data_sink.add_item(item)
                await asyncio.sleep(0.01)

        async def get_items():
            retrieved = []
            for _ in range(len(sample_content_items)):
                while await data_sink.is_empty():
                    await asyncio.sleep(0.01)
                item = await data_sink.get_item()
                if item:
                    retrieved.append(item)
            return retrieved

        # Run add and get operations concurrently
        add_task = asyncio.create_task(add_items())
        get_task = asyncio.create_task(get_items())

        await add_task
        retrieved_items = await get_task

        assert len(retrieved_items) == len(sample_content_items)
        assert await data_sink.is_empty()

    @pytest.mark.asyncio
    async def test_stream_items_basic(self, data_sink, sample_content_items):
        """Test basic stream_items functionality."""
        await data_sink.add_items(sample_content_items)

        streamed_items = []
        stream = data_sink.stream_items()

        # Get the first few items from the stream
        for _ in range(len(sample_content_items)):
            item = await stream.__anext__()
            streamed_items.append(item)

        assert len(streamed_items) == len(sample_content_items)

        # Verify items are the same (might be in batches)
        streamed_ids = {item.id for item in streamed_items}
        expected_ids = {item.id for item in sample_content_items}
        assert streamed_ids == expected_ids
