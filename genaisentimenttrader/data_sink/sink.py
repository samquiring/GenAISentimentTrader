"""Data sink implementation for buffering input content."""

import asyncio
from typing import AsyncGenerator, List, Optional
from uuid import UUID

from genaisentimenttrader.models.types import ContentItem


class DataSink:
    """Centralized data sink for receiving and buffering content items."""
    
    def __init__(self, max_buffer_size: int = 1000) -> None:
        """Initialize the data sink.
        
        Args:
            max_buffer_size: Maximum number of items to buffer before dropping oldest
        """
        self._buffer: List[ContentItem] = []
        self._max_buffer_size = max_buffer_size
        self._lock = asyncio.Lock()
        self._new_item_event = asyncio.Event()
    
    async def add_item(self, item: ContentItem) -> None:
        """Add a content item to the sink.
        
        Args:
            item: Content item to add
        """
        async with self._lock:
            self._buffer.append(item)
            
            # Remove oldest items if buffer is full
            if len(self._buffer) > self._max_buffer_size:
                self._buffer.pop(0)
            
            # Notify waiters that a new item is available
            self._new_item_event.set()
            self._new_item_event.clear()
    
    async def add_items(self, items: List[ContentItem]) -> None:
        """Add multiple content items to the sink.
        
        Args:
            items: List of content items to add
        """
        for item in items:
            await self.add_item(item)
    
    async def get_item(self) -> Optional[ContentItem]:
        """Get the oldest item from the sink.
        
        Returns:
            The oldest content item, or None if sink is empty
        """
        async with self._lock:
            if self._buffer:
                return self._buffer.pop(0)
            return None
    
    async def get_items(self, count: int) -> List[ContentItem]:
        """Get multiple items from the sink.
        
        Args:
            count: Maximum number of items to retrieve
            
        Returns:
            List of content items (may be fewer than requested)
        """
        items: List[ContentItem] = []
        async with self._lock:
            for _ in range(min(count, len(self._buffer))):
                if self._buffer:
                    items.append(self._buffer.pop(0))
        return items
    
    async def get_item_by_id(self, item_id: UUID) -> Optional[ContentItem]:
        """Get a specific item by ID.
        
        Args:
            item_id: UUID of the item to retrieve
            
        Returns:
            The content item if found, None otherwise
        """
        async with self._lock:
            for i, item in enumerate(self._buffer):
                if item.id == item_id:
                    return self._buffer.pop(i)
        return None
    
    async def peek_items(self, count: int = 1) -> List[ContentItem]:
        """Peek at items without removing them.
        
        Args:
            count: Number of items to peek at
            
        Returns:
            List of content items (not removed from sink)
        """
        async with self._lock:
            return self._buffer[:count].copy()
    
    async def wait_for_item(self, timeout: Optional[float] = None) -> bool:
        """Wait for a new item to be added to the sink.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if an item was added, False if timeout occurred
        """
        try:
            await asyncio.wait_for(self._new_item_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    async def stream_items(self) -> AsyncGenerator[ContentItem, None]:
        """Stream items as they become available.
        
        Yields:
            Content items as they are added to the sink
        """
        while True:
            # Get any existing items first
            items = await self.get_items(10)  # Batch processing
            for item in items:
                yield item
            
            # Wait for new items if buffer was empty
            if not items:
                await self.wait_for_item()
    
    async def size(self) -> int:
        """Get the current size of the buffer.
        
        Returns:
            Number of items currently in the buffer
        """
        async with self._lock:
            return len(self._buffer)
    
    async def is_empty(self) -> bool:
        """Check if the sink is empty.
        
        Returns:
            True if the sink contains no items
        """
        return await self.size() == 0
    
    async def clear(self) -> None:
        """Clear all items from the sink."""
        async with self._lock:
            self._buffer.clear()