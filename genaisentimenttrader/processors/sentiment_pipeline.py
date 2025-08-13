"""Sentiment processing pipeline that combines data sink and entity extraction."""

import asyncio
import time
from typing import List, Optional

from loguru import logger

from genaisentimenttrader.data_sink.sink import DataSink
from genaisentimenttrader.models.types import ContentItem, ProcessingResult
from genaisentimenttrader.processors.entity_extractor import EntityExtractor


class SentimentProcessingPipeline:
    """Pipeline that processes content from data sink through entity extraction."""
    
    def __init__(self, data_sink: DataSink, entity_extractor: Optional[EntityExtractor] = None):
        """Initialize the processing pipeline.
        
        Args:
            data_sink: DataSink instance to pull content from
            entity_extractor: EntityExtractor instance, creates new one if None
        """
        self.data_sink = data_sink
        self.entity_extractor = entity_extractor or EntityExtractor()
        self._running = False
        self._processing_stats = {"processed": 0, "failed": 0}
    
    async def process_single_item(self, content_item: ContentItem) -> ProcessingResult:
        """Process a single content item through entity extraction.
        
        Args:
            content_item: Content item to process
            
        Returns:
            Processing result with extracted entities
        """
        start_time = time.time()
        
        try:
            extracted_entities = await self.entity_extractor.forward(content_item)
            processing_duration = (time.time() - start_time) * 1000
            
            result = ProcessingResult(
                content_id=content_item.id,
                extracted_entities=extracted_entities,
                processing_duration_ms=processing_duration
            )
            
            self._processing_stats["processed"] += 1
            return result
            
        except Exception as e:
            self._processing_stats["failed"] += 1
            logger.error("Failed to process content item {}: {}", content_item.id, e)
            
            # Return empty result on failure
            return ProcessingResult(
                content_id=content_item.id,
                extracted_entities=[],
                processing_duration_ms=(time.time() - start_time) * 1000
            )
    
    async def process_batch(self, batch_size: int = 5) -> List[ProcessingResult]:
        """Process a batch of items from the sink.
        
        Args:
            batch_size: Number of items to process in parallel
            
        Returns:
            List of processing results
        """
        content_items = await self.data_sink.get_items(batch_size)
        
        if not content_items:
            return []
        
        # Process items in parallel
        tasks = [self.process_single_item(item) for item in content_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, ProcessingResult):
                valid_results.append(result)
            else:
                logger.error("Processing task failed: {}", result)
                
        return valid_results
    
    async def run_continuous(self, batch_size: int = 5, poll_interval: float = 1.0):
        """Run the pipeline continuously, processing items as they arrive.
        
        Args:
            batch_size: Number of items to process in each batch
            poll_interval: Time to wait between polling for new items
        """
        self._running = True
        logger.info("Starting sentiment processing pipeline (batch_size={})", batch_size)
        
        while self._running:
            try:
                # Check if there are items to process
                sink_size = await self.data_sink.size()
                
                if sink_size > 0:
                    logger.info("Processing {} items from sink...", min(batch_size, sink_size))
                    results = await self.process_batch(batch_size)
                    
                    if results:
                        logger.info("Processed {} items successfully", len(results))
                        for result in results:
                            entities_count = len(result.extracted_entities)
                            duration = result.processing_duration_ms
                            logger.debug("Item {}: {} entities ({:.2f}ms)", result.content_id, entities_count, duration)
                else:
                    # Wait for new items
                    await self.data_sink.wait_for_item(timeout=poll_interval)
                    
            except Exception as e:
                logger.error("Error in processing pipeline: {}", e)
                await asyncio.sleep(poll_interval)
    
    def stop(self):
        """Stop the continuous processing pipeline."""
        self._running = False
        logger.info("Stopping sentiment processing pipeline...")
    
    def get_stats(self) -> dict:
        """Get processing statistics.
        
        Returns:
            Dictionary with processing stats
        """
        return self._processing_stats.copy()
    
    async def process_all_pending(self) -> List[ProcessingResult]:
        """Process all items currently in the sink.
        
        Returns:
            List of all processing results
        """
        all_results = []
        
        while not await self.data_sink.is_empty():
            batch_results = await self.process_batch(10)  # Larger batch for cleanup
            all_results.extend(batch_results)
            
        return all_results