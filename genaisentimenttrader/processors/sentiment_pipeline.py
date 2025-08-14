"""Sentiment processing pipeline that combines data sink and entity extraction."""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from genaisentimenttrader.data_sink.sink import DataSink
from genaisentimenttrader.models.types import (
    ContentItem,
    ExtractedEntity,
    ProcessingResult,
)
from genaisentimenttrader.processors.entity_extractor import EntityExtractor


class SentimentProcessingPipeline:
    """Pipeline that processes content from data sink through entity extraction."""

    def __init__(
        self, data_sink: DataSink, entity_extractor: Optional[EntityExtractor] = None
    ):
        """Initialize the processing pipeline.

        Args:
            data_sink: DataSink instance to pull content from
            entity_extractor: EntityExtractor instance, creates new one if None
        """
        self.data_sink = data_sink
        self.entity_extractor = entity_extractor or EntityExtractor()
        self._running = False
        self._processing_stats = {"processed": 0, "failed": 0}
        self._ticker_entities: Dict[str, List[ExtractedEntity]] = {}

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
                processing_duration_ms=processing_duration,
            )

            # Store entities by ticker
            self._store_entities_by_ticker(extracted_entities)

            self._processing_stats["processed"] += 1
            return result

        except Exception as e:
            self._processing_stats["failed"] += 1
            logger.error("Failed to process content item {}: {}", content_item.id, e)

            # Return empty result on failure
            return ProcessingResult(
                content_id=content_item.id,
                extracted_entities=[],
                processing_duration_ms=(time.time() - start_time) * 1000,
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

    async def run_continuous(self, batch_size: int = 5, poll_interval: float = 1.0) -> None:
        """Run the pipeline continuously, processing items as they arrive.

        Args:
            batch_size: Number of items to process in each batch
            poll_interval: Time to wait between polling for new items
        """
        self._running = True
        logger.info(
            "Starting sentiment processing pipeline (batch_size={})", batch_size
        )

        while self._running:
            try:
                # Check if there are items to process
                sink_size = await self.data_sink.size()

                if sink_size > 0:
                    logger.info(
                        "Processing {} items from sink...", min(batch_size, sink_size)
                    )
                    results = await self.process_batch(batch_size)

                    if results:
                        logger.info("Processed {} items successfully", len(results))
                        for result in results:
                            entities_count = len(result.extracted_entities)
                            duration = result.processing_duration_ms
                            logger.debug(
                                "Item {}: {} entities ({:.2f}ms)",
                                result.content_id,
                                entities_count,
                                duration,
                            )
                else:
                    # Wait for new items
                    await self.data_sink.wait_for_item(timeout=poll_interval)

            except Exception as e:
                logger.error("Error in processing pipeline: {}", e)
                await asyncio.sleep(poll_interval)

    def stop(self) -> None:
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

    def _store_entities_by_ticker(self, entities: List[ExtractedEntity]) -> None:
        """Store extracted entities organized by ticker symbol.

        Args:
            entities: List of extracted entities to store
        """
        for entity in entities:
            ticker = entity.symbol.upper()  # Normalize ticker to uppercase
            if ticker not in self._ticker_entities:
                self._ticker_entities[ticker] = []
            self._ticker_entities[ticker].append(entity)

    def get_ticker_entities(self) -> Dict[str, List[ExtractedEntity]]:
        """Get the flattened ticker -> entities mapping.

        Returns:
            Dictionary with ticker symbols as keys and lists of entities as values
        """
        return self._ticker_entities.copy()

    def save_ticker_entities(self, filepath: Path) -> None:
        """Save the ticker -> entities mapping to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        # Convert ExtractedEntity objects to dictionaries for JSON serialization
        serializable_data = {}
        for ticker, entities in self._ticker_entities.items():
            serializable_data[ticker] = [entity.model_dump(mode='json') for entity in entities]

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, default=str)

        logger.info("Saved ticker entities to {}", filepath)

    def load_ticker_entities(self, filepath: Path) -> None:
        """Load ticker -> entities mapping from a JSON file.

        Args:
            filepath: Path to the JSON file to load
        """
        if not filepath.exists():
            logger.warning(
                "File {} does not exist, starting with empty ticker entities",
                filepath,
            )
            return

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert dictionaries back to ExtractedEntity objects
        self._ticker_entities = {}
        for ticker, entities_data in data.items():
            self._ticker_entities[ticker] = [
                ExtractedEntity.model_validate(entity_data) for entity_data in entities_data
            ]

        logger.info("Loaded ticker entities from {}", filepath)

    def clear_ticker_entities(self) -> None:
        """Clear all stored ticker -> entities mappings."""
        self._ticker_entities.clear()
        logger.info("Cleared all ticker entities")

    def get_entities_for_ticker(self, ticker: str) -> List[ExtractedEntity]:
        """Get all entities for a specific ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of entities for the ticker, empty list if ticker not found
        """
        return self._ticker_entities.get(ticker.upper(), [])
