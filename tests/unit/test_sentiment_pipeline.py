"""Unit tests for SentimentProcessingPipeline."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genaisentimenttrader.data_sink.sink import DataSink
from genaisentimenttrader.models.types import (
    AssetType,
    ContentItem,
    ContentType,
    ExtractedEntity,
    ProcessingResult,
    SentimentType,
)
from genaisentimenttrader.processors.entity_extractor import EntityExtractor
from genaisentimenttrader.processors.sentiment_pipeline import (
    SentimentProcessingPipeline,
)


class TestSentimentProcessingPipeline:
    """Test the SentimentProcessingPipeline class."""

    @pytest.fixture
    def mock_data_sink(self):
        """Create a mock DataSink for testing."""
        mock_sink = AsyncMock(spec=DataSink)
        mock_sink.size.return_value = 0
        mock_sink.is_empty.return_value = True
        mock_sink.get_items.return_value = []
        mock_sink.wait_for_item.return_value = True
        return mock_sink

    @pytest.fixture
    def mock_entity_extractor(self):
        """Create a mock EntityExtractor for testing."""
        mock_extractor = AsyncMock(spec=EntityExtractor)
        return mock_extractor

    @pytest.fixture
    def sample_content_items(self):
        """Create sample ContentItem objects for testing."""
        return [
            ContentItem(
                content_type=ContentType.TWEET,
                text="Apple (AAPL) stock is soaring! Great earnings.",
                source_url="https://example.com/tweet/1",
                author="trader_joe",
            ),
            ContentItem(
                content_type=ContentType.NEWS,
                text="Bitcoin (BTC) crashed due to regulatory concerns.",
                source_url="https://example.com/news/1",
                author="crypto_reporter",
            ),
        ]

    @pytest.fixture
    def sample_extracted_entities(self):
        """Create sample ExtractedEntity objects for testing."""
        return [
            ExtractedEntity(
                symbol="AAPL",
                asset_type=AssetType.STOCK,
                sentiment=SentimentType.POSITIVE,
                confidence=0.95,
                reasoning="Positive earnings report",
                context="Apple stock is soaring",
            ),
            ExtractedEntity(
                symbol="BTC",
                asset_type=AssetType.CRYPTO,
                sentiment=SentimentType.NEGATIVE,
                confidence=0.90,
                reasoning="Regulatory concerns",
                context="Bitcoin crashed due to regulations",
            ),
        ]

    @pytest.fixture
    def pipeline(self, mock_data_sink, mock_entity_extractor):
        """Create a SentimentProcessingPipeline with mocked dependencies."""
        return SentimentProcessingPipeline(mock_data_sink, mock_entity_extractor)

    def test_init_with_extractor(self, mock_data_sink, mock_entity_extractor):
        """Test pipeline initialization with provided entity extractor."""
        pipeline = SentimentProcessingPipeline(mock_data_sink, mock_entity_extractor)

        assert pipeline.data_sink == mock_data_sink
        assert pipeline.entity_extractor == mock_entity_extractor
        assert not pipeline._running
        assert pipeline._processing_stats == {"processed": 0, "failed": 0}

    @patch("genaisentimenttrader.processors.sentiment_pipeline.EntityExtractor")
    def test_init_without_extractor(self, mock_extractor_class, mock_data_sink):
        """Test pipeline initialization without provided entity extractor."""
        mock_extractor_instance = MagicMock()
        mock_extractor_class.return_value = mock_extractor_instance

        pipeline = SentimentProcessingPipeline(mock_data_sink)

        assert pipeline.data_sink == mock_data_sink
        assert pipeline.entity_extractor == mock_extractor_instance
        mock_extractor_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_single_item_success(
        self, pipeline, sample_content_items, sample_extracted_entities
    ):
        """Test successful processing of a single item."""
        content_item = sample_content_items[0]
        expected_entities = [sample_extracted_entities[0]]

        pipeline.entity_extractor.forward.return_value = expected_entities

        result = await pipeline.process_single_item(content_item)

        assert isinstance(result, ProcessingResult)
        assert result.content_id == content_item.id
        assert result.extracted_entities == expected_entities
        assert result.processing_duration_ms > 0
        assert pipeline._processing_stats["processed"] == 1
        assert pipeline._processing_stats["failed"] == 0

        pipeline.entity_extractor.forward.assert_called_once_with(content_item)

        # Test ticker storage
        ticker_entities = pipeline.get_ticker_entities()
        assert "AAPL" in ticker_entities
        assert len(ticker_entities["AAPL"]) == 1
        assert ticker_entities["AAPL"][0] == expected_entities[0]

    @pytest.mark.asyncio
    async def test_process_single_item_failure(self, pipeline, sample_content_items):
        """Test handling of processing failure for a single item."""
        content_item = sample_content_items[0]

        # Mock the entity extractor to raise an exception
        pipeline.entity_extractor.forward.side_effect = Exception("Processing error")

        result = await pipeline.process_single_item(content_item)

        assert isinstance(result, ProcessingResult)
        assert result.content_id == content_item.id
        assert result.extracted_entities == []
        assert result.processing_duration_ms > 0
        assert pipeline._processing_stats["processed"] == 0
        assert pipeline._processing_stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_process_batch_success(
        self, pipeline, sample_content_items, sample_extracted_entities
    ):
        """Test successful batch processing."""
        pipeline.data_sink.get_items.return_value = sample_content_items

        # Mock entity extractor to return different entities for each item
        pipeline.entity_extractor.forward.side_effect = [
            [sample_extracted_entities[0]],
            [sample_extracted_entities[1]],
        ]

        results = await pipeline.process_batch(batch_size=2)

        assert len(results) == 2
        assert all(isinstance(result, ProcessingResult) for result in results)
        assert results[0].content_id == sample_content_items[0].id
        assert results[1].content_id == sample_content_items[1].id
        assert pipeline._processing_stats["processed"] == 2

        pipeline.data_sink.get_items.assert_called_once_with(2)
        assert pipeline.entity_extractor.forward.call_count == 2

        # Test that both tickers are stored
        ticker_entities = pipeline.get_ticker_entities()
        assert "AAPL" in ticker_entities
        assert "BTC" in ticker_entities
        assert len(ticker_entities["AAPL"]) == 1
        assert len(ticker_entities["BTC"]) == 1

    @pytest.mark.asyncio
    async def test_process_batch_empty_sink(self, pipeline):
        """Test batch processing when sink is empty."""
        pipeline.data_sink.get_items.return_value = []

        results = await pipeline.process_batch(batch_size=5)

        assert results == []
        pipeline.data_sink.get_items.assert_called_once_with(5)
        pipeline.entity_extractor.forward.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_batch_with_exceptions(
        self, pipeline, sample_content_items, sample_extracted_entities
    ):
        """Test batch processing when some items fail."""
        pipeline.data_sink.get_items.return_value = sample_content_items

        # First item succeeds, second fails
        pipeline.entity_extractor.forward.side_effect = [
            [sample_extracted_entities[0]],
            Exception("Processing error"),
        ]

        with patch(
            "genaisentimenttrader.processors.sentiment_pipeline.logger"
        ) as mock_logger:
            results = await pipeline.process_batch(batch_size=2)

        assert len(results) == 2
        assert results[0].content_id == sample_content_items[0].id
        assert pipeline._processing_stats["processed"] == 1
        assert pipeline._processing_stats["failed"] == 1

        # Error should be logged
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_process_all_pending_single_batch(
        self, pipeline, sample_content_items, sample_extracted_entities
    ):
        """Test processing all pending items in a single batch."""
        # Mock sink to return items once, then be empty
        pipeline.data_sink.is_empty.side_effect = [False, True]
        pipeline.data_sink.get_items.return_value = sample_content_items

        pipeline.entity_extractor.forward.side_effect = [
            [sample_extracted_entities[0]],
            [sample_extracted_entities[1]],
        ]

        results = await pipeline.process_all_pending()

        assert len(results) == 2
        pipeline.data_sink.is_empty.assert_called()
        pipeline.data_sink.get_items.assert_called_once_with(
            10
        )  # Default large batch size

    @pytest.mark.asyncio
    async def test_process_all_pending_multiple_batches(
        self, pipeline, sample_content_items
    ):
        """Test processing all pending items across multiple batches."""
        # Mock sink to return items twice, then be empty
        pipeline.data_sink.is_empty.side_effect = [False, False, True]
        pipeline.data_sink.get_items.side_effect = [
            [sample_content_items[0]],
            [sample_content_items[1]],
            [],
        ]

        pipeline.entity_extractor.forward.return_value = []

        results = await pipeline.process_all_pending()

        assert len(results) == 2
        assert pipeline.data_sink.get_items.call_count == 2

    @pytest.mark.asyncio
    async def test_run_continuous_with_items(self, pipeline, sample_content_items):
        """Test continuous processing when items are available."""
        # Mock sink to have items initially, then be empty
        pipeline.data_sink.size.side_effect = [2, 0, 0]  # Has items, then empty
        pipeline.data_sink.get_items.return_value = sample_content_items
        pipeline.entity_extractor.forward.return_value = []

        # Create a task that will stop the pipeline after a short delay
        async def stop_pipeline():
            await asyncio.sleep(0.1)
            pipeline.stop()

        stop_task = asyncio.create_task(stop_pipeline())

        await pipeline.run_continuous(batch_size=2, poll_interval=0.05)

        await stop_task

        # Verify processing occurred
        pipeline.data_sink.size.assert_called()
        pipeline.data_sink.get_items.assert_called()
        assert not pipeline._running

    @pytest.mark.asyncio
    async def test_run_continuous_exception_handling(self, pipeline):
        """Test that continuous processing handles exceptions gracefully."""
        pipeline.data_sink.size.side_effect = Exception("Sink error")

        async def stop_pipeline():
            await asyncio.sleep(0.05)
            pipeline.stop()

        stop_task = asyncio.create_task(stop_pipeline())

        with patch(
            "genaisentimenttrader.processors.sentiment_pipeline.logger"
        ) as mock_logger:
            try:
                await asyncio.wait_for(
                    pipeline.run_continuous(poll_interval=0.02), timeout=1.0
                )
            except asyncio.TimeoutError:
                pipeline.stop()

        await stop_task

        # Verify error was logged
        mock_logger.error.assert_called()
        # Check that at least one error call contains our expected message
        error_calls = [str(call) for call in mock_logger.error.call_args_list]
        assert any(
            "Error in processing pipeline" in call or "Sink error" in call
            for call in error_calls
        )

    def test_stop(self, pipeline):
        """Test stopping the pipeline."""
        pipeline._running = True

        with patch(
            "genaisentimenttrader.processors.sentiment_pipeline.logger"
        ) as mock_logger:
            pipeline.stop()

        assert not pipeline._running
        mock_logger.info.assert_called_once()
        assert "Stopping sentiment processing pipeline" in str(
            mock_logger.info.call_args
        )

    def test_get_stats(self, pipeline):
        """Test getting processing statistics."""
        # Modify stats
        pipeline._processing_stats["processed"] = 10
        pipeline._processing_stats["failed"] = 2

        stats = pipeline.get_stats()

        assert stats == {"processed": 10, "failed": 2}
        assert stats is not pipeline._processing_stats  # Should be a copy

    def test_get_stats_initial(self, pipeline):
        """Test getting initial statistics."""
        stats = pipeline.get_stats()
        assert stats == {"processed": 0, "failed": 0}

    def test_store_entities_by_ticker(self, pipeline, sample_extracted_entities):
        """Test storing entities by ticker symbol."""
        # Initially empty
        assert len(pipeline.get_ticker_entities()) == 0

        # Store entities
        pipeline._store_entities_by_ticker(sample_extracted_entities)

        ticker_entities = pipeline.get_ticker_entities()
        assert len(ticker_entities) == 2
        assert "AAPL" in ticker_entities
        assert "BTC" in ticker_entities
        assert len(ticker_entities["AAPL"]) == 1
        assert len(ticker_entities["BTC"]) == 1

        # Test case normalization
        lower_case_entity = ExtractedEntity(
            symbol="aapl",
            asset_type=AssetType.STOCK,
            sentiment=SentimentType.NEUTRAL,
            confidence=0.80,
            reasoning="Test entity",
            context="Test context",
        )
        pipeline._store_entities_by_ticker([lower_case_entity])

        # Should be stored under uppercase ticker
        assert len(ticker_entities["AAPL"]) == 2  # Original count, dict is a copy
        updated_entities = pipeline.get_ticker_entities()
        assert len(updated_entities["AAPL"]) == 2  # Now has both entities

    def test_get_entities_for_ticker(self, pipeline, sample_extracted_entities):
        """Test retrieving entities for a specific ticker."""
        # Store test entities
        pipeline._store_entities_by_ticker(sample_extracted_entities)

        # Test uppercase lookup
        aapl_entities = pipeline.get_entities_for_ticker("AAPL")
        assert len(aapl_entities) == 1
        assert aapl_entities[0].symbol == "AAPL"

        # Test lowercase lookup (should work due to normalization)
        aapl_entities_lower = pipeline.get_entities_for_ticker("aapl")
        assert len(aapl_entities_lower) == 1
        assert aapl_entities_lower == aapl_entities

        # Test non-existent ticker
        non_existent = pipeline.get_entities_for_ticker("NONEXISTENT")
        assert len(non_existent) == 0

    def test_clear_ticker_entities(self, pipeline, sample_extracted_entities):
        """Test clearing ticker entities."""
        # Store test entities
        pipeline._store_entities_by_ticker(sample_extracted_entities)
        assert len(pipeline.get_ticker_entities()) == 2

        # Clear and verify
        pipeline.clear_ticker_entities()
        assert len(pipeline.get_ticker_entities()) == 0
        assert len(pipeline.get_entities_for_ticker("AAPL")) == 0

    def test_save_and_load_ticker_entities(self, pipeline, sample_extracted_entities, tmp_path):
        """Test saving and loading ticker entities."""
        import json

        # Store test entities
        pipeline._store_entities_by_ticker(sample_extracted_entities)
        original_entities = pipeline.get_ticker_entities()
        tmp_file = tmp_path / "tmp_file.json"
        # Save entities
        pipeline.save_ticker_entities(tmp_file)

        # Verify file was created and contains expected data
        assert os.path.exists(tmp_file)
        with open(tmp_file, "r") as f:
            saved_data = json.load(f)
        assert "AAPL" in saved_data
        assert "BTC" in saved_data

        # Create new pipeline and load
        new_pipeline = SentimentProcessingPipeline(pipeline.data_sink)
        assert len(new_pipeline.get_ticker_entities()) == 0

        new_pipeline.load_ticker_entities(tmp_file)
        loaded_entities = new_pipeline.get_ticker_entities()

        # Verify loaded data matches original
        assert len(loaded_entities) == len(original_entities)
        assert "AAPL" in loaded_entities
        assert "BTC" in loaded_entities

    def test_load_ticker_entities_nonexistent_file(self, pipeline):
        """Test loading from non-existent file."""
        from pathlib import Path
        # Should not raise exception, just log warning
        pipeline.load_ticker_entities(Path("/nonexistent/path/file.json"))
        assert len(pipeline.get_ticker_entities()) == 0
