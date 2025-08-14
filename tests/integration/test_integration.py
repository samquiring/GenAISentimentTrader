"""Test script to validate the integration between data sink and entity extractor."""

import asyncio
import contextlib

import pytest
from loguru import logger

from genaisentimenttrader.data_sink.sink import DataSink
from genaisentimenttrader.models.types import (
    AssetType,
    ContentItem,
    ContentType,
    SentimentType,
)
from genaisentimenttrader.processors.sentiment_pipeline import (
    SentimentProcessingPipeline,
)


@pytest.fixture()
def create_sample_content() -> tuple[list[ContentItem], list[dict]]:
    """Create sample content items for testing with expected extraction results."""
    sample_data = [
        {
            "text": "Apple (AAPL) stock is soaring after their latest earnings report showed record"
                    " revenue. The iPhone sales exceeded expectations.",
            "expected": {
                "symbols": ["AAPL"],
                "asset_types": [AssetType.STOCK],
                "sentiments": [SentimentType.POSITIVE],
            },
        },
        {
            "text": "Bitcoin (BTC) crashed today amid regulatory concerns. "
                    "Many investors are selling their crypto holdings.",
            "expected": {
                "symbols": ["BTC"],
                "asset_types": [AssetType.CRYPTO],
                "sentiments": [SentimentType.NEGATIVE],
            },
        },
        {
            "text": "Tesla (TSLA) announced a new gigafactory in Texas. "
                    "Elon Musk tweeted that production will begin next year.",
            "expected": {
                "symbols": ["TSLA"],
                "asset_types": [AssetType.STOCK],
                "sentiments": [SentimentType.POSITIVE],
            },
        },
        {
            "text": "Google parent Alphabet (GOOGL) reported strong Q4 results. "
                    "Their cloud division grew 45% year-over-year.",
            "expected": {
                "symbols": ["GOOGL"],
                "asset_types": [AssetType.STOCK],
                "sentiments": [SentimentType.POSITIVE],
            },
        },
        {
            "text": "Oil prices (WTI) are rising due to geopolitical tensions in the Middle East. "
                    "Energy stocks are benefiting.",
            "expected": {
                "symbols": ["WTI"],
                "asset_types": [AssetType.COMMODITY],
                "sentiments": [SentimentType.POSITIVE],
            },
        },
        {
            "text": "Netflix (NFLX) subscriber growth disappointed investors."
                    " The stock fell 8% in after-hours trading.",
            "expected": {
                "symbols": ["NFLX"],
                "asset_types": [AssetType.STOCK],
                "sentiments": [SentimentType.NEGATIVE],
            },
        },
        {
            "text": "Ethereum (ETH) hit a new all-time high as DeFi "
                    "adoption continues to grow rapidly.",
            "expected": {
                "symbols": ["ETH"],
                "asset_types": [AssetType.CRYPTO],
                "sentiments": [SentimentType.POSITIVE],
            },
        },
        {
            "text": "Microsoft (MSFT) Azure revenue grew 50% as companies accelerate "
                    "digital transformation initiatives.",
            "expected": {
                "symbols": ["MSFT"],
                "asset_types": [AssetType.STOCK],
                "sentiments": [SentimentType.POSITIVE],
            },
        },
    ]

    content_items = []
    expected_results = []

    for i, data in enumerate(sample_data):
        item = ContentItem(
            content_type=ContentType.TWEET if i % 2 == 0 else ContentType.NEWS,
            text=data["text"],
            source_url=f"https://example.com/content/{i}",
            author=f"author_{i}",
            metadata={"test_item": i},
        )
        content_items.append(item)
        expected_results.append(data["expected"])

    return content_items, expected_results


@pytest.mark.asyncio
async def test_basic_integration(create_sample_content):
    """Test basic integration between sink and entity extractor."""
    # Create components
    sink = DataSink(max_buffer_size=100)
    pipeline = SentimentProcessingPipeline(sink)

    # Add sample content to sink
    sample_content, expected_results = create_sample_content
    await sink.add_items(sample_content)

    # Process all items
    results = await pipeline.process_all_pending()

    # Validate results match expectations
    assert len(results) == len(expected_results), (
        f"Expected {len(expected_results)} results, got {len(results)}"
    )

    correct_extractions = 0
    total_entities = 0

    for i, result in enumerate(results):
        entities = result.extracted_entities
        expected = expected_results[i]
        total_entities += len(entities)

        logger.info(f"\nResult {i + 1}: {sample_content[i].text[:50]}...")
        logger.info(
            f"  Expected: {expected['symbols']} ({[at.value for at in expected['asset_types']]}) "
            f"- {[st.value for st in expected['sentiments']]}"
        )
        logger.info(f"  Found {len(entities)} entities:")

        # Check if any extracted entities match expected ones
        expected_symbols = set(expected["symbols"])
        {entity.symbol for entity in entities}

        for entity in entities:
            logger.info(
                f"    {entity.symbol} ({entity.asset_type.value}) - "
                f"{entity.sentiment.value} (confidence: {entity.confidence:.2f})"
            )
            logger.info(f"      Context: {entity.context[:80]}...")

            # Check if this entity matches expectations
            if (
                entity.symbol in expected_symbols
                and entity.asset_type in expected["asset_types"]
                and entity.sentiment in expected["sentiments"]
            ):
                correct_extractions += 1
                logger.info("      ✓ MATCHES EXPECTED")
            else:
                logger.info("      ✗ Does not match expected")

    accuracy = (
        correct_extractions / max(len(expected_results), 1) if expected_results else 0
    )
    logger.info("\n=== VALIDATION RESULTS ===")
    logger.info(f"Total items processed: {len(results)}")
    logger.info(f"Total entities extracted: {total_entities}")
    logger.info(f"Correct extractions: {correct_extractions}/{len(expected_results)}")
    logger.info(f"Accuracy: {accuracy:.2%}")

    # Assert that we have at least some correct extractions
    assert correct_extractions > 0, "No correct extractions found!"
    logger.info(f"\n✓ Test passed with {correct_extractions} correct extractions")

    # Test ticker entities storage
    ticker_entities = pipeline.get_ticker_entities()
    logger.info("\n=== TICKER STORAGE VALIDATION ===")
    logger.info(f"Tickers stored: {list(ticker_entities.keys())}")
    logger.info(f"Total tickers: {len(ticker_entities)}")

    total_stored_entities = sum(len(entities) for entities in ticker_entities.values())
    assert total_stored_entities == total_entities, (
        f"Stored entities ({total_stored_entities}) != extracted entities ({total_entities})"
    )
    logger.info(f"✓ All {total_stored_entities} entities correctly stored by ticker")


@pytest.mark.asyncio
async def test_batch_processing(create_sample_content):
    """Test batch processing functionality."""
    logger.info("\n\n=== Testing Batch Processing ===")

    sink = DataSink()
    pipeline = SentimentProcessingPipeline(sink)

    # Add sample content
    sample_content, _ = create_sample_content
    await sink.add_items(sample_content)

    logger.info(f"Added {len(sample_content)} items to sink")

    # Process in batches
    batch_size = 3
    all_results = []

    while not await sink.is_empty():
        logger.info(f"\nProcessing batch of {batch_size} items...")
        batch_results = await pipeline.process_batch(batch_size)
        all_results.extend(batch_results)

        logger.info(f"Batch completed: {len(batch_results)} items processed")
        logger.info(f"Remaining in sink: {await sink.size()}")

    logger.info(f"\nTotal processed: {len(all_results)} items")
    logger.info(f"Final stats: {pipeline.get_stats()}")

    # Verify ticker entities are accumulated across batches
    ticker_entities = pipeline.get_ticker_entities()
    total_stored = sum(len(entities) for entities in ticker_entities.values())
    total_extracted = sum(len(result.extracted_entities) for result in all_results)

    logger.info(
        f"Ticker storage check: {total_stored} stored, {total_extracted} extracted"
    )
    assert total_stored == total_extracted, (
        "Ticker storage should accumulate all entities from batches"
    )


@pytest.mark.asyncio
async def test_continuous_processing(create_sample_content):
    """Test continuous processing with simulated real-time data."""
    logger.info("\n\n=== Testing Continuous Processing ===")

    sink = DataSink()
    pipeline = SentimentProcessingPipeline(sink)

    async def add_content_periodically():
        """Simulate adding content over time."""
        sample_content, _ = create_sample_content

        for i, item in enumerate(sample_content):
            await asyncio.sleep(0.5)  # Add item every 500ms
            await sink.add_item(item)
            logger.info(f"Added item {i + 1}: {item.text[:50]}...")

    # Start continuous processing
    processing_task = asyncio.create_task(
        pipeline.run_continuous(batch_size=2, poll_interval=0.3)
    )

    # Start adding content
    content_task = asyncio.create_task(add_content_periodically())

    # Let it run for a bit
    await asyncio.sleep(6)

    # Stop processing
    pipeline.stop()
    content_task.cancel()

    with contextlib.suppress(asyncio.CancelledError):
        await processing_task

    logger.info(f"\nFinal stats: {pipeline.get_stats()}")
    logger.info(f"Items remaining in sink: {await sink.size()}")

    # Check ticker entities were stored during continuous processing
    ticker_entities = pipeline.get_ticker_entities()
    if ticker_entities:
        logger.info(
            f"Continuous processing stored entities for {len(ticker_entities)} tickers"
        )
        logger.info(f"Tickers: {list(ticker_entities.keys())}")
    else:
        logger.info("No ticker entities stored during continuous processing")


@pytest.mark.asyncio
async def test_single_item_processing():
    """Test processing individual items."""
    logger.info("\n\n=== Testing Single Item Processing ===")

    sink = DataSink()
    pipeline = SentimentProcessingPipeline(sink)

    # Create a single test item
    test_item = ContentItem(
        content_type=ContentType.ARTICLE,
        text="Amazon (AMZN) stock surged 12% after announcing record-breaking Prime Day sales. "
             "The e-commerce giant also reported strong AWS growth.",
        source_url="https://example.com/amazon-news",
        author="financial_reporter",
    )

    logger.info(f"Processing single item: {test_item.text}")

    # Process the item
    result = await pipeline.process_single_item(test_item)

    logger.info(f"\nProcessing completed in {result.processing_duration_ms:.2f}ms")
    logger.info(f"Entities extracted: {len(result.extracted_entities)}")

    for entity in result.extracted_entities:
        logger.info(f"\n  Symbol: {entity.symbol}")
        logger.info(f"  Asset Type: {entity.asset_type.value}")
        logger.info(f"  Sentiment: {entity.sentiment.value}")
        logger.info(f"  Confidence: {entity.confidence:.3f}")
        logger.info(f"  Context: {entity.context}")
        logger.info(f"  Reasoning: {entity.reasoning}")

    # Test ticker storage for single item
    ticker_entities = pipeline.get_ticker_entities()
    logger.info(f"\nTicker storage: {len(ticker_entities)} tickers stored")

    amzn_entities = pipeline.get_entities_for_ticker("AMZN")
    logger.info(f"AMZN entities: {len(amzn_entities)}")
    assert len(amzn_entities) > 0, "Should have stored AMZN entities"
    logger.info("✓ AMZN ticker storage validated")
