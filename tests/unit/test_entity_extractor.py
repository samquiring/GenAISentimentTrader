"""Unit tests for EntityExtractor."""

import pytest
from unittest.mock import MagicMock, patch

from genaisentimenttrader.models.types import AssetType, ContentItem, ContentType, ExtractedEntity, SentimentType
from genaisentimenttrader.processors.entity_extractor import EntityExtractor


class TestEntityExtractor:
    """Test the EntityExtractor class."""
    
    @pytest.fixture
    def mock_dspy_settings(self):
        """Mock dspy settings configuration."""
        with patch('genaisentimenttrader.processors.entity_extractor.dspy.settings') as mock_settings:
            yield mock_settings
    
    @pytest.fixture
    def mock_dspy_predict(self):
        """Mock dspy.Predict class."""
        with patch('genaisentimenttrader.processors.entity_extractor.dspy.Predict') as mock_predict:
            yield mock_predict
    
    @pytest.fixture
    def mock_dspy_lm(self):
        """Mock dspy.LM class."""
        with patch('genaisentimenttrader.processors.entity_extractor.dspy.LM') as mock_lm:
            yield mock_lm
    
    @pytest.fixture
    def entity_extractor(self, mock_dspy_settings, mock_dspy_predict, mock_dspy_lm):
        """Create EntityExtractor instance with mocked dependencies."""
        mock_predict_instance = MagicMock()

        extractor = EntityExtractor()
        extractor.extract_module = mock_predict_instance
        return extractor
    
    @pytest.fixture
    def sample_content_item(self):
        """Create a sample ContentItem for testing."""
        return ContentItem(
            content_type=ContentType.TWEET,
            text="Apple (AAPL) stock is soaring! Great earnings report.",
            source_url="https://example.com/tweet",
            author="trader_joe"
        )
    
    def test_init(self, mock_dspy_settings, mock_dspy_predict, mock_dspy_lm):
        """Test EntityExtractor initialization."""
        # Test with default model
        EntityExtractor()
        mock_dspy_settings.configure.assert_called_once()
        mock_dspy_lm.assert_called_with(model="gemini/gemini-2.5-flash-lite")
        mock_dspy_predict.assert_called_once()
    
    def test_init_custom_model(self, mock_dspy_settings, mock_dspy_predict, mock_dspy_lm):
        """Test EntityExtractor initialization with custom model."""
        custom_model = "gpt-4"
        EntityExtractor(model=custom_model)
        mock_dspy_lm.assert_called_with(model=custom_model)
    
    @pytest.mark.asyncio
    async def test_forward_single_entity(self, entity_extractor, sample_content_item):
        """Test forward method with single entity extraction."""
        # Mock the prediction result
        mock_prediction = MagicMock()
        mock_prediction.symbols = ["AAPL"]
        mock_prediction.asset_types = ["STOCK"]
        mock_prediction.confidences = [0.95]
        mock_prediction.contexts = ["Apple (AAPL) stock is soaring"]
        mock_prediction.sentiments = ["POSITIVE"]
        mock_prediction.reasonings = ["Stock price increase mentioned"]
        
        entity_extractor.extract_module.return_value = mock_prediction
        
        # Test the forward method
        result = await entity_extractor.forward(sample_content_item)
        
        # Verify the call was made
        entity_extractor.extract_module.assert_called_once_with(
            content=sample_content_item.text
        )
        
        # Verify the result
        assert len(result) == 1
        entity = result[0]
        assert isinstance(entity, ExtractedEntity)
        assert entity.symbol == "AAPL"
        assert entity.asset_type == AssetType.STOCK
        assert entity.sentiment == SentimentType.POSITIVE
        assert entity.confidence == 0.95
        assert entity.context == "Apple (AAPL) stock is soaring"
        assert entity.reasoning == "Stock price increase mentioned"
    
    @pytest.mark.asyncio
    async def test_forward_multiple_entities(self, entity_extractor, sample_content_item):
        """Test forward method with multiple entity extraction."""
        # Mock the prediction result with multiple entities
        mock_prediction = MagicMock()
        mock_prediction.symbols = ["AAPL", "MSFT"]
        mock_prediction.asset_types = ["STOCK", "STOCK"]
        mock_prediction.confidences = [0.95, 0.88]
        mock_prediction.contexts = ["Apple (AAPL) stock", "Microsoft (MSFT) mentioned"]
        mock_prediction.sentiments = ["POSITIVE", "NEUTRAL"]
        mock_prediction.reasonings = ["Positive earnings", "Neutral mention"]
        
        entity_extractor.extract_module.return_value = mock_prediction
        
        result = await entity_extractor.forward(sample_content_item)
        
        assert len(result) == 2
        
        # Check first entity
        assert result[0].symbol == "AAPL"
        assert result[0].asset_type == AssetType.STOCK
        assert result[0].sentiment == SentimentType.POSITIVE
        assert result[0].confidence == 0.95
        
        # Check second entity
        assert result[1].symbol == "MSFT"
        assert result[1].asset_type == AssetType.STOCK
        assert result[1].sentiment == SentimentType.NEUTRAL
        assert result[1].confidence == 0.88
    
    @pytest.mark.asyncio
    async def test_forward_crypto_entity(self, entity_extractor):
        """Test forward method with crypto entity."""
        crypto_content = ContentItem(
            content_type=ContentType.NEWS,
            text="Bitcoin (BTC) crashed today due to regulatory concerns.",
            source_url="https://example.com/crypto-news",
            author="crypto_reporter"
        )
        
        mock_prediction = MagicMock()
        mock_prediction.symbols = ["BTC"]
        mock_prediction.asset_types = ["CRYPTO"]
        mock_prediction.confidences = [0.92]
        mock_prediction.contexts = ["Bitcoin (BTC) crashed today"]
        mock_prediction.sentiments = ["NEGATIVE"]
        mock_prediction.reasonings = ["Price crash mentioned"]
        
        entity_extractor.extract_module.return_value = mock_prediction
        
        result = await entity_extractor.forward(crypto_content)
        
        assert len(result) == 1
        entity = result[0]
        assert entity.symbol == "BTC"
        assert entity.asset_type == AssetType.CRYPTO
        assert entity.sentiment == SentimentType.NEGATIVE
        assert entity.confidence == 0.92
    
    @pytest.mark.asyncio
    async def test_forward_no_entities(self, entity_extractor, sample_content_item):
        """Test forward method when no entities are found."""
        mock_prediction = MagicMock()
        mock_prediction.symbols = []
        mock_prediction.asset_types = []
        mock_prediction.confidences = []
        mock_prediction.contexts = []
        mock_prediction.sentiments = []
        mock_prediction.reasonings = []
        
        entity_extractor.extract_module.return_value = mock_prediction
        
        result = await entity_extractor.forward(sample_content_item)
        
        assert len(result) == 0
        assert result == []
    
    @pytest.mark.asyncio
    async def test_forward_symbol_normalization(self, entity_extractor, sample_content_item):
        """Test that symbols are normalized to uppercase and stripped."""
        mock_prediction = MagicMock()
        mock_prediction.symbols = [" aapl ", "msft"]
        mock_prediction.asset_types = ["STOCK", "STOCK"]
        mock_prediction.confidences = [0.95, 0.88]
        mock_prediction.contexts = ["Apple mentioned", "Microsoft mentioned"]
        mock_prediction.sentiments = ["POSITIVE", "NEUTRAL"]
        mock_prediction.reasonings = ["Positive news", "Neutral mention"]
        
        entity_extractor.extract_module.return_value = mock_prediction
        
        result = await entity_extractor.forward(sample_content_item)
        
        assert len(result) == 2
        assert result[0].symbol == "AAPL"  # Uppercase and stripped
        assert result[1].symbol == "MSFT"  # Already uppercase
    
    @pytest.mark.asyncio
    async def test_forward_context_stripping(self, entity_extractor, sample_content_item):
        """Test that context strings are stripped of whitespace."""
        mock_prediction = MagicMock()
        mock_prediction.symbols = ["AAPL"]
        mock_prediction.asset_types = ["STOCK"]
        mock_prediction.confidences = [0.95]
        mock_prediction.contexts = ["  Apple stock is great  "]
        mock_prediction.sentiments = ["POSITIVE"]
        mock_prediction.reasonings = ["Positive sentiment"]
        
        entity_extractor.extract_module.return_value = mock_prediction
        
        result = await entity_extractor.forward(sample_content_item)
        
        assert len(result) == 1
        assert result[0].context == "Apple stock is great"  # Whitespace stripped
    
    @pytest.mark.asyncio
    async def test_forward_all_asset_types(self, entity_extractor, sample_content_item):
        """Test forward method with all supported asset types."""
        mock_prediction = MagicMock()
        mock_prediction.symbols = ["AAPL", "BTC", "SPY", "GLD"]
        mock_prediction.asset_types = ["STOCK", "CRYPTO", "ETF", "COMMODITY"]
        mock_prediction.confidences = [0.95, 0.92, 0.88, 0.85]
        mock_prediction.contexts = ["Apple stock", "Bitcoin crypto", "SPY ETF", "Gold commodity"]
        mock_prediction.sentiments = ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"]
        mock_prediction.reasonings = ["Good earnings", "Regulatory concerns", "Market neutral", "Mixed signals"]
        
        entity_extractor.extract_module.return_value = mock_prediction
        
        result = await entity_extractor.forward(sample_content_item)
        
        assert len(result) == 4
        
        # Verify all asset types are correctly mapped
        assert result[0].asset_type == AssetType.STOCK
        assert result[1].asset_type == AssetType.CRYPTO
        assert result[2].asset_type == AssetType.ETF
        assert result[3].asset_type == AssetType.COMMODITY
        
        # Verify all sentiment types are correctly mapped
        assert result[0].sentiment == SentimentType.POSITIVE
        assert result[1].sentiment == SentimentType.NEGATIVE
        assert result[2].sentiment == SentimentType.NEUTRAL
        assert result[3].sentiment == SentimentType.MIXED
