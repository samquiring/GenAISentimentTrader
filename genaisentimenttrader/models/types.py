"""Core type definitions for the sentiment trading system."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ContentType(Enum):
    """Type of content being processed."""

    TWEET = "TWEET"
    ARTICLE = "ARTICLE"
    NEWS = "NEWS"
    BLOG = "BLOG"


class AssetType(Enum):
    """Type of financial asset."""

    STOCK = "STOCK"
    CRYPTO = "CRYPTO"
    ETF = "ETF"
    COMMODITY = "COMMODITY"


class SentimentType(Enum):
    """Sentiment classification."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"


class TradingAction(Enum):
    """Trading action to take."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_ACTION = "NO_ACTION"


class ContentItem(BaseModel):
    """Input content item for processing."""

    id: UUID = Field(default_factory=uuid4)
    content_type: ContentType
    text: str = Field(..., min_length=1)
    source_url: Optional[str] = None
    author: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Union[str, int, float]] = Field(default_factory=dict)


class ExtractedEntity(BaseModel):
    """An extracted financial entity from content."""

    symbol: str = Field(..., min_length=1)
    asset_type: AssetType
    sentiment: SentimentType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1)
    context: str = Field(..., min_length=1)


class ProcessingResult(BaseModel):
    """Result of processing a content item."""

    content_id: UUID
    extracted_entities: List[ExtractedEntity] = Field(default_factory=list)
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_duration_ms: Optional[float] = None


class TradingSignal(BaseModel):
    """Trading signal generated from analysis."""

    id: UUID = Field(default_factory=uuid4)
    content_id: UUID
    symbol: str = Field(..., min_length=1)
    asset_type: AssetType
    action: TradingAction
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1)
    suggested_quantity: Optional[float] = Field(default=None, ge=0.0)
    risk_level: float = Field(default=0.5, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TradingOrder(BaseModel):
    """Actual trading order to execute."""

    id: UUID = Field(default_factory=uuid4)
    signal_id: UUID
    symbol: str = Field(..., min_length=1)
    asset_type: AssetType
    action: TradingAction
    quantity: float = Field(..., gt=0.0)
    expected_price: Optional[float] = Field(default=None, gt=0.0)
    status: str = Field(default="pending")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    actual_price: Optional[float] = None
