"""Entity extraction processor using dspy."""

from typing import List, Literal

import dspy  # type: ignore[import-untyped]

from genaisentimenttrader.models.types import (
    AssetType,
    ContentItem,
    ExtractedEntity,
    SentimentType,
)


class EntityExtractionSignature(dspy.Signature):
    """Extract 0-N financial entities (stocks, crypto) from text content."""

    content: str = dspy.InputField(
        desc="Text content to analyze for financial entities"
    )
    symbols: list[str] = dspy.OutputField(
        desc="List of financial symbols found (e.g., ['AAPL', 'BTC'])"
    )
    asset_types: list[Literal["STOCK", "CRYPTO", "ETF", "COMMODITY"]] = (
        dspy.OutputField(desc="List of asset types corresponding to symbols")
    )
    sentiments: list[Literal["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"]] = (
        dspy.OutputField(desc="List of sentiment for each entity")
    )
    confidences: list[float] = dspy.OutputField(
        desc="List of confidence scores from 0.0 to 1.0"
    )
    reasonings: list[str] = dspy.OutputField(desc="List of reasoning for sentiments")
    contexts: list[str] = dspy.OutputField(
        desc="List of relevant context around each entity mention"
    )


class EntityExtractor:
    """Extract financial entities from content using dspy."""

    def __init__(self, model: str = "gemini/gemini-2.5-flash-lite"):
        super().__init__()
        dspy.settings.configure(lm=dspy.LM(model=model))
        self.extract_module = dspy.Predict(EntityExtractionSignature)

    async def forward(self, content_item: ContentItem) -> List[ExtractedEntity]:
        """Extract financial entities from a content item.

        Args:
            content_item: The content item to analyze

        Returns:
            List of extracted entities with confidence scores
        """
        entities: List[ExtractedEntity] = []

        # Use dspy to extract entities
        pred = self.extract_module(content=content_item.text)

        # Create ExtractedEntity objects
        for (
                symbol,
                asset_type_str,
                confidence,
                context,
                sentiment_str,
                reasoning,
        ) in zip(
            pred.symbols,
            pred.asset_types,
            pred.confidences,
            pred.contexts,
            pred.sentiments,
            pred.reasonings,
            strict=False,
        ):
            entities.append(
                ExtractedEntity(
                    symbol=symbol.upper().strip(),
                    asset_type=AssetType[asset_type_str],
                    confidence=confidence,
                    context=context.strip(),
                    sentiment=SentimentType[sentiment_str],
                    reasoning=reasoning,
                )
            )

        return entities
