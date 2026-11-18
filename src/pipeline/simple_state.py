"""Evidence returned by a single cor_simple retrieval attempt."""

from dataclasses import dataclass, field


@dataclass
class SubQuestionState:
    id: int
    original_question: str
    resolved_question: str
    chosen_strategy: str
    confidence: float = 0.0
    failure_type: str = ""
    text_evidence: list[dict] = field(default_factory=list)
    image_evidence: list[dict] = field(default_factory=list)
    visual_observations: list[dict] = field(default_factory=list)
