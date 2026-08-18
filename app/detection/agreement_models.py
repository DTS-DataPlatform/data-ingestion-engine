from dataclasses import dataclass, field


@dataclass
class AnomalyAgreement:

    row_index: int

    column: str

    value: object

    detectors: list[str] = field(
        default_factory=list
    )

    agreement_count: int = 0

    agreement_ratio: float = 0.0

    confidence: float = 0.0

    severity: str = "low"