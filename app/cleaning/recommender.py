from .models import CleaningRecommendation
from .classifier import classify_anomaly
from .confidence import calculate_cleaning_confidence


def recommend_cleaning(
    anomaly: dict,
) -> CleaningRecommendation:

    category = classify_anomaly(
        anomaly
    )

    confidence = calculate_cleaning_confidence(
        anomaly
    )

    if category == "invalid":

        action = "replace_with_missing"

        reason = (
            "Value violates a semantic "
            "or domain rule."
        )

        requires_review = False

    elif category == "outlier":

        action = "review"

        reason = (
            "Value is statistically unusual "
            "but may still be valid."
        )

        requires_review = True

    else:

        action = "review"

        reason = (
            "Unable to determine a safe "
            "automatic cleaning strategy."
        )

        requires_review = True

    return CleaningRecommendation(

        row_index=anomaly["row_index"],

        column=anomaly["column"],

        value=anomaly["value"],

        anomaly_types=anomaly["anomaly_types"],

        action=action,

        confidence=confidence,

        reason=reason,

        requires_review=requires_review,
    )