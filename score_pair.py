"""Customer-record pair scoring for the Greyquill assignment.

Uses only the Python standard library.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
import math
import re
import unicodedata


@dataclass
class FieldComparison:
    """How to compare one field."""

    attribute: str
    method: str
    weight: float


@dataclass
class MatchResult:
    confidence: float
    algorithm: str
    field_scores: dict[str, float]


PLAN = [
    FieldComparison("name", "phonetic", 0.25),
    FieldComparison("email", "edit_distance", 0.20),
    FieldComparison("phone", "normalized_phone", 0.20),
    FieldComparison("date_of_birth", "date", 0.20),
    FieldComparison("pin_code", "exact", 0.10),
    FieldComparison("city", "exact", 0.05),
]


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return not text or text.lower() in {"nan", "none", "null", "nat"}


def _clean_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).strip().lower()


def _tokens(value: object) -> list[str]:
    # Commas and other punctuation are separators, so "Patel, Fatima"
    # can be compared without depending on the order of the name parts.
    return re.findall(r"[a-z0-9]+", _clean_text(value))


def _soundex(token: str) -> str:
    token = re.sub(r"[^a-z]", "", token.lower())
    if not token:
        return ""
    groups = {
        **{c: "1" for c in "bfpv"},
        **{c: "2" for c in "cgjkqsxz"},
        **{c: "3" for c in "dt"},
        "l": "4",
        **{c: "5" for c in "mn"},
        "r": "6",
    }
    result = token[0].upper()
    previous = groups.get(token[0], "")
    for char in token[1:]:
        digit = groups.get(char, "")
        if digit and digit != previous:
            result += digit
        previous = digit
    return (result + "000")[:4]


def _token_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if len(left) == 1 or len(right) == 1:
        short, long = (left, right) if len(left) < len(right) else (right, left)
        return 0.90 if long.startswith(short) else 0.0
    edit_ratio = SequenceMatcher(None, left, right).ratio()
    phonetic_ratio = 0.90 if _soundex(left) == _soundex(right) else 0.0
    return max(edit_ratio, phonetic_ratio)


def _phonetic_name_score(left: object, right: object) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0

    # Match each token to its best counterpart in the other name. Averaging
    # both directions handles reordered names and extra/missing name parts.
    left_to_right = sum(
        max(_token_similarity(token, other) for other in right_tokens)
        for token in left_tokens
    ) / len(left_tokens)
    right_to_left = sum(
        max(_token_similarity(token, other) for other in left_tokens)
        for token in right_tokens
    ) / len(right_tokens)
    return (left_to_right + right_to_left) / 2.0


def _edit_distance(left: str, right: str) -> int:
    """Levenshtein distance, using O(min(len(left), len(right))) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_left != char_right),
            ))
        previous = current
    return previous[-1]


def _edit_similarity(left: object, right: object) -> float:
    a = _clean_text(left)
    b = _clean_text(right)
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return 1.0 - (_edit_distance(a, b) / longest)


def _normalize_phone(value: object) -> str:
    digits = re.sub(r"\D", "", str(value))
    # Remove India's country code only when it is a 12-digit number.
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    # Remove one domestic trunk zero from an 11-digit number.
    if digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    return digits


def _parse_date(value: object) -> date | None:
    text = str(value).strip()
    formats = (
        "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%y", "%d-%b-%Y",
        "%d-%b-%y", "%d/%m/%y", "%d-%m-%Y", "%Y/%m/%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _compare(left: object, right: object, method: str) -> float:
    if method == "phonetic":
        return _phonetic_name_score(left, right)
    if method == "edit_distance":
        return _edit_similarity(left, right)
    if method == "normalized_phone":
        a, b = _normalize_phone(left), _normalize_phone(right)
        return float(bool(a) and a == b)
    if method == "date":
        a, b = _parse_date(left), _parse_date(right)
        return float(a is not None and a == b)
    if method == "exact":
        return float(_clean_text(left) == _clean_text(right))
    raise ValueError(f"Unsupported comparison method: {method}")


def score_pair(
    a: dict[str, object],
    b: dict[str, object],
    plan: list[FieldComparison],
) -> MatchResult:
    """Return a weighted similarity score for two customer records."""
    field_scores: dict[str, float] = {}
    weighted_total = 0.0
    compared_weight = 0.0
    methods: list[str] = []

    for comparison in plan:
        left = a.get(comparison.attribute)
        right = b.get(comparison.attribute)

        # Missing data is not evidence for or against a match.
        if _is_blank(left) or _is_blank(right):
            continue
        if comparison.weight < 0:
            raise ValueError("Field weights must be non-negative")

        score = _compare(left, right, comparison.method)
        field_scores[comparison.attribute] = score
        weighted_total += score * comparison.weight
        compared_weight += comparison.weight
        if comparison.method not in methods:
            methods.append(comparison.method)

    confidence = weighted_total / compared_weight if compared_weight else 0.0
    return MatchResult(
        confidence=max(0.0, min(1.0, confidence)),
        algorithm="+".join(methods),
        field_scores=field_scores,
    )


def main() -> None:
    """Small CLI so the README can show one reproducible run command."""
    import argparse
    import csv
    import json

    parser = argparse.ArgumentParser(description="Compare two customer records.")
    parser.add_argument("--csv", default="customer-records.csv", help="Path to the records CSV")
    parser.add_argument("--id1", required=True, help="First record_id")
    parser.add_argument("--id2", required=True, help="Second record_id")
    args = parser.parse_args()

    with open(args.csv, newline="", encoding="utf-8-sig") as file:
        records = list(csv.DictReader(file))

    by_id = {record.get("record_id"): record for record in records}
    if args.id1 not in by_id or args.id2 not in by_id:
        parser.error("One or both record IDs were not found in the CSV.")

    result = score_pair(by_id[args.id1], by_id[args.id2], PLAN)
    print(json.dumps({
        "record_ids": [args.id1, args.id2],
        "confidence": round(result.confidence, 4),
        "algorithm": result.algorithm,
        "field_scores": result.field_scores,
    }, indent=2))


if __name__ == "__main__":
    main()
