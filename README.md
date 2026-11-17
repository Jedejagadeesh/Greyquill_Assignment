  # Greyquill FDE Internship Task — Customer Record Pair Scoring

## Overview

`score_pair(a, b, plan)` compares two customer records and returns a `MatchResult`
with a confidence score from 0.0 to 1.0, the comparison methods used, and the
individual field scores.

The implementation uses only the Python standard library.

## Comparison approach

- **Phonetic name:** Tokenizes names so order and punctuation do not matter. It
  compares each token using character similarity, a Soundex-style phonetic key,
  and initial-to-full-name matching. Scores are averaged in both directions.
- **Email edit distance:** Uses normalized Levenshtein similarity:
  `1 - distance / length_of_longer_value`.
- **Normalized phone:** Removes non-digits, a leading `+91` country code, and a
  leading domestic zero according to the task specification; then compares digits.
- **Date:** Parses the date formats used in the supplied records and compares
  calendar dates.
- **Exact:** Trims surrounding whitespace and ignores case.

If either side of a field is blank or missing, that field is omitted from both
`field_scores` and the weighted average. The weighted average is recalculated
using only the weights of fields that were compared. If no fields can be
compared, confidence is `0.0`.

## Run

Keep the supplied `customer-records.csv` in the same folder as `score_pair.py`,
then run this one command:

```bash
python score_pair.py --csv customer-records.csv --id1 R0047 --id2 R0076
```

You can replace `R0047` and `R0076` with any two `record_id` values in the CSV.

## Example pair selected for the task

- `R0047` — Harpreet Bhat
- `R0076` — Haprit Bhat

The names and email local parts are similar enough to make the records look
related at first glance. However, their phone numbers, dates of birth, PIN
codes, and cities differ, which is strong evidence that they represent
different people rather than duplicate records.

A similarity score is a ranking signal, not proof of identity. Conflicting
high-value fields should be reviewed rather than treating the score as a
definitive identity decision.

## Dependencies

No third-party packages are required. Python 3.10+ is recommended.
