# Rules: company financials from SEC filings (EDGAR XBRL style)

Evidence: one line per reported value: form, fiscal period, the date range the value covers, and the filing time. A filing can report several ranges (three-month, year-to-date, prior-year comparatives). Amended filings (10-K/A, 10-Q/A) restate earlier values.

1. Latest value for a date range = the value in the most recent filing (at the as-of time) that reports exactly that range.
2. Originally reported value = the value in the earliest filing that reported that range.
3. Three-month vs year-to-date: a 10-Q reports a three-month figure and a year-to-date figure. Do not mix them.
4. Q4 is never filed directly: Q4 = fiscal-year total minus nine-month year-to-date, each using its latest value.
5. Most recently reported quarter = the latest quarter with a directly reported three-month figure (Q1-Q3 in a 10-Q). Year-over-year growth compares it with the same quarter one year earlier (latest values).
6. Trailing twelve months (TTM) = sum of the latest four quarters ending with the most recently reported quarter, using latest values (Q4 derived by rule 4).
7. Count rule for "filings reporting the fiscal-year total": count each filing (10-K, 10-K/A) that reports the full-year value.
8. As-of rule: the cutoff is a timestamp. A filing accepted later does not exist yet.

## Fact schema (used by the knowledge graph and code-writing methods)

Every fact has six fields: subject, attribute, period, value, effective_at, kind.
Inside a subject, " | " (space, vertical bar, space) is an ordinary character separating the ticker from the concept. Never escape it.

- subject = ticker | Revenues, e.g. {SUBJECT_EX}
- attribute = value_usd_m
- period = start..end of the date range, e.g. {PERIOD_EX}
- value = the number in $ millions
- effective_at = empty
- kind = new the first time a range is reported, revision if a later filing reports a different value, corroborate if it repeats the same value
