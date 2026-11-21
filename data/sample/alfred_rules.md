# Rules: macroeconomic data vintages (ALFRED style)

Evidence: one line per published estimate of one period's value, with its release time. The same period is estimated several times (advance, second, third, annual update or benchmark revision).

1. Latest estimate of a period = the most recently published estimate for that period at the as-of time.
2. First print (advance estimate) = the earliest published estimate for that period.
3. Total revision = latest estimate minus first print, in the unit of the series.
4. Revision count = number of estimates published after the first print for that period (each later release counts, even if the value is unchanged).
5. Published periods: a period is published once its advance estimate is released. A period that has ended but has no released estimate yet cannot be answered.
6. Annual updates and benchmark revisions can revise old periods, including periods whose third estimate looked final.
7. As-of rule: the cutoff is a timestamp. A release later on the same day does not exist yet.

Note for this dataset: periods are quarters (2026Q2) or months (2026-08); for monthly series read "quarter" as "month". Release names: advance/first, second, third, then annual update, benchmark revision or revision.

## Fact schema (used by the knowledge graph and code-writing methods)

Every fact has six fields: subject, attribute, period, value, effective_at, kind.
Inside a subject, " | " (space, vertical bar, space) is an ordinary character separating the series id from its name. Never escape it.

- subject = series id | series name, e.g. ZETA_GDP | real GDP growth % SAAR
- attribute = estimate
- period = the period label, e.g. 2026Q3
- value = the number, in the series' unit
- effective_at = empty
- kind = new for the first (advance) estimate of a period, revision for every later estimate of the same period
